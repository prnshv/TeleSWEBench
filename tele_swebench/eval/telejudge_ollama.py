#!/usr/bin/env python3
"""
TeleJudge (Ollama): same srsRAN-aware judge pipeline as TeleJudge/tele_judge.py,
but calls a local Ollama OpenAI-compatible endpoint instead of NRP.

Default judge model: gemma4:31b (Ollama tag).

Usage:
    ollama pull gemma4:31b   # or your local equivalent tag
    python3 tele_judge.py \\
        --copilot_model gemma \\
        --difficulty all \\
        --experiments_dir "../../AIDER-NRP/experiments_full_gt" \\
        --output_dir outputs_full_gt \\
        --logs_dir outputs_logs

Environment:
    OLLAMA_BASE_URL  e.g. http://127.0.0.1:11434/v1
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import re
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from openai import OpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger("tele_judge_ollama")

DEFAULT_TIMEOUT = 600  # local 31B can be slow; 10 minutes default

_config = {"timeout": 600, "thinking": False}

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")
DEFAULT_JUDGE_MODEL = "gemma4:31b"
# OpenAI client requires a non-empty key; Ollama ignores it
DEFAULT_API_KEYS = ["ollama"]

FILE_JUDGE_SYSTEM = """\
You are an expert judge evaluating whether an AI coding assistant correctly \
addressed a software engineering task on the **srsRAN Project** 5G codebase \
(C++, telecom / 3GPP protocols).

You will receive:
1. The original task description (question).
2. The ground truth fix (the accepted human commit diff) for ONE file.
3. The copilot's proposed changes for the SAME file.

**Accept** if the copilot's changes are functionally equivalent to the ground \
truth for this file — they address the same root cause or implement the same \
feature, even if variable names, formatting, ordering, or minor stylistic \
details differ.

**Reject** if the copilot's changes miss the core issue, modify the wrong \
logic, introduce regressions, or are substantively incomplete.

Return ONLY a JSON object with exactly these fields (no other text):
{"verdict": "accept" or "reject", "confidence": 0-100, "reasons": ["...", "..."]}

confidence: 0 = pure guess, 100 = absolutely certain.
Keep each reason to at most 25 words."""

META_JUDGE_SYSTEM = """\
You are a senior reviewer making a final decision on whether an AI copilot's \
commit correctly addresses a task on the srsRAN 5G codebase.

Below you will receive:
1. The original task description.
2. Per-file verdicts from a first-pass review, each with reasons.

Read the reasons carefully, think about the commit holistically, and make \
your own judgment. A single rejected file does NOT automatically mean the \
whole commit fails — consider whether the rejected file is critical to the \
task or a minor ancillary change.

Return ONLY a JSON object with exactly these fields (no other text):
{"verdict": "accept" or "reject", "confidence": 0-100, "reasons": ["...", "..."]}

confidence: 0 = pure guess, 100 = absolutely certain.
Keep each reason to at most 25 words."""


class KeyPool:
    """Thread-safe round-robin pool of OpenAI clients (multiple keys optional)."""

    def __init__(self, api_keys: List[str], base_url: str):
        if not api_keys:
            raise ValueError("At least one API key is required (use 'ollama' for Ollama).")
        self._clients = [OpenAI(api_key=k, base_url=base_url) for k in api_keys]
        self._idx = 0
        self._lock = threading.Lock()

    @property
    def size(self) -> int:
        return len(self._clients)

    def next_client(self) -> OpenAI:
        with self._lock:
            client = self._clients[self._idx % len(self._clients)]
            self._idx += 1
            return client


def llm_call(
    pool: KeyPool,
    model: str,
    system: str,
    user: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> Dict[str, Optional[str]]:
    client = pool.next_client()
    timeout = _config["timeout"]
    result: Dict[str, Any] = {"content": None, "error": None}

    def _do_call():
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                extra_body={"think": _config["thinking"]},
            )
            result["content"] = resp.choices[0].message.content
        except Exception as exc:
            result["error"] = exc

    t = threading.Thread(target=_do_call, daemon=True)
    t.start()
    t.join(timeout=timeout)

    if t.is_alive():
        log.warning("LLM call timed out after %ds", timeout)
        return {
            "status": "timeout",
            "content": None,
            "error": f"LLM call timed out after {timeout}s",
        }
    if result["error"]:
        log.error("LLM call error: %s", result["error"])
        return {
            "status": "error",
            "content": None,
            "error": str(result["error"])[:500],
        }
    return {"status": "ok", "content": result["content"], "error": None}


def _extract_json(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    txt = raw.strip()

    for pat in [r"```json\s*\n(.*?)\n```", r"```(.*?)```"]:
        m = re.search(pat, txt, re.DOTALL)
        if m:
            txt = m.group(1).strip()
            break

    start = txt.find("{")
    end = txt.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    txt = txt[start : end + 1]

    txt = re.sub(r",\s*}", "}", txt)
    txt = re.sub(r",\s*]", "]", txt)

    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        return None


def _parse_verdict(raw: Optional[str]) -> Dict[str, Any]:
    parsed = _extract_json(raw)
    if parsed and parsed.get("verdict") in ("accept", "reject"):
        return {
            "verdict": parsed["verdict"],
            "confidence": max(0, min(100, int(parsed.get("confidence", 0)))),
            "reasons": parsed.get("reasons", []),
        }
    return {
        "verdict": "misc_error",
        "confidence": 0,
        "reasons": ["Failed to parse LLM response"],
        "failure_type": "parse_error",
        "_parse_error_raw": raw or "",
    }


def _extract_files(section: Dict[str, Any]) -> Set[str]:
    files: Set[str] = set()
    if isinstance(section, dict):
        for change in section.get("changes", []):
            if isinstance(change, dict) and change.get("file"):
                files.add(change["file"].strip())
    return files


def _extract_diff(section: Dict[str, Any], file_path: str) -> Optional[str]:
    if not isinstance(section, dict):
        return None
    for change in section.get("changes", []):
        if isinstance(change, dict) and change.get("file", "").strip() == file_path:
            diff = change.get("diff", "")
            if diff and diff.strip():
                return f"File: {file_path}\n{diff}"
    return None


def _format_all_diffs(section: Dict[str, Any]) -> str:
    parts: List[str] = []
    if isinstance(section, dict):
        for change in section.get("changes", []):
            if isinstance(change, dict) and change.get("diff", "").strip():
                parts.append(f"File: {change.get('file', '?')}\n{change['diff']}")
    return "\n\n".join(parts)


def _public_verdict(v: Dict[str, Any]) -> Dict[str, Any]:
    return {k: val for k, val in v.items() if not k.startswith("_")}


def judge_file(
    pool: KeyPool,
    model: str,
    question: str,
    gt_diff: str,
    copilot_diff: str,
    file_path: str,
) -> Dict[str, Any]:
    user_prompt = (
        f"Task description:\n{question}\n\n"
        f"Ground truth change for this file:\n{gt_diff}\n\n"
        f"Copilot change for this file:\n{copilot_diff}\n\n"
        "Return ONLY the JSON verdict."
    )

    llm_result = llm_call(pool, model, FILE_JUDGE_SYSTEM, user_prompt, max_tokens=1024)
    if llm_result["status"] != "ok":
        if llm_result["status"] == "timeout":
            return {
                "file": file_path,
                "verdict": "timeout",
                "confidence": 0,
                "failure_type": "timeout",
                "reasons": [f"LLM call timed out after {_config['timeout']}s"],
                "_llm_status": "timeout",
                "_llm_error": llm_result.get("error"),
                "_raw_response": None,
            }
        return {
            "file": file_path,
            "verdict": "misc_error",
            "confidence": 0,
            "failure_type": "api_error",
            "reasons": [f"LLM call error: {llm_result.get('error', 'unknown error')}"],
            "_llm_status": "error",
            "_llm_error": llm_result.get("error"),
            "_raw_response": None,
        }

    raw = llm_result["content"]
    result = _parse_verdict(raw)
    result["file"] = file_path
    result["_llm_status"] = "ok"
    result["_llm_error"] = None
    result["_raw_response"] = raw
    return result


def combine_verdicts(
    pool: KeyPool,
    model: str,
    question: str,
    file_verdicts: List[Dict[str, Any]],
) -> Dict[str, Any]:
    verdict_lines: List[str] = []
    for fv in file_verdicts:
        reasons_str = "; ".join(fv.get("reasons", [])[:3])
        verdict_lines.append(
            f"  File: {fv['file']}\n"
            f"    Verdict: {fv['verdict']}\n"
            f"    Confidence: {fv.get('confidence', '?')}\n"
            f"    Reasons: {reasons_str}"
        )

    user_prompt = (
        f"Task description:\n{question}\n\n"
        f"Per-file verdicts:\n" + "\n\n".join(verdict_lines) + "\n\n"
        "Think holistically and return ONLY the JSON verdict."
    )

    llm_result = llm_call(pool, model, META_JUDGE_SYSTEM, user_prompt)
    if llm_result["status"] != "ok":
        if llm_result["status"] == "timeout":
            return {
                "verdict": "timeout",
                "confidence": 0,
                "failure_type": "timeout",
                "reasons": [f"Meta-agent timed out after {_config['timeout']}s"],
                "_llm_status": "timeout",
                "_llm_error": llm_result.get("error"),
                "_raw_response": None,
            }
        return {
            "verdict": "misc_error",
            "confidence": 0,
            "failure_type": "api_error",
            "reasons": [f"Meta-agent error: {llm_result.get('error', 'unknown error')}"],
            "_llm_status": "error",
            "_llm_error": llm_result.get("error"),
            "_raw_response": None,
        }
    result = _parse_verdict(llm_result["content"])
    result["_llm_status"] = "ok"
    result["_llm_error"] = None
    result["_raw_response"] = llm_result["content"]
    return result


def judge_commit(
    pool: KeyPool,
    model: str,
    experiment: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    status = experiment.get("status", "")
    if status in ("timeout", "no_changes", "error"):
        return None

    gt = experiment.get("ground_truth", {})
    aider = experiment.get("aider", {})

    gt_files = _extract_files(gt)
    copilot_files = _extract_files(aider)
    if not gt_files or not copilot_files:
        return None

    if gt_files != copilot_files:
        return None

    question = experiment.get("question", "(no question provided)")
    llm_logs: Dict[str, Any] = {"file_calls": [], "meta_call": None}

    file_verdicts: List[Dict[str, Any]] = []
    for fp in sorted(gt_files):
        gt_diff = _extract_diff(gt, fp)
        cop_diff = _extract_diff(aider, fp)
        if not gt_diff or not cop_diff:
            file_verdicts.append({
                "file": fp,
                "verdict": "misc_error",
                "confidence": 0,
                "failure_type": "missing_diff",
                "reasons": ["Could not extract diff content"],
                "_llm_status": "skipped",
                "_llm_error": "Could not extract diff content",
                "_raw_response": None,
            })
            continue

        log.info("  Judging file: %s", fp)
        fv = judge_file(pool, model, question, gt_diff, cop_diff, fp)
        llm_logs["file_calls"].append({
            "file": fp,
            "status": fv.get("_llm_status", "unknown"),
            "failure_type": fv.get("failure_type"),
            "error": fv.get("_llm_error"),
            "raw_response": fv.get("_raw_response"),
        })
        file_verdicts.append(fv)

    if not file_verdicts:
        return None

    if len(file_verdicts) == 1:
        final = {
            "verdict": file_verdicts[0]["verdict"],
            "confidence": file_verdicts[0].get("confidence", 0),
            "reasons": file_verdicts[0].get("reasons", []),
        }
        if file_verdicts[0].get("failure_type"):
            final["failure_type"] = file_verdicts[0]["failure_type"]
    else:
        if all(fv["verdict"] == "timeout" for fv in file_verdicts):
            final = {
                "verdict": "timeout",
                "confidence": 0,
                "reasons": ["All file-level calls timed out"],
            }
        elif all(fv["verdict"] in ("timeout", "misc_error") for fv in file_verdicts):
            final = {
                "verdict": "misc_error",
                "confidence": 0,
                "failure_type": "all_file_calls_failed",
                "reasons": ["All file-level calls failed (timeout and/or misc errors)"],
            }
        else:
            log.info("  Meta-agent combining %d file verdicts", len(file_verdicts))
            final = combine_verdicts(pool, model, question, file_verdicts)
            llm_logs["meta_call"] = {
                "status": final.get("_llm_status", "unknown"),
                "failure_type": final.get("failure_type"),
                "error": final.get("_llm_error"),
                "raw_response": final.get("_raw_response"),
            }

    public_file_verdicts = [_public_verdict(fv) for fv in file_verdicts]
    public_final = _public_verdict(final)
    return {
        "question_id": experiment.get("question_id", ""),
        "commit_id": experiment.get("commit_id", ""),
        "commit_sha_short": experiment.get("commit_sha_short", ""),
        "difficulty": experiment.get("difficulty", ""),
        "copilot_model": experiment.get("nrp_model_id", ""),
        "judge_model": model,
        "verdict": public_final["verdict"],
        "confidence": public_final.get("confidence", 0),
        "reasons": public_final.get("reasons", []),
        "failure_type": public_final.get("failure_type"),
        "file_verdicts": public_file_verdicts,
        "_llm_logs": llm_logs,
        "timestamp": datetime.now().isoformat(),
    }


def _collect_work_items(
    experiments_dir: Path,
    output_dir: Path,
    copilot_model: str,
    difficulties: List[str],
    *,
    commit_id: Optional[str] = None,
    num_commits: Optional[int] = None,
    resume: bool = False,
) -> List[Tuple[Path, str, Path]]:
    items: List[Tuple[Path, str, Path]] = []
    for difficulty in difficulties:
        diff_dir = experiments_dir / "aider" / copilot_model / difficulty
        if not diff_dir.exists():
            log.warning("Directory not found, skipping: %s", diff_dir)
            continue

        json_files = sorted(diff_dir.glob("*.json"))

        if commit_id:
            json_files = [f for f in json_files if f.stem.startswith(commit_id)]
            if not json_files:
                log.warning("Commit %s not found in %s", commit_id, diff_dir)
                continue

        if num_commits:
            json_files = json_files[:num_commits]

        for jf in json_files:
            question_id = jf.stem
            out_path = output_dir / copilot_model / f"{question_id}.json"
            if resume and out_path.exists():
                continue
            items.append((jf, question_id, out_path))
    return items


def _collect_timeout_items(
    output_dir: Path,
    experiments_dir: Path,
    copilot_model: str,
) -> List[Tuple[Path, str, Path]]:
    model_out_dir = output_dir / copilot_model
    if not model_out_dir.exists():
        log.warning("No outputs found for %s", copilot_model)
        return []

    items: List[Tuple[Path, str, Path]] = []
    for out_path in sorted(model_out_dir.glob("*.json")):
        try:
            result = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        verdict = result.get("verdict")
        failure_type = (result.get("failure_type") or "").lower()
        reasons = " | ".join(result.get("reasons", []))
        reasons_lower = reasons.lower()

        is_retriable = verdict in ("timeout", "misc_error")
        if not is_retriable and verdict == "reject":
            is_retriable = (
                failure_type in ("parse_error", "api_error")
                or "failed to parse llm response" in reasons_lower
                or "llm call error" in reasons_lower
                or "meta-agent error" in reasons_lower
            )

        if not is_retriable:
            continue

        question_id = out_path.stem
        difficulty = result.get("difficulty", "")
        if not difficulty:
            parts = question_id.rsplit("_", 1)
            difficulty = parts[1] if len(parts) == 2 else ""

        if not difficulty:
            log.warning("Cannot determine difficulty for %s, skipping", question_id)
            continue

        experiment_path = experiments_dir / "aider" / copilot_model / difficulty / f"{question_id}.json"
        if not experiment_path.exists():
            log.warning("Experiment file not found: %s", experiment_path)
            continue

        items.append((experiment_path, question_id, out_path))

    log.info("Found %d retriable timeout/misc_error/legacy-error results for %s", len(items), copilot_model)
    return items


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="TeleJudge (Ollama): srsRAN-aware LLM judge via local Ollama",
    )
    parser.add_argument(
        "--copilot_model",
        type=str,
        required=True,
        help="Which copilot model subfolder to evaluate (e.g. 'gemma', 'qwen3-small')",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        choices=["easy", "medium", "hard", "all"],
        default="all",
        help="Difficulty level to process (default: all)",
    )
    parser.add_argument(
        "--experiments_dir",
        type=str,
        required=True,
        help="Root experiments directory (containing aider/{model}/{difficulty}/)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for judge verdict JSONs",
    )
    parser.add_argument(
        "--logs_dir",
        type=str,
        required=True,
        help="Directory for raw LLM response logs",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default=OLLAMA_BASE_URL,
        help=f"Ollama OpenAI-compatible base URL (default: {OLLAMA_BASE_URL})",
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        default=DEFAULT_JUDGE_MODEL,
        help=f"Ollama model tag for the judge (default: {DEFAULT_JUDGE_MODEL})",
    )
    parser.add_argument(
        "--num_commits",
        type=int,
        default=None,
        help="Max number of commits to process per difficulty (default: all)",
    )
    parser.add_argument(
        "--commit_id",
        type=str,
        default=None,
        help="Process only this specific commit SHA (short)",
    )
    parser.add_argument(
        "--api_keys",
        type=str,
        default=None,
        help="Comma-separated dummy keys for OpenAI client (default: ollama)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Per-attempt timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip commits that already have an output JSON",
    )
    parser.add_argument(
        "--fix_timeout",
        action="store_true",
        help="Re-judge only commits whose previous verdict was 'timeout' or 'misc_error'",
    )
    parser.add_argument(
        "--max_parallel",
        type=int,
        default=1,
        help="Max concurrent commits (default: 1 for single local GPU)",
    )

    args = parser.parse_args(argv)

    _config["timeout"] = args.timeout
    _config["thinking"] = False

    if args.api_keys:
        keys = [k.strip() for k in args.api_keys.split(",") if k.strip()]
    else:
        keys = list(DEFAULT_API_KEYS)
    base = args.base_url.rstrip("/")
    if not base.endswith("/v1"):
        base = base + "/v1"
    pool = KeyPool(keys, base_url=base)

    max_parallel = max(1, args.max_parallel)
    log.info(
        "Ollama judge at %s model=%s keys=%d max_parallel=%d think=%s",
        base,
        args.judge_model,
        pool.size,
        max_parallel,
        _config["thinking"],
    )

    experiments_dir = Path(args.experiments_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    logs_dir = Path(args.logs_dir).resolve()
    copilot_model = args.copilot_model

    difficulties = ["easy", "medium", "hard"] if args.difficulty == "all" else [args.difficulty]

    if args.fix_timeout:
        work_items = _collect_timeout_items(output_dir, experiments_dir, copilot_model)
    else:
        work_items = _collect_work_items(
            experiments_dir, output_dir, copilot_model, difficulties,
            commit_id=args.commit_id, num_commits=args.num_commits,
            resume=args.resume,
        )

    log.info("Collected %d work items for copilot=%s", len(work_items), copilot_model)
    if not work_items:
        print("Nothing to process.")
        return

    counters = {"processed": 0, "accept": 0, "reject": 0, "timeout": 0, "misc_error": 0, "skipped": 0}
    counter_lock = threading.Lock()
    start_time = time.time()

    def _process_one(item: Tuple[Path, str, Path]) -> None:
        jf, question_id, out_path = item
        try:
            experiment = json.loads(jf.read_text(encoding="utf-8"))
        except Exception as exc:
            log.error("Failed to read %s: %s", jf, exc)
            return

        result = judge_commit(pool, args.judge_model, experiment)
        if result is None:
            with counter_lock:
                counters["skipped"] += 1
            return

        llm_logs = result.pop("_llm_logs", None)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        if llm_logs is not None:
            logs_path = logs_dir / copilot_model / f"{question_id}.json"
            logs_path.parent.mkdir(parents=True, exist_ok=True)
            logs_payload = {
                "question_id": result.get("question_id"),
                "commit_id": result.get("commit_id"),
                "commit_sha_short": result.get("commit_sha_short"),
                "difficulty": result.get("difficulty"),
                "copilot_model": result.get("copilot_model"),
                "judge_model": result.get("judge_model"),
                "timestamp": result.get("timestamp"),
                "llm_logs": llm_logs,
            }
            logs_path.write_text(
                json.dumps(logs_payload, indent=2, ensure_ascii=False), encoding="utf-8",
            )

        v = result["verdict"]
        with counter_lock:
            counters["processed"] += 1
            if v == "accept":
                counters["accept"] += 1
            elif v == "timeout":
                counters["timeout"] += 1
            elif v == "misc_error":
                counters["misc_error"] += 1
            else:
                counters["reject"] += 1
            p = counters["processed"]
            elapsed = time.time() - start_time
            avg = elapsed / p
            print(
                f"\r  {p} done | "
                f"accept={counters['accept']} reject={counters['reject']} "
                f"timeout={counters['timeout']} misc_error={counters['misc_error']} | "
                f"avg={avg:.1f}s/commit  ({max_parallel} workers)",
                end="", flush=True,
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = [executor.submit(_process_one, item) for item in work_items]
        for fut in concurrent.futures.as_completed(futures):
            exc = fut.exception()
            if exc:
                log.error("Worker exception: %s", exc)

    print()
    elapsed_total = time.time() - start_time
    print(
        f"\nFinished: {counters['processed']} commits in {timedelta(seconds=int(elapsed_total))}\n"
        f"  accept={counters['accept']}  reject={counters['reject']}  "
        f"timeout={counters['timeout']}  misc_error={counters['misc_error']}  skipped={counters['skipped']}"
    )


if __name__ == "__main__":
    main()
