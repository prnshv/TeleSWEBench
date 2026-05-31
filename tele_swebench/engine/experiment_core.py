"""
TeleSWEBench experiment core (vendored from AIDER-NRP).

Output paths are driven by engine_scope() so all artifacts stay under a TeleSWEBench run directory.
"""

from __future__ import annotations

import hashlib
import json
import re
import os
import shutil
import subprocess
import tempfile
import threading
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import requests

ZIP_CACHE_DIR: Path = Path("/tmp")

# Defaults (overridden by engine_scope for each run)
_HERE = Path(__file__).resolve().parent
AIDER_NRP_ROOT: Path = _HERE
NEURIPS_ROOT: Path = _HERE
SR_ROOT: Path = _HERE
COPILOT_AIDER_VENV: Path = Path("/nonexistent")
COMMIT_METADATA_DIR: Path = _HERE
MATCHED_COMMITS_DIR: Path = _HERE

DIFFICULTY_TO_JSON = {
    "easy": "questions_with_test_match_easy.json",
    "medium": "questions_with_test_match_medium.json",
    "hard": "questions_with_test_match_difficult.json",
}

TEST_RESULTS_DIR: Path = _HERE / "test_results"
RESULTS_ROOT: Path = _HERE / "results"
GROUND_TRUTH_SHARED_DIR: Path = RESULTS_ROOT / "_ground_truth"
EXPERIMENTS_AIDER_ROOT: Path = _HERE / "experiments" / "aider"
EXPERIMENTS_AIDER_LOG_ROOT: Path = _HERE / "experiments" / "aider_logs"


def _run_agent_unconfigured(**kwargs: Any) -> Dict[str, Any]:
    raise RuntimeError("TeleSWEBench engine not configured; use engine_scope().")


DEFAULT_RUN_AGENT: Callable[..., Dict[str, Any]] = _run_agent_unconfigured
RUN_AGENT: Callable[..., Dict[str, Any]] = _run_agent_unconfigured


@contextmanager
def engine_scope(
    *,
    run_dir: Path,
    tele_root: Path,
    commit_metadata_dir: Path,
    zip_cache_dir: Path,
    copilot_aider_venv: Path | None = None,
    agent_runner: Callable[..., Dict[str, Any]] | None = None,
) -> Iterator[None]:
    """Pin all output roots to a single TeleSWEBench run directory."""
    global AIDER_NRP_ROOT, NEURIPS_ROOT, SR_ROOT, COPILOT_AIDER_VENV
    global COMMIT_METADATA_DIR, MATCHED_COMMITS_DIR, ZIP_CACHE_DIR
    global TEST_RESULTS_DIR, RESULTS_ROOT, GROUND_TRUTH_SHARED_DIR
    global EXPERIMENTS_AIDER_ROOT, EXPERIMENTS_AIDER_LOG_ROOT, RUN_AGENT

    backup = {
        "AIDER_NRP_ROOT": AIDER_NRP_ROOT,
        "NEURIPS_ROOT": NEURIPS_ROOT,
        "SR_ROOT": SR_ROOT,
        "COPILOT_AIDER_VENV": COPILOT_AIDER_VENV,
        "COMMIT_METADATA_DIR": COMMIT_METADATA_DIR,
        "MATCHED_COMMITS_DIR": MATCHED_COMMITS_DIR,
        "ZIP_CACHE_DIR": ZIP_CACHE_DIR,
        "TEST_RESULTS_DIR": TEST_RESULTS_DIR,
        "RESULTS_ROOT": RESULTS_ROOT,
        "GROUND_TRUTH_SHARED_DIR": GROUND_TRUTH_SHARED_DIR,
        "EXPERIMENTS_AIDER_ROOT": EXPERIMENTS_AIDER_ROOT,
        "EXPERIMENTS_AIDER_LOG_ROOT": EXPERIMENTS_AIDER_LOG_ROOT,
        "RUN_AGENT": RUN_AGENT,
    }

    AIDER_NRP_ROOT = run_dir
    NEURIPS_ROOT = tele_root
    SR_ROOT = tele_root
    COPILOT_AIDER_VENV = copilot_aider_venv if copilot_aider_venv is not None else Path("/nonexistent")
    COMMIT_METADATA_DIR = commit_metadata_dir
    MATCHED_COMMITS_DIR = tele_root / "_unused_matched_commits"
    ZIP_CACHE_DIR = zip_cache_dir
    TEST_RESULTS_DIR = run_dir / "test_results"
    RESULTS_ROOT = run_dir / "results"
    GROUND_TRUTH_SHARED_DIR = RESULTS_ROOT / "_ground_truth"
    EXPERIMENTS_AIDER_ROOT = run_dir / "experiments" / "aider"
    EXPERIMENTS_AIDER_LOG_ROOT = run_dir / "logs" / "aider_sessions"
    RUN_AGENT = agent_runner or DEFAULT_RUN_AGENT

    try:
        yield
    finally:
        AIDER_NRP_ROOT = backup["AIDER_NRP_ROOT"]
        NEURIPS_ROOT = backup["NEURIPS_ROOT"]
        SR_ROOT = backup["SR_ROOT"]
        COPILOT_AIDER_VENV = backup["COPILOT_AIDER_VENV"]
        COMMIT_METADATA_DIR = backup["COMMIT_METADATA_DIR"]
        MATCHED_COMMITS_DIR = backup["MATCHED_COMMITS_DIR"]
        ZIP_CACHE_DIR = backup["ZIP_CACHE_DIR"]
        TEST_RESULTS_DIR = backup["TEST_RESULTS_DIR"]
        RESULTS_ROOT = backup["RESULTS_ROOT"]
        GROUND_TRUTH_SHARED_DIR = backup["GROUND_TRUTH_SHARED_DIR"]
        EXPERIMENTS_AIDER_ROOT = backup["EXPERIMENTS_AIDER_ROOT"]
        EXPERIMENTS_AIDER_LOG_ROOT = backup["EXPERIMENTS_AIDER_LOG_ROOT"]
        RUN_AGENT = backup["RUN_AGENT"]


def experiment_json_path(nrp_model_id: str, difficulty: str, question_id: str) -> Path:
    safe_id = question_id.replace("/", "_")
    return EXPERIMENTS_AIDER_ROOT / nrp_model_id / difficulty / f"{safe_id}.json"


def experiment_aider_session_log_path(aider_model: str, difficulty: str, question_id: str) -> Path:
    """Path for the saved Aider session log (same layout as experiment_json_path, under aider_logs/)."""
    safe_id = question_id.replace("/", "_")
    model_slug = aider_model.split("/")[-1] if "/" in aider_model else aider_model
    return EXPERIMENTS_AIDER_LOG_ROOT / model_slug / difficulty / f"{safe_id}.log"


def write_aider_session_log_from_workspace(repo_dir: Path, dest: Path) -> None:
    """Copy prompt + stdout + stderr from aider's temp workspace into one text file."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    parts: List[str] = []
    prompt_p = repo_dir / "_aider_prompt.txt"
    out_p = repo_dir / "_aider_stdout.txt"
    err_p = repo_dir / "_aider_stderr.txt"
    if prompt_p.is_file():
        parts.append("=== AIDER PROMPT ===\n")
        parts.append(prompt_p.read_text(encoding="utf-8", errors="replace"))
        if not parts[-1].endswith("\n"):
            parts.append("\n")
    if out_p.is_file():
        parts.append("\n=== AIDER STDOUT ===\n")
        parts.append(out_p.read_text(encoding="utf-8", errors="replace"))
    if err_p.is_file() and err_p.stat().st_size > 0:
        parts.append("\n=== AIDER STDERR ===\n")
        parts.append(err_p.read_text(encoding="utf-8", errors="replace"))
    dest.write_text("".join(parts), encoding="utf-8")


LOG_NAME_PATTERNS = ("_aider_", "_copilot_", "_gemini_", "_cursor_")


def aider_executable() -> str:
    if COPILOT_AIDER_VENV.is_file():
        return str(COPILOT_AIDER_VENV)
    w = shutil.which("aider")
    if w:
        return w
    return "aider"


def load_matched_questions(difficulty: str) -> List[Dict[str, Any]]:
    if difficulty not in DIFFICULTY_TO_JSON:
        raise ValueError(f"difficulty must be one of {list(DIFFICULTY_TO_JSON)}")
    path = MATCHED_COMMITS_DIR / DIFFICULTY_TO_JSON[difficulty]
    if not path.is_file():
        raise FileNotFoundError(path)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def find_question_for_commit(commit_sha_short: str, difficulty: str) -> Tuple[Dict[str, Any], str]:
    for q in load_matched_questions(difficulty):
        if q.get("commit_sha_short") == commit_sha_short:
            return q, q["id"]
    raise KeyError(f"No {difficulty} matched question for commit {commit_sha_short}")


def download_zip_if_url(zip_path_or_url: str, local_zip_path: Path) -> Path:
    if zip_path_or_url.startswith("http://") or zip_path_or_url.startswith("https://"):
        local_zip_path.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(zip_path_or_url, stream=True, timeout=300)
        r.raise_for_status()
        with open(local_zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        return local_zip_path
    p = Path(zip_path_or_url)
    if not p.is_file():
        raise FileNotFoundError(zip_path_or_url)
    return p


def unzip_repository(zip_path: str, target_dir: Path) -> Tuple[Path, Path]:
    actual = Path(zip_path)
    if zip_path.startswith("http://") or zip_path.startswith("https://"):
        temp_zip_dir = ZIP_CACHE_DIR / "_zips"
        url_hash = hashlib.md5(zip_path.encode()).hexdigest()[:8]
        cached = temp_zip_dir / f"downloaded_{url_hash}.zip"
        if not cached.is_file():
            actual = download_zip_if_url(zip_path, cached)
        else:
            actual = cached
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True)
    with zipfile.ZipFile(actual, "r") as z:
        z.extractall(target_dir)
    items = list(target_dir.iterdir())
    if len(items) == 1 and items[0].is_dir():
        return items[0], actual
    return target_dir, actual


def initialize_git_repo(repo_dir: Path) -> bool:
    try:
        subprocess.run(["git", "init"], cwd=repo_dir, capture_output=True, check=True)
        subprocess.run(["git", "config", "user.name", "Benchmark"], cwd=repo_dir, capture_output=True, check=True)
        subprocess.run(["git", "config", "user.email", "benchmark@local"], cwd=repo_dir, capture_output=True, check=True)
        subprocess.run(["git", "config", "core.longpaths", "true"], cwd=repo_dir, capture_output=True, check=True)
        subprocess.run(["git", "add", "."], cwd=repo_dir, capture_output=True, text=True)
        subprocess.run(
            ["git", "commit", "-m", "Initial commit - baseline"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def git_diff(repo_dir: Path) -> str:
    try:
        r = subprocess.run(["git", "diff", "HEAD"], cwd=repo_dir, capture_output=True, text=True, check=True)
        return r.stdout or ""
    except subprocess.CalledProcessError:
        r = subprocess.run(["git", "diff"], cwd=repo_dir, capture_output=True, text=True)
        return r.stdout or ""


def _allowed_gt_files(question_entry: Dict[str, Any]) -> set:
    matched = question_entry.get("matched_ground_truth_files") or []
    if matched:
        return set(matched)
    return set(question_entry.get("ground_truth_diff_files") or [])


def extract_ground_truth_diff(question_entry: Dict[str, Any]) -> str:
    commit_sha_short = question_entry.get("commit_sha_short")
    if not commit_sha_short:
        return ""
    metadata_path = COMMIT_METADATA_DIR / f"{commit_sha_short}.json"
    if not metadata_path.is_file():
        return ""
    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)
    code_changes = metadata.get("code_changes", [])
    allowed = _allowed_gt_files(question_entry)
    parts: List[str] = []
    for change in code_changes:
        filename = change.get("filename")
        patch = change.get("patch") or ""
        if not filename or not patch:
            continue
        if allowed and filename not in allowed:
            continue
        parts.extend(
            [
                f"diff --git a/{filename} b/{filename}",
                f"--- a/{filename}",
                f"+++ b/{filename}",
                patch,
            ]
        )
    return "\n".join(parts)


def count_diff_files(diff_text: str) -> int:
    return len([ln for ln in diff_text.split("\n") if ln.startswith("diff --git")])


def split_unified_diff_blocks(diff_text: str) -> List[Tuple[str, str]]:
    """Split a unified diff into (file_path, block) pairs (path from ``b/...``)."""
    text = (diff_text or "").strip()
    if not text:
        return []
    pat = re.compile(r"^diff --git a/(.+?) b/(.+?)$", re.MULTILINE)
    matches = list(pat.finditer(text))
    if not matches:
        return []
    out: List[Tuple[str, str]] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end].rstrip() + "\n"
        out.append((m.group(2).strip(), block))
    return out


def diff_to_files_and_changes(diff_text: str) -> Dict[str, Any]:
    blocks = split_unified_diff_blocks(diff_text)
    files = [p for p, _ in blocks]
    changes = [{"file": p, "diff": b} for p, b in blocks]
    return {"files": files, "changes": changes}


def experiment_status_from_run(*, cli_status: str, assistant_changes: List[Any]) -> str:
    if cli_status == "timeout":
        return "timeout"
    if cli_status == "failed":
        return "failed"
    if assistant_changes:
        return "success"
    return "no_changes"


def build_experiment_json_record(
    question_entry: Dict[str, Any],
    *,
    difficulty: str,
    gt_diff: str,
    assistant_diff: str,
    meta: Dict[str, Any],
    nrp_model_id: str,
) -> Dict[str, Any]:
    """Shape aligned with CoPilot Inference/experiments/aider/easy/*.json."""
    gt = diff_to_files_and_changes(gt_diff)
    ad = diff_to_files_and_changes(assistant_diff)
    status = experiment_status_from_run(
        cli_status=meta.get("status", "unknown"),
        assistant_changes=ad["changes"],
    )
    rec: Dict[str, Any] = {
        "question_id": question_entry["id"],
        "commit_id": question_entry["commit_id"],
        "commit_sha_short": question_entry["commit_sha_short"],
        "difficulty": difficulty,
        "question": question_entry["question"],
        "nrp_model_id": nrp_model_id,
        "aider_model": meta.get("aider_model"),
        "ground_truth": gt,
        "aider": ad,
        "status": status,
        "timing": meta.get("timing", {}),
        "timestamp": meta.get("timestamp", ""),
    }
    cli = meta.get("cli")
    if isinstance(cli, dict):
        rec["cli"] = {
            "return_code": cli.get("return_code"),
            "files_changed": cli.get("files_changed"),
            "status": cli.get("status"),
        }
    if meta.get("aider_session_log"):
        rec["aider_session_log"] = meta["aider_session_log"]
    return rec


def count_real_changes(repo_dir: Path) -> int:
    r = subprocess.run(["git", "status", "--short"], cwd=repo_dir, capture_output=True, text=True)
    if r.returncode != 0:
        return 0
    n = 0
    for line in (r.stdout or "").splitlines():
        line = line.strip()
        if not line or len(line) < 4:
            continue
        status, name = line[:2], line[3:].strip()
        if not any(c in status for c in ("M", "A", "D")):
            continue
        if any(p in name for p in LOG_NAME_PATTERNS):
            continue
        if name.endswith(".log") or name.endswith("_prompt.txt"):
            continue
        n += 1
    return n


def run_aider(
    *,
    repo_dir: Path,
    question: str,
    model: str,
    timeout: Optional[int],
    env_extra: Dict[str, str],
    openai_api_base: Optional[str] = None,
    openai_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    aider_exe = aider_executable()
    cmd = [
        aider_exe,
        "--model",
        model,
        "--yes-always",
        "--message",
        question,
        "--no-auto-commits",
        "--no-stream",
        "--no-pretty",
        "--no-fancy-input",
        "--subtree-only",
        "--no-gitignore",
        "--no-show-model-warnings",
        "--no-check-update",
        "--analytics-disable",
    ]
    if openai_api_base:
        cmd.extend(["--openai-api-base", openai_api_base])
    if openai_api_key:
        cmd.extend(["--openai-api-key", openai_api_key])

    env = os.environ.copy()
    env.update(env_extra)

    stdout_path = repo_dir / "_aider_stdout.txt"
    stderr_path = repo_dir / "_aider_stderr.txt"
    (repo_dir / "_aider_prompt.txt").write_text(question, encoding="utf-8")

    with open(stdout_path, "w", encoding="utf-8") as out_f, open(stderr_path, "w", encoding="utf-8") as err_f:
        proc = subprocess.Popen(
            cmd,
            cwd=repo_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            text=True,
            env=env,
        )

        def pump(pipe, fh, prefix: str) -> None:
            try:
                for line in iter(pipe.readline, ""):
                    if line:
                        fh.write(line)
                        fh.flush()
                        print(f"   {prefix}{line.rstrip()}")
            finally:
                pipe.close()

        t1 = threading.Thread(target=pump, args=(proc.stdout, out_f, "[aider] "))
        t2 = threading.Thread(target=pump, args=(proc.stderr, err_f, "[aider-err] "))
        t1.start()
        t2.start()
        try:
            rc = proc.wait(timeout=timeout) if timeout else proc.wait()
        except subprocess.TimeoutExpired:
            proc.kill()
            rc = -1
        t1.join(timeout=5)
        t2.join(timeout=5)

    stdout_content = stdout_path.read_text(encoding="utf-8", errors="replace")
    stderr_content = stderr_path.read_text(encoding="utf-8", errors="replace")
    files_changed = count_real_changes(repo_dir)
    ok = rc == 0 or files_changed > 0
    return {
        "return_code": rc,
        "stdout_chars": len(stdout_content),
        "stderr_chars": len(stderr_content),
        "stderr_tail": stderr_content[-4000:] if stderr_content else "",
        "files_changed": files_changed,
        "status": "success" if ok else ("timeout" if rc == -1 else "failed"),
    }


DEFAULT_RUN_AGENT = run_aider
RUN_AGENT = run_aider


def _safe_slug(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s)[:80]


def run_single_experiment(
    *,
    question_entry: Dict[str, Any],
    question_id: str,
    difficulty: str,
    backend: str,
    aider_model: str,
    timeout: Optional[int],
    openai_api_base: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    ollama_base: str = "http://127.0.0.1:11434",
    patch_output_dir: Optional[Path] = None,
    embed_diff_bodies: bool = False,
    save_aider_session_log: bool = True,
    workspace_parent: Optional[Path] = None,
) -> Dict[str, Any]:
    t0 = time.time()
    safe_qid = question_id.replace("/", "_")
    if workspace_parent is not None:
        temp_root = workspace_parent / safe_qid
        if temp_root.exists():
            shutil.rmtree(temp_root, ignore_errors=True)
        temp_root.mkdir(parents=True, exist_ok=True)
        cleanup_root = False
    else:
        temp_root = Path(tempfile.mkdtemp(prefix=f"aider_nrp_{question_id}_"))
        cleanup_root = True
    try:
        before = question_entry["before_archive_path"]
        repo_dir, _ = unzip_repository(before, temp_root / "ws")
        git_ok = initialize_git_repo(repo_dir)
        gt_diff = extract_ground_truth_diff(question_entry)
        gt_files = count_diff_files(gt_diff)

        env_extra: Dict[str, str] = {}
        if backend == "ollama":
            env_extra["OLLAMA_API_BASE"] = ollama_base

        cli_t0 = time.time()
        cli_result = RUN_AGENT(
            repo_dir=repo_dir,
            question=question_entry["question"],
            model=aider_model,
            timeout=timeout,
            env_extra=env_extra,
            openai_api_base=openai_api_base if backend == "nrp" else None,
            openai_api_key=openai_api_key if backend == "nrp" else None,
        )
        cli_dt = time.time() - cli_t0

        session_log_rel: Optional[str] = None
        if save_aider_session_log:
            session_log_path = experiment_aider_session_log_path(aider_model, difficulty, question_id)
            write_aider_session_log_from_workspace(repo_dir, session_log_path)
            session_log_rel = session_log_path.relative_to(AIDER_NRP_ROOT).as_posix()

        assistant_diff = git_diff(repo_dir) if git_ok else ""
        asst_files = count_diff_files(assistant_diff)

        total_dt = time.time() - t0
        meta: Dict[str, Any] = {
            "question_id": question_id,
            "commit_id": question_entry["commit_id"],
            "commit_sha_short": question_entry["commit_sha_short"],
            "difficulty": difficulty,
            "backend": backend,
            "aider_model": aider_model,
            "openai_api_base": openai_api_base if backend == "nrp" else None,
            "ground_truth_files": gt_files,
            "assistant_files": asst_files,
            "cli": cli_result,
            "status": cli_result.get("status", "unknown"),
            "timing": {
                "cli_duration_seconds": round(cli_dt, 2),
                "total_experiment_duration_seconds": round(total_dt, 2),
                "cli_duration_formatted": f"{int(cli_dt // 60)}m {int(cli_dt % 60)}s",
                "total_experiment_duration_formatted": f"{int(total_dt // 60)}m {int(total_dt % 60)}s",
            },
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        if session_log_rel:
            meta["aider_session_log"] = session_log_rel
        if assistant_diff.strip():
            meta["assistant_diff_chars"] = len(assistant_diff)

        out_assistant = patch_output_dir if patch_output_dir is not None else TEST_RESULTS_DIR
        out_assistant.mkdir(parents=True, exist_ok=True)
        slug = question_id.replace("/", "_")

        gt_shared = GROUND_TRUTH_SHARED_DIR / difficulty / f"{slug}_ground_truth.patch"
        if gt_diff:
            gt_shared.parent.mkdir(parents=True, exist_ok=True)
            if not gt_shared.is_file():
                gt_shared.write_text(gt_diff, encoding="utf-8")
            meta["ground_truth_patch_path"] = str(gt_shared)

        if assistant_diff.strip():
            patch_name = f"{slug}_assistant_{backend}_{_safe_slug(aider_model)}.patch"
            (out_assistant / patch_name).write_text(assistant_diff, encoding="utf-8")
            meta["assistant_patch_path"] = str(out_assistant / patch_name)

        if embed_diff_bodies:
            meta["_ground_truth_diff"] = gt_diff
            meta["_assistant_diff"] = assistant_diff

        meta["workspace_repo"] = str(repo_dir)
        meta["workspace_root"] = str(temp_root)
        return meta
    finally:
        if cleanup_root:
            shutil.rmtree(temp_root, ignore_errors=True)
