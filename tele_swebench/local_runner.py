from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .artifacts import RunArtifacts
from .config import Paths, validate_framework_provider
from .engine.agents import run_claude_code, run_openhands
from .engine.experiment_core import (
    build_experiment_json_record,
    engine_scope,
    experiment_json_path,
    run_single_experiment,
)
from .full_gt import rewrite_experiments_to_full_gt
from .isolation import assert_path_inside
from .vendor import claude_nrp_llm, nrp_llm, ollama_llm


def _select_agent(framework: str) -> Optional[Callable[..., Dict[str, Any]]]:
    if framework == "aider":
        return None
    if framework == "claudecode":
        return run_claude_code
    if framework == "openhands":
        return run_openhands
    raise ValueError(f"Unknown framework: {framework}")


def run_benchmark(
    paths: Paths,
    artifacts: RunArtifacts,
    *,
    framework: str,
    provider: str,
    model: str,
    difficulties: List[str],
    questions: List[Dict[str, Any]],
    limit: int,
    timeout: int,
    resume: bool,
    skip_patch_files: bool,
    dry_run: bool,
    log_file: Path,
) -> Dict[str, Any]:
    validate_framework_provider(framework, provider)
    assert_path_inside(paths.tele_root, artifacts.run_dir, name="run_dir")
    assert_path_inside(paths.tele_root, paths.commit_metadata_dir, name="commit_metadata_dir")

    if not paths.commit_metadata_dir.is_dir() or not any(paths.commit_metadata_dir.glob("*.json")):
        raise FileNotFoundError(
            f"Bundled commit metadata missing under {paths.commit_metadata_dir}. "
            "Run: TeleSWEBench vendor sync-commit-metadata"
        )

    zip_cache = paths.workspaces_root / "cache" / "zips"
    zip_cache.mkdir(parents=True, exist_ok=True)
    workspace_parent = artifacts.run_dir / "workspaces"
    workspace_parent.mkdir(parents=True, exist_ok=True)

    aider_bin: Optional[Path] = None
    raw_venv = os.environ.get("TELESWEBENCH_AIDER_BIN", "").strip()
    if raw_venv:
        aider_bin = Path(raw_venv).expanduser().resolve()

    agent_runner = _select_agent(framework)
    timeout_sec = None if timeout == 0 else timeout

    if dry_run:
        return {
            "status": "dry_run",
            "questions_planned": len(questions),
            "workspace_parent": str(workspace_parent),
        }

    n_written = 0
    n_skipped = 0

    with engine_scope(
        run_dir=artifacts.run_dir,
        tele_root=paths.tele_root,
        commit_metadata_dir=paths.commit_metadata_dir,
        zip_cache_dir=zip_cache,
        copilot_aider_venv=aider_bin,
        agent_runner=agent_runner,
    ), log_file.open("w", encoding="utf-8") as lf:

        for q in questions:
            qid = q["id"]
            difficulty = q.get("difficulty") or difficulties[0]
            if difficulty not in difficulties:
                continue

            openai_base: Optional[str] = None
            openai_key: Optional[str] = None
            ollama_base = os.environ.get("OLLAMA_API_BASE", ollama_llm.DEFAULT_OLLAMA_API_BASE)
            backend = "nrp"
            exp_model_id = model.strip()
            aider_flag = ""

            if framework == "aider" and provider == "nrp":
                exp_model_id = model.strip()
                aider_flag = nrp_llm.nrp_openai_model_flag(exp_model_id)
                key = nrp_llm.effective_nrp_api_key(None)
                if not key:
                    lf.write("ERROR: no NRP API key (set NRP_API_KEY or OPENAI_API_KEY)\n")
                    lf.flush()
                    return {"status": "error", "error": "missing_api_key", "written": n_written, "skipped": n_skipped}
                openai_base = os.environ.get("NRP_BASE_URL", nrp_llm.DEFAULT_NRP_BASE_URL)
                openai_key = key
            elif framework == "aider" and provider == "ollama":
                backend = "ollama"
                exp_model_id = ollama_llm.experiment_model_id(ollama_model=model, explicit_id=None)
                aider_flag = ollama_llm.ollama_aider_model_flag(model)
                openai_base = None
                openai_key = None
            elif framework == "claudecode" and provider == "nrp":
                if model.strip() != "qwen3":
                    lf.write("ERROR: claudecode+NRP supports --model qwen3 only in this release.\n")
                    lf.flush()
                    return {"status": "error", "error": "unsupported_model", "written": n_written, "skipped": n_skipped}
                exp_model_id = claude_nrp_llm.resolve_nrp_model_id("qwen3")
                aider_flag = exp_model_id
                ck = claude_nrp_llm.effective_claude_api_key(None)
                if not ck:
                    lf.write("ERROR: no Anthropic/NRP API key for ClaudeCode\n")
                    lf.flush()
                    return {"status": "error", "error": "missing_anthropic_key", "written": n_written, "skipped": n_skipped}
                openai_base = os.environ.get("ANTHROPIC_BASE_URL", claude_nrp_llm.DEFAULT_ANTHROPIC_BASE_URL)
                openai_key = ck
            elif framework == "openhands" and provider == "nrp":
                if model.strip() != "qwen3":
                    lf.write("ERROR: openhands+NRP supports --model qwen3 only in this release.\n")
                    lf.flush()
                    return {"status": "error", "error": "unsupported_model", "written": n_written, "skipped": n_skipped}
                exp_model_id = "qwen3"
                aider_flag = nrp_llm.nrp_openai_model_flag(exp_model_id)
                key = nrp_llm.effective_nrp_api_key(None)
                if not key:
                    lf.write("ERROR: no NRP API key\n")
                    lf.flush()
                    return {"status": "error", "error": "missing_api_key", "written": n_written, "skipped": n_skipped}
                openai_base = os.environ.get("NRP_BASE_URL", nrp_llm.DEFAULT_NRP_BASE_URL)
                openai_key = key
            else:
                lf.write(f"ERROR: unsupported {framework}+{provider}\n")
                lf.flush()
                return {"status": "error", "error": "unsupported_combo", "written": n_written, "skipped": n_skipped}

            out_json = experiment_json_path(exp_model_id, difficulty, qid)
            if resume and out_json.is_file():
                lf.write(f"SKIP (exists) {qid}\n")
                lf.flush()
                n_skipped += 1
                continue

            patch_dir: Optional[Path] = None
            if not skip_patch_files:
                patch_dir = artifacts.results_root / exp_model_id / difficulty
                patch_dir.mkdir(parents=True, exist_ok=True)

            lf.write(f"RUN {qid}\n")
            lf.flush()

            meta = run_single_experiment(
                question_entry=q,
                question_id=qid,
                difficulty=difficulty,
                backend=backend,
                aider_model=aider_flag,
                timeout=timeout_sec,
                openai_api_base=openai_base if backend == "nrp" else None,
                openai_api_key=openai_key if backend == "nrp" else None,
                ollama_base=ollama_base,
                patch_output_dir=patch_dir,
                embed_diff_bodies=True,
                workspace_parent=workspace_parent,
            )

            gt_diff = meta.pop("_ground_truth_diff", "")
            as_diff = meta.pop("_assistant_diff", "")
            record = build_experiment_json_record(
                q,
                difficulty=difficulty,
                gt_diff=gt_diff,
                assistant_diff=as_diff,
                meta=meta,
                nrp_model_id=exp_model_id,
            )
            out_json.parent.mkdir(parents=True, exist_ok=True)
            out_json.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
            lf.write(f"    wrote {out_json}\n")
            lf.flush()
            n_written += 1

    rewrite_experiments_to_full_gt(paths, artifacts.experiments_root, difficulties)
    return {"status": "ok", "written": n_written, "skipped": n_skipped}
