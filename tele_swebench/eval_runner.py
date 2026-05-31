from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Any

from .eval.localization_metrics import collect_metrics_by_model, print_model_block
from .eval.telejudge_ollama import main as telejudge_main


def run_localization(experiments_root: Path) -> dict[str, Any]:
    buf = io.StringIO()
    old_out = sys.stdout
    err: dict[str, Any] | None = None
    try:
        sys.stdout = buf
        by_model, aggregate = collect_metrics_by_model(experiments_root)
        aider_root = experiments_root / "aider"
        print(f"Task localization by model — experiments root: {experiments_root}")
        for model_name in sorted(by_model.keys()):
            m = by_model[model_name]
            print_model_block(f"Model: {model_name}", m, aider_root / model_name)
        print_model_block("ALL MODELS (pooled)", aggregate, aider_root)
    except FileNotFoundError as e:
        err = {"status": "error", "error": str(e)}
    finally:
        sys.stdout = old_out
    if err:
        return err
    return {"status": "ok", "stdout": buf.getvalue()}


def run_telejudge(
    *,
    experiments_dir: Path,
    output_dir: Path,
    logs_dir: Path,
    copilot_model: str,
    judge_model: str,
    resume: bool = True,
) -> dict[str, Any]:
    argv = [
        "telejudge",
        "--copilot_model",
        copilot_model,
        "--difficulty",
        "all",
        "--experiments_dir",
        str(experiments_dir),
        "--output_dir",
        str(output_dir),
        "--logs_dir",
        str(logs_dir),
        "--judge_model",
        judge_model,
    ]
    if resume:
        argv.append("--resume")
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    buf_out = io.StringIO()
    buf_err = io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout = buf_out
        sys.stderr = buf_err
        telejudge_main(argv)
    finally:
        sys.stdout = old_out
        sys.stderr = old_err
    return {
        "status": "ok",
        "stdout": buf_out.getvalue(),
        "stderr": buf_err.getvalue(),
    }


def write_eval_report(target: Path, payload: dict) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
