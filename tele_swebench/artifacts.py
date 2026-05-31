from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunArtifacts:
    run_dir: Path
    logs_dir: Path
    evaluation_dir: Path

    @property
    def config_path(self) -> Path:
        return self.run_dir / "config.json"

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "manifest.json"

    @property
    def summary_path(self) -> Path:
        return self.run_dir / "summary.json"

    @property
    def experiments_root(self) -> Path:
        return self.run_dir / "experiments"

    @property
    def results_root(self) -> Path:
        return self.run_dir / "results"


def create_run_artifacts(outputs_root: Path, framework: str, provider: str, model: str) -> RunArtifacts:
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_dir = outputs_root / "runs" / f"{stamp}_{framework}_{provider}_{model.replace('/', '_')}"
    logs_dir = run_dir / "logs"
    evaluation_dir = run_dir / "evaluation"
    for path in (run_dir, logs_dir, evaluation_dir):
        path.mkdir(parents=True, exist_ok=True)
    return RunArtifacts(run_dir=run_dir, logs_dir=logs_dir, evaluation_dir=evaluation_dir)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
