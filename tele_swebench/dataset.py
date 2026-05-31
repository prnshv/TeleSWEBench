from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import DIFFICULTIES, Paths, import_root


SOURCE_FILE_BY_DIFFICULTY = {
    "easy": "questions_with_test_match_easy.json",
    "medium": "questions_with_test_match_medium.json",
    "hard": "questions_with_test_match_difficult.json",
}


@dataclass(frozen=True)
class DatasetPaths:
    root: Path
    bundled_dir: Path
    manifest_path: Path


def dataset_paths(paths: Paths) -> DatasetPaths:
    root = paths.dataset_root
    return DatasetPaths(
        root=root,
        bundled_dir=root / "bundled",
        manifest_path=root / "manifest.json",
    )


def bundle_dataset(paths: Paths) -> dict[str, Any]:
    """Copy matched-commit JSONs into TeleSWEBench. Requires TELESWEBENCH_IMPORT_ROOT."""
    imp = import_root()
    if imp is None:
        raise FileNotFoundError(
            "Set TELESWEBENCH_IMPORT_ROOT to the srsRANCoPilot (or repo) directory "
            "that contains NeurIPS2/Matched commits, then run: TeleSWEBench dataset bundle"
        )

    src_root = imp / "NeurIPS2" / "Matched commits"
    if not src_root.is_dir():
        raise FileNotFoundError(f"Expected Matched commits at {src_root}")

    dpaths = dataset_paths(paths)
    dpaths.bundled_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    total = 0

    for difficulty in DIFFICULTIES:
        src = src_root / SOURCE_FILE_BY_DIFFICULTY[difficulty]
        dst = dpaths.bundled_dir / f"{difficulty}.json"
        if not src.is_file():
            raise FileNotFoundError(f"Missing source dataset file: {src}")
        shutil.copy2(src, dst)
        rows = json.loads(dst.read_text(encoding="utf-8"))
        counts[difficulty] = len(rows)
        total += len(rows)

    manifest = {
        "name": "TeleSWEBench-734",
        "total_questions": total,
        "counts_by_difficulty": counts,
        "files": {d: f"bundled/{d}.json" for d in DIFFICULTIES},
    }
    dpaths.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def verify_dataset(paths: Paths) -> dict[str, Any]:
    dpaths = dataset_paths(paths)
    if not dpaths.manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest: {dpaths.manifest_path}. Run: TeleSWEBench dataset bundle")
    manifest = json.loads(dpaths.manifest_path.read_text(encoding="utf-8"))
    counts: dict[str, int] = {}
    total = 0
    for difficulty in DIFFICULTIES:
        fp = dpaths.root / manifest["files"][difficulty]
        if not fp.is_file():
            raise FileNotFoundError(f"Missing bundled dataset file: {fp}")
        rows = json.loads(fp.read_text(encoding="utf-8"))
        counts[difficulty] = len(rows)
        total += len(rows)
    manifest["verified_counts_by_difficulty"] = counts
    manifest["verified_total_questions"] = total
    manifest["is_expected_734"] = total == 734
    return manifest


def load_questions(paths: Paths, difficulties: list[str]) -> list[dict[str, Any]]:
    dpaths = dataset_paths(paths)
    if not dpaths.manifest_path.is_file():
        raise FileNotFoundError("Dataset not bundled. Run: TeleSWEBench dataset bundle")
    manifest = json.loads(dpaths.manifest_path.read_text(encoding="utf-8"))
    all_rows: list[dict[str, Any]] = []
    for difficulty in difficulties:
        fp = dpaths.root / manifest["files"][difficulty]
        rows = json.loads(fp.read_text(encoding="utf-8"))
        all_rows.extend(rows)
    return all_rows
