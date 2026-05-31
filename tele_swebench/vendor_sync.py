from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .config import Paths, import_root


def _load_required_sha_shorts(paths: Paths) -> set[str]:
    manifest_path = paths.dataset_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing dataset manifest at {manifest_path}. Run: TeleSWEBench dataset bundle"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required: set[str] = set()
    for rel_path in manifest.get("files", {}).values():
        fp = paths.dataset_root / rel_path
        if not fp.is_file():
            raise FileNotFoundError(f"Missing bundled dataset file: {fp}")
        rows = json.loads(fp.read_text(encoding="utf-8"))
        for row in rows:
            sha = (row.get("commit_sha_short") or "").strip()
            if sha:
                required.add(sha)
    return required


def sync_commit_metadata(paths: Paths) -> dict[str, Any]:
    """Copy only benchmark-required commit_metadata files into TeleSWEBench vendor dir."""
    imp = import_root()
    if imp is None:
        raise FileNotFoundError(
            "Set TELESWEBENCH_IMPORT_ROOT to srsRANCoPilot (parent of Benchmark Generation), "
            "then run: TeleSWEBench vendor sync-commit-metadata"
        )
    src = imp / "Benchmark Generation" / "final_codebase" / "commit_metadata"
    if not src.is_dir():
        raise FileNotFoundError(f"Missing commit metadata source: {src}")

    dst = paths.commit_metadata_dir
    dst.mkdir(parents=True, exist_ok=True)
    required = _load_required_sha_shorts(paths)

    # Remove stale metadata first, then copy only required files.
    removed = 0
    for old in dst.glob("*.json"):
        old.unlink()
        removed += 1

    copied = 0
    missing: list[str] = []
    for sha in sorted(required):
        src_file = src / f"{sha}.json"
        if not src_file.is_file():
            missing.append(sha)
            continue
        shutil.copy2(src_file, dst / src_file.name)
        copied += 1

    manifest = {
        "source": str(src),
        "destination": str(dst),
        "required_unique_commits": len(required),
        "files_copied": copied,
        "removed_stale_files": removed,
        "missing_required_files": missing,
    }
    mf = paths.vendor_root / "commit_metadata_manifest.json"
    mf.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def verify_commit_metadata(paths: Paths) -> dict[str, Any]:
    dst = paths.commit_metadata_dir
    if not dst.is_dir():
        raise FileNotFoundError(f"Missing {dst}. Run: TeleSWEBench vendor sync-commit-metadata")
    files = list(dst.glob("*.json"))
    required = _load_required_sha_shorts(paths)
    present = {p.stem for p in files}
    missing = sorted(required - present)
    return {
        "commit_metadata_dir": str(dst),
        "json_files": len(files),
        "required_unique_commits": len(required),
        "missing_required_files": missing,
        "is_complete": len(missing) == 0,
    }
