#!/usr/bin/env python3
"""
Task localization metrics from AIDER-NRP experiment JSON files.

Computes file-set match categories (same rules as Judge/initial_filter_judge.py):
  EM — Exact Match: predicted file set P equals ground-truth set T
  PM — Partial Match: some but not all ground-truth files appear in P
  OA — Over Addressed: all ground-truth files in P, plus extra files
  NC — No Changes: no predicted files (P empty) and run finished as no_changes or timed out
       (timeout with no file edits). Timeout with files in P uses EM/PM/OA/NM from P vs T.
  NM — No Match: P vs T is a no_match and the run is not classified as NC (e.g. wrong files,
       or no edits with a non–no_changes/non-timeout status).

Rates are count / N where N is the number of experiment JSON files scanned.
One block per model (subdirectory of experiments/aider/), plus an ALL MODELS summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def default_experiments_dir() -> Path:
    return Path.cwd() / "experiments"


def extract_files_from_ground_truth(experiment_data: Dict[str, Any]) -> Set[str]:
    ground_truth = experiment_data.get("ground_truth", {})
    if not ground_truth:
        return set()
    files: Set[str] = set()
    if isinstance(ground_truth, dict) and "files" in ground_truth:
        files_list = ground_truth["files"]
        if isinstance(files_list, list):
            files.update(f.strip() for f in files_list if f and isinstance(f, str))
    if isinstance(ground_truth, dict) and "changes" in ground_truth:
        changes = ground_truth["changes"]
        if isinstance(changes, list):
            for change in changes:
                if isinstance(change, dict) and "file" in change:
                    fp = change["file"]
                    if fp and isinstance(fp, str):
                        files.add(fp.strip())
    return files


def extract_files_from_aider(experiment_data: Dict[str, Any]) -> Set[str]:
    files: Set[str] = set()
    aider_output = experiment_data.get("aider", {})
    if aider_output and isinstance(aider_output, dict):
        if "files" in aider_output:
            fl = aider_output["files"]
            if isinstance(fl, list):
                files.update(f.strip() for f in fl if f and isinstance(f, str))
        if "changes" in aider_output:
            changes = aider_output["changes"]
            if isinstance(changes, list):
                for change in changes:
                    if isinstance(change, dict) and "file" in change:
                        fp = change["file"]
                        if fp and isinstance(fp, str):
                            files.add(fp.strip())
    if not files:
        copilot_output = experiment_data.get("copilot", {})
        if isinstance(copilot_output, dict):
            if "files" in copilot_output:
                fl = copilot_output["files"]
                if isinstance(fl, list):
                    files.update(f.strip() for f in fl if f and isinstance(f, str))
            if "changes" in copilot_output:
                changes = copilot_output["changes"]
                if isinstance(changes, list):
                    for change in changes:
                        if isinstance(change, dict) and "file" in change:
                            fp = change["file"]
                            if fp and isinstance(fp, str):
                                files.add(fp.strip())
    return files


def categorize_file_match(gt_files: Set[str], pred_files: Set[str]) -> str:
    if not gt_files:
        if pred_files:
            return "over_addressed"
        return "exact_match"
    if not pred_files:
        return "no_match"
    intersection = gt_files & pred_files
    gt_only = gt_files - pred_files
    copilot_only = pred_files - gt_files
    if gt_files == pred_files:
        return "exact_match"
    if not gt_only and copilot_only:
        return "over_addressed"
    if intersection and gt_only:
        return "partial_match"
    return "no_match"


def assign_localization_label(gt_files: Set[str], pred_files: Set[str], status: str) -> str:
    """
    Single bucket per question: exact_match | partial_match | over_addressed | no_match | nc.

    If the agent touched no files and the run is no_changes or timeout, count as NC.
    If the agent touched files (including under timeout), use file overlap vs ground truth only.
    """
    s = (status or "").strip().lower()
    if not pred_files and s in ("no_changes", "timeout"):
        return "nc"
    return categorize_file_match(gt_files, pred_files)


def _process_json_paths(
    paths: Iterable[Path],
    *,
    status_filter: Optional[str] = None,
    exclude_status: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Aggregate localization labels over JSON experiment files.

    If ``status_filter`` is set (e.g. ``\"timeout\"``), only files whose top-level
    ``status`` matches (case-insensitive) are included in ``n`` and the buckets.

    If ``exclude_status`` is set (e.g. ``\"timeout\"``), files with that status are
    skipped entirely (not counted in ``n`` or any bucket). Applied before
    ``status_filter``.
    """
    want_status = (status_filter or "").strip().lower() or None
    skip_status = (exclude_status or "").strip().lower() or None
    counts = {
        "exact_match": 0,
        "partial_match": 0,
        "over_addressed": 0,
        "no_match": 0,
        "nc": 0,
        "nc_no_changes": 0,
        "nc_timeout": 0,
    }
    n = 0
    errors = 0

    for path in paths:
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            errors += 1
            continue

        status = data.get("status", "")
        s = (status or "").strip().lower()
        if skip_status is not None and s == skip_status:
            continue
        if want_status is not None and s != want_status:
            continue

        n += 1
        gt = extract_files_from_ground_truth(data)
        pred = extract_files_from_aider(data)
        label = assign_localization_label(gt, pred, status)
        counts[label] += 1
        if label == "nc":
            s = (status or "").strip().lower()
            if s == "no_changes":
                counts["nc_no_changes"] += 1
            elif s == "timeout":
                counts["nc_timeout"] += 1

    return {
        "n": n,
        "n_em": counts["exact_match"],
        "n_pm": counts["partial_match"],
        "n_oa": counts["over_addressed"],
        "n_nc": counts["nc"],
        "n_nc_no_changes": counts["nc_no_changes"],
        "n_nc_timeout": counts["nc_timeout"],
        "n_nm": counts["no_match"],
        "errors": errors,
    }


def _merge_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = (
        "n",
        "n_em",
        "n_pm",
        "n_oa",
        "n_nc",
        "n_nc_no_changes",
        "n_nc_timeout",
        "n_nm",
        "errors",
    )
    out = {k: 0 for k in keys}
    for m in rows:
        for k in keys:
            out[k] += m[k]
    return out


def collect_metrics_by_model(experiments_root: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    aider_root = experiments_root / "aider"
    if not aider_root.is_dir():
        raise FileNotFoundError(f"Missing aider results directory: {aider_root}")

    model_dirs = sorted(p for p in aider_root.iterdir() if p.is_dir())
    by_model: Dict[str, Dict[str, Any]] = {}

    for model_dir in model_dirs:
        name = model_dir.name
        json_paths = sorted(model_dir.rglob("*.json"))
        by_model[name] = _process_json_paths(json_paths)

    aggregate = _merge_metrics(list(by_model.values())) if by_model else {
        "n": 0,
        "n_em": 0,
        "n_pm": 0,
        "n_oa": 0,
        "n_nc": 0,
        "n_nc_no_changes": 0,
        "n_nc_timeout": 0,
        "n_nm": 0,
        "errors": 0,
    }
    return by_model, aggregate


def pct(num: int, den: int) -> float:
    return 100.0 * num / den if den else 0.0


def print_model_block(label: str, m: Dict[str, Any], base_path: Path) -> None:
    n = m["n"]
    print(f"\n{label} (N={n} under {base_path})")
    print(f"  EM (Exact Match):     {pct(m['n_em'], n):6.2f}%  ({m['n_em']})")
    print(f"  PM (Partial Match):   {pct(m['n_pm'], n):6.2f}%  ({m['n_pm']})")
    print(f"  OA (Over Addressed):  {pct(m['n_oa'], n):6.2f}%  ({m['n_oa']})")
    print(f"  NM (No Match):        {pct(m['n_nm'], n):6.2f}%  ({m['n_nm']})")
    print(f"  NC (No Changes):      {pct(m['n_nc'], n):6.2f}%  ({m['n_nc']})")
    nnc = m["n_nc_no_changes"]
    nnt = m["n_nc_timeout"]
    print(
        f"      no_changes         {pct(nnc, n):6.2f}%  ({nnc})"
        f"          timeout        {pct(nnt, n):6.2f}%  ({nnt})"
    )
    if m.get("errors"):
        print(f"  (skipped {m['errors']} unreadable JSON file(s))")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Task localization rates (EM, PM, OA, NM, NC) per model from experiment JSON files."
    )
    parser.add_argument(
        "--experiments",
        type=Path,
        default=default_experiments_dir(),
        help="Path to AIDER-NRP experiments directory (default: ../AIDER-NRP/experiments)",
    )
    parser.add_argument(
        "--no-aggregate",
        action="store_true",
        help="Do not print the ALL MODELS summary line.",
    )
    args = parser.parse_args()
    root = args.experiments.expanduser().resolve()
    aider_root = root / "aider"

    try:
        by_model, aggregate = collect_metrics_by_model(root)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1

    if not by_model:
        print(f"No model subdirectories found under {aider_root}", file=sys.stderr)
        return 1

    total_errors = aggregate["errors"]
    if total_errors:
        print(f"warning: skipped {total_errors} unreadable JSON file(s) total", file=sys.stderr)

    print(f"Task localization by model — experiments root: {root}")
    for model_name in sorted(by_model.keys()):
        m = by_model[model_name]
        print_model_block(f"Model: {model_name}", m, aider_root / model_name)

    if not args.no_aggregate:
        print_model_block("ALL MODELS (pooled)", aggregate, aider_root)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
