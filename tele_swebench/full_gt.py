from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .config import Paths
from .dataset import load_questions


def _split_unified_diff_blocks(diff_text: str) -> list[tuple[str, str]]:
    text = (diff_text or "").strip()
    if not text:
        return []
    pattern = re.compile(r"^diff --git a/(.+?) b/(.+?)$", re.MULTILINE)
    matches = list(pattern.finditer(text))
    if not matches:
        return []
    out: list[tuple[str, str]] = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end].rstrip() + "\n"
        out.append((match.group(2).strip(), block))
    return out


def _diff_to_files_and_changes(diff_text: str) -> dict[str, Any]:
    blocks = _split_unified_diff_blocks(diff_text)
    files = [p for p, _ in blocks]
    changes = [{"file": p, "diff": d} for p, d in blocks]
    return {"files": files, "changes": changes}


def _build_full_gt_diff(question_entry: dict[str, Any], commit_metadata_dir: Path) -> str:
    sha_short = question_entry.get("commit_sha_short")
    if not sha_short:
        return ""
    meta_path = commit_metadata_dir / f"{sha_short}.json"
    if not meta_path.is_file():
        return ""
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    code_changes = metadata.get("code_changes", [])
    allowed_files = set(question_entry.get("ground_truth_diff_files") or [])
    parts: list[str] = []
    for change in code_changes:
        filename = change.get("filename")
        patch = change.get("patch") or ""
        if not filename or not patch:
            continue
        if allowed_files and filename not in allowed_files:
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


def rewrite_experiments_to_full_gt(paths: Paths, experiments_root: Path, difficulties: list[str]) -> dict[str, int]:
    """
    Rewrite ground_truth in experiments/aider/<model>/<difficulty>/*.json
    to full GT based on commit metadata and matched benchmark index.
    """
    questions = load_questions(paths, difficulties)
    q_index = {(q.get("difficulty"), q.get("id")): q for q in questions}
    commit_metadata_dir = paths.commit_metadata_dir

    rewritten = 0
    skipped = 0
    errors = 0
    for src_path in sorted(experiments_root.glob("aider/*/*/*.json")):
        payload = json.loads(src_path.read_text(encoding="utf-8"))
        difficulty = payload.get("difficulty")
        question_id = payload.get("question_id")
        if not difficulty or not question_id:
            errors += 1
            continue
        q_entry = q_index.get((difficulty, question_id))
        if not q_entry:
            skipped += 1
            continue
        full_gt = _diff_to_files_and_changes(_build_full_gt_diff(q_entry, commit_metadata_dir))
        payload["ground_truth"] = full_gt
        src_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        rewritten += 1
    return {"rewritten": rewritten, "skipped": skipped, "errors": errors}
