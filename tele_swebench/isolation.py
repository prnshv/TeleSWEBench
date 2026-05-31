from __future__ import annotations

from pathlib import Path


def assert_path_inside(root: Path, path: Path, *, name: str = "path") -> Path:
    root_r = root.resolve()
    path_r = path.expanduser().resolve()
    try:
        path_r.relative_to(root_r)
    except ValueError as exc:
        raise ValueError(f"{name} must stay inside TeleSWEBench root {root_r}: {path_r}") from exc
    return path_r
