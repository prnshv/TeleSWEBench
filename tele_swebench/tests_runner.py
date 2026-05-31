from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

import requests

from .config import Paths
from .isolation import assert_path_inside


def _scrub(s: str, tele_root: Path) -> str:
    root_s = str(tele_root.resolve())
    return s.replace(root_s, "<TeleSWEBench>")


def _run_prepare_step(
    *,
    cmd: str,
    cwd: Path,
    timeout_sec: int,
    tele_root: Path,
) -> dict[str, Any]:
    start = time.time()
    env = os.environ.copy()
    local_include = Path.home() / ".local" / "include"
    local_lib = Path.home() / ".local" / "lib"
    if local_include.is_dir():
        env["CPLUS_INCLUDE_PATH"] = (
            f"{local_include}:{env['CPLUS_INCLUDE_PATH']}" if env.get("CPLUS_INCLUDE_PATH") else str(local_include)
        )
        env["CPATH"] = f"{local_include}:{env['CPATH']}" if env.get("CPATH") else str(local_include)
    if local_lib.is_dir():
        env["LIBRARY_PATH"] = f"{local_lib}:{env['LIBRARY_PATH']}" if env.get("LIBRARY_PATH") else str(local_lib)
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            shell=True,
            text=True,
            capture_output=True,
            timeout=timeout_sec,
            check=False,
            env=env,
        )
        return {
            "command": cmd,
            "status": "pass" if proc.returncode == 0 else "fail",
            "return_code": proc.returncode,
            "duration_sec": round(time.time() - start, 3),
            "stdout": _scrub((proc.stdout or "")[-4000:], tele_root),
            "stderr": _scrub((proc.stderr or "")[-4000:], tele_root),
        }
    except subprocess.TimeoutExpired:
        return {
            "command": cmd,
            "status": "timeout",
            "return_code": None,
            "duration_sec": round(time.time() - start, 3),
            "stdout": "",
            "stderr": "TimeoutExpired",
        }


def _extract_ctest_exact_targets(commands: list[str]) -> list[str]:
    """Extract exact test names from commands like: ctest -R "^name$" ..."""
    targets: list[str] = []
    seen: set[str] = set()

    for cmd in commands:
        try:
            tokens = shlex.split(cmd)
        except ValueError:
            continue

        pattern: str | None = None
        for idx, tok in enumerate(tokens):
            if tok == "-R" and idx + 1 < len(tokens):
                pattern = tokens[idx + 1]
                break
            if tok.startswith("-R") and tok != "-R":
                pattern = tok[2:]
                break

        if not pattern:
            continue
        # Keep only exact regexes (^test_name$), which map cleanly to test targets.
        m = re.fullmatch(r"\^([A-Za-z0-9_.:+-]+)\$", pattern)
        if not m:
            continue
        target = m.group(1)
        if target not in seen:
            seen.add(target)
            targets.append(target)
    return targets


def _normalize_test_command(cmd: str) -> str:
    """
    Ensure ctest commands execute against the generated build directory.

    Dataset commands commonly use `ctest -R ...` without `--test-dir`.
    """
    stripped = cmd.strip()
    if not stripped:
        return cmd
    try:
        tokens = shlex.split(stripped)
    except ValueError:
        return cmd
    if not tokens or tokens[0] != "ctest":
        return cmd
    if "--test-dir" in tokens:
        return cmd
    tail = stripped[len("ctest") :].strip()
    return f"ctest --test-dir build {tail}".strip()


def _cmake_prefix_path() -> str:
    """
    Return a CMake prefix path for dependency discovery.

    Priority:
    1) TELESWEBENCH_CMAKE_PREFIX_PATH env var
    2) ~/.local (common user-space install prefix)
    """
    env_prefix = os.environ.get("TELESWEBENCH_CMAKE_PREFIX_PATH", "").strip()
    if env_prefix:
        return env_prefix
    local_prefix = Path.home() / ".local"
    if local_prefix.is_dir():
        return str(local_prefix)
    return ""


def _gtest_cmake_hints() -> str:
    """Return explicit GTest hints for CMake when local libs are present."""
    prefix = Path.home() / ".local"
    inc = prefix / "include"
    gtest = prefix / "lib" / "libgtest.a"
    gtest_main = prefix / "lib" / "libgtest_main.a"
    if inc.is_dir() and gtest.is_file() and gtest_main.is_file():
        return (
            f" -DGTEST_INCLUDE_DIR={shlex.quote(str(inc))}"
            f" -DGTEST_LIBRARY={shlex.quote(str(gtest))}"
            f" -DGTEST_MAIN_LIBRARY={shlex.quote(str(gtest_main))}"
        )
    return ""


def _resolve_archive_path(archive_path_or_url: str, cache_dir: Path) -> Path:
    """Return a local zip path for a local file or URL."""
    p = Path(archive_path_or_url)
    if p.is_file():
        return p
    if archive_path_or_url.startswith("http://") or archive_path_or_url.startswith("https://"):
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_name = archive_path_or_url.rstrip("/").split("/")[-1] or "archive.zip"
        local_zip = cache_dir / cache_name
        if not local_zip.is_file():
            resp = requests.get(archive_path_or_url, timeout=300)
            resp.raise_for_status()
            local_zip.write_bytes(resp.content)
        return local_zip
    raise FileNotFoundError(f"Unsupported archive path: {archive_path_or_url}")


def _overlay_after_tests(repo_dir: Path, after_archive_path: str, cache_dir: Path) -> None:
    """
    Overlay test files from the after snapshot into the active workspace.

    This keeps model edits on the before snapshot while executing tests from after.
    """
    zip_path = _resolve_archive_path(after_archive_path, cache_dir)
    with tempfile.TemporaryDirectory(prefix="tele_after_tests_") as tmp:
        tmp_dir = Path(tmp)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_dir)
        roots = [p for p in tmp_dir.iterdir() if p.is_dir()]
        src_root = roots[0] if len(roots) == 1 else tmp_dir
        src_tests = src_root / "tests"
        if not src_tests.is_dir():
            return
        dst_tests = repo_dir / "tests"
        if dst_tests.exists():
            shutil.rmtree(dst_tests)
        shutil.copytree(src_tests, dst_tests)


def _prepare_workspace_for_ctest(
    *,
    repo_dir: Path,
    test_commands: list[str],
    timeout_sec: int,
    tele_root: Path,
) -> dict[str, Any]:
    """Ensure there is a CMake build tree and test targets are built."""
    jobs = max(1, os.cpu_count() or 1)
    prep_timeout = max(120, min(timeout_sec, 1800))

    steps: list[dict[str, Any]] = []
    cmake_cmd = "cmake -S . -B build"
    prefix_path = _cmake_prefix_path()
    if prefix_path:
        cmake_cmd += f" -DCMAKE_PREFIX_PATH={shlex.quote(prefix_path)}"
    cmake_cmd += _gtest_cmake_hints()
    configure_step = _run_prepare_step(
        cmd=cmake_cmd,
        cwd=repo_dir,
        timeout_sec=prep_timeout,
        tele_root=tele_root,
    )
    steps.append(configure_step)
    if configure_step["status"] == "timeout":
        return {"status": "timeout", "steps": steps, "mode": "configure"}
    if configure_step["status"] == "fail":
        return {"status": "fail", "steps": steps, "mode": "configure"}

    targets = _extract_ctest_exact_targets(test_commands)
    if targets:
        for target in targets:
            step = _run_prepare_step(
                cmd=f"cmake --build build --target {shlex.quote(target)} -j{jobs}",
                cwd=repo_dir,
                timeout_sec=prep_timeout,
                tele_root=tele_root,
            )
            step["build_target"] = target
            steps.append(step)
            if step["status"] == "timeout":
                return {"status": "timeout", "steps": steps, "mode": "targeted", "targets": targets}
            if step["status"] == "fail":
                # Fallback to full build in case target naming differs from ctest regex.
                fallback_step = _run_prepare_step(
                    cmd=f"cmake --build build -j{jobs}",
                    cwd=repo_dir,
                    timeout_sec=prep_timeout,
                    tele_root=tele_root,
                )
                fallback_step["fallback_after_target"] = target
                steps.append(fallback_step)
                if fallback_step["status"] == "timeout":
                    return {"status": "timeout", "steps": steps, "mode": "fallback_full", "targets": targets}
                if fallback_step["status"] == "fail":
                    return {"status": "fail", "steps": steps, "mode": "fallback_full", "targets": targets}
                return {"status": "prepared", "steps": steps, "mode": "fallback_full", "targets": targets}
        return {"status": "prepared", "steps": steps, "mode": "targeted", "targets": targets}

    full_step = _run_prepare_step(
        cmd=f"cmake --build build -j{jobs}",
        cwd=repo_dir,
        timeout_sec=prep_timeout,
        tele_root=tele_root,
    )
    steps.append(full_step)
    if full_step["status"] == "timeout":
        return {"status": "timeout", "steps": steps, "mode": "full"}
    if full_step["status"] == "fail":
        return {"status": "fail", "steps": steps, "mode": "full"}
    return {"status": "prepared", "steps": steps, "mode": "full"}


def _load_experiment_record(run_dir: Path, question_id: str, difficulty: str) -> dict[str, Any] | None:
    safe_qid = question_id.replace("/", "_")
    experiments_root = run_dir / "experiments"
    if not experiments_root.is_dir():
        return None

    candidates = list(experiments_root.glob(f"**/{difficulty}/{safe_qid}.json"))
    if not candidates:
        candidates = list(experiments_root.glob(f"**/{safe_qid}.json"))
    if not candidates:
        return None
    # Prefer the newest file if multiple layouts exist.
    chosen = max(candidates, key=lambda p: p.stat().st_mtime)
    try:
        import json

        return json.loads(chosen.read_text(encoding="utf-8"))
    except Exception:
        return None


def _matches_full_gt_files(exp_record: dict[str, Any]) -> bool:
    gt_files = set(exp_record.get("ground_truth", {}).get("files") or [])
    changed_files = set(exp_record.get("aider", {}).get("files") or [])
    if not gt_files:
        return False
    return gt_files == changed_files


def run_question_tests(
    paths: Paths,
    run_dir: Path,
    questions: list[dict[str, Any]],
    *,
    timeout_sec: int = 1800,
    force_test_all: bool = False,
) -> dict[str, Any]:
    """Run ctest_r_exact in each question workspace under run_dir/workspaces/<qid>/ws/..."""
    run_dir = assert_path_inside(paths.tele_root, run_dir, name="run_dir")
    workspaces = run_dir / "workspaces"
    results: list[dict[str, Any]] = []

    for row in questions:
        qid = row.get("id", "")
        difficulty = row.get("difficulty", "")
        commands = row.get("ctest_r_exact") or []
        if not commands:
            results.append({"question_id": qid, "status": "no_tests", "commands": []})
            continue

        if not force_test_all:
            exp_record = _load_experiment_record(run_dir, qid, difficulty)
            if exp_record is None:
                results.append(
                    {
                        "question_id": qid,
                        "status": "skipped",
                        "error": "missing_experiment_record",
                        "hint": "Skipped tests because experiment record was not found.",
                        "commands": [],
                    }
                )
                continue
            if not _matches_full_gt_files(exp_record):
                results.append(
                    {
                        "question_id": qid,
                        "status": "skipped",
                        "error": "full_gt_file_mismatch",
                        "hint": "Skipped tests because changed files do not match full ground-truth files.",
                        "changed_files": exp_record.get("aider", {}).get("files") or [],
                        "ground_truth_files": exp_record.get("ground_truth", {}).get("files") or [],
                        "commands": [],
                    }
                )
                continue

        safe = qid.replace("/", "_")
        base = workspaces / safe / "ws"
        repo_dir: Path | None = None
        if base.is_dir():
            inner = list(base.iterdir())
            if len(inner) == 1 and inner[0].is_dir():
                repo_dir = inner[0]
            else:
                repo_dir = base

        if repo_dir is None or not repo_dir.is_dir():
            results.append(
                {
                    "question_id": qid,
                    "status": "error",
                    "error": "workspace_not_found",
                    "commands": [],
                }
            )
            continue

        try:
            assert_path_inside(paths.tele_root, repo_dir, name="test cwd")
        except ValueError as e:
            results.append({"question_id": qid, "status": "error", "error": str(e), "commands": []})
            continue

        after_archive = row.get("after_archive_path")
        if isinstance(after_archive, str) and after_archive.strip():
            try:
                _overlay_after_tests(
                    repo_dir=repo_dir,
                    after_archive_path=after_archive,
                    cache_dir=run_dir / "cache" / "after_archives",
                )
            except Exception as e:
                results.append(
                    {
                        "question_id": qid,
                        "status": "error",
                        "error": "after_tests_overlay_failed",
                        "hint": str(e),
                        "commands": [],
                    }
                )
                continue

        prepare_result = _prepare_workspace_for_ctest(
            repo_dir=repo_dir,
            test_commands=commands,
            timeout_sec=timeout_sec,
            tele_root=paths.tele_root,
        )
        if prepare_result["status"] == "timeout":
            results.append(
                {
                    "question_id": qid,
                    "status": "timeout",
                    "error": "prepare_timeout",
                    "hint": "Workspace preparation timed out before tests could run.",
                    "prepare": prepare_result,
                    "commands": [],
                }
            )
            continue
        if prepare_result["status"] == "fail":
            results.append(
                {
                    "question_id": qid,
                    "status": "error",
                    "error": "prepare_failed",
                    "hint": "Automatic CMake configure/build failed before ctest could run.",
                    "prepare": prepare_result,
                    "commands": [],
                }
            )
            continue

        cmd_results = []
        overall = "pass"
        for raw_cmd in commands:
            cmd = _normalize_test_command(raw_cmd)
            start = time.time()
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=repo_dir,
                    shell=True,
                    text=True,
                    capture_output=True,
                    timeout=timeout_sec,
                    check=False,
                )
                out = _scrub((proc.stdout or "")[-4000:], paths.tele_root)
                err = _scrub((proc.stderr or "")[-4000:], paths.tele_root)
                if proc.returncode == 0 and "No tests were found" in (proc.stderr or ""):
                    status = "error"
                    overall = "error"
                elif proc.returncode == 0:
                    status = "pass"
                else:
                    status = "fail"
                    overall = "fail"
                cmd_results.append(
                    {
                        "command": cmd,
                        "status": status,
                        "return_code": proc.returncode,
                        "duration_sec": round(time.time() - start, 3),
                        "stdout": out,
                        "stderr": err,
                    }
                )
            except subprocess.TimeoutExpired:
                overall = "timeout"
                cmd_results.append(
                    {
                        "command": cmd,
                        "status": "timeout",
                        "return_code": None,
                        "duration_sec": round(time.time() - start, 3),
                        "stdout": "",
                        "stderr": "TimeoutExpired",
                    }
                )

        results.append(
            {
                "question_id": qid,
                "status": overall,
                "prepare": prepare_result,
                "commands": cmd_results,
            }
        )

    counts = {"pass": 0, "fail": 0, "timeout": 0, "no_tests": 0, "error": 0, "skipped": 0}
    for row in results:
        s = row["status"]
        if s in counts:
            counts[s] += 1
    return {"counts": counts, "results": results}
