from __future__ import annotations

import os
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from .experiment_core import count_real_changes


def _claude_executable() -> str:
    w = shutil.which("claude")
    if w:
        return w
    local = Path.home() / ".local" / "bin" / "claude"
    if local.is_file():
        return str(local)
    return "claude"


def _openhands_executable() -> str:
    w = shutil.which("openhands")
    if w:
        return w
    local = Path.home() / ".local" / "bin" / "openhands"
    if local.is_file():
        return str(local)
    return "openhands"


def run_claude_code(
    *,
    repo_dir: Path,
    question: str,
    model: str,
    timeout: Optional[int],
    env_extra: Dict[str, str],
    openai_api_base: Optional[str] = None,
    openai_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    cmd = [
        _claude_executable(),
        "--bare",
        "-p",
        question,
        "--permission-mode",
        "acceptEdits",
        "--allowedTools",
        "Bash,Read,Edit,Glob,Grep",
    ]

    env = os.environ.copy()
    env.update(env_extra)
    if openai_api_base:
        env["ANTHROPIC_BASE_URL"] = openai_api_base
    if openai_api_key:
        env["ANTHROPIC_API_KEY"] = openai_api_key
    env["ANTHROPIC_DEFAULT_OPUS_MODEL"] = model
    env["ANTHROPIC_DEFAULT_SONNET_MODEL"] = model
    env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = model
    env.setdefault("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1")
    env.setdefault("CLAUDE_CODE_DISABLE_FEEDBACK_SURVEY", "1")
    env.setdefault("CLAUDE_CODE_ENABLE_TELEMETRY", "0")
    env.setdefault("API_TIMEOUT_MS", "3000000")
    env.setdefault("DISABLE_TELEMETRY", "1")

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

        t1 = threading.Thread(target=pump, args=(proc.stdout, out_f, "[claude] "))
        t2 = threading.Thread(target=pump, args=(proc.stderr, err_f, "[claude-err] "))
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


def run_openhands(
    *,
    repo_dir: Path,
    question: str,
    model: str,
    timeout: Optional[int],
    env_extra: Dict[str, str],
    openai_api_base: Optional[str] = None,
    openai_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    cmd = [
        _openhands_executable(),
        "--headless",
        "--task",
        question,
        "--override-with-envs",
    ]
    env = os.environ.copy()
    env.update(env_extra)
    env["LLM_MODEL"] = model
    if openai_api_key:
        env["LLM_API_KEY"] = openai_api_key
    if openai_api_base:
        env["LLM_BASE_URL"] = openai_api_base

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

        t1 = threading.Thread(target=pump, args=(proc.stdout, out_f, "[openhands] "))
        t2 = threading.Thread(target=pump, args=(proc.stderr, err_f, "[openhands-err] "))
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
