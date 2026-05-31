from __future__ import annotations

import os

DEFAULT_OLLAMA_API_BASE = os.environ.get("OLLAMA_API_BASE", "http://127.0.0.1:11434")
DEFAULT_OLLAMA_MODEL = os.environ.get("AIDER_OLLAMA_MODEL", "qwen2.5-coder:1.5b")


def ollama_aider_model_flag(ollama_model: str | None = None) -> str:
    tag = (ollama_model or DEFAULT_OLLAMA_MODEL).strip()
    return f"ollama/{tag}"


def experiment_model_id(*, ollama_model: str | None = None, explicit_id: str | None = None) -> str:
    if explicit_id and explicit_id.strip():
        return explicit_id.strip()
    m = (ollama_model or DEFAULT_OLLAMA_MODEL).strip()
    return m.replace(":", "-").replace("/", "_")
