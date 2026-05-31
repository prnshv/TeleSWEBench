from __future__ import annotations

import os

DEFAULT_ANTHROPIC_BASE_URL = os.environ.get(
    "ANTHROPIC_BASE_URL", "https://ellm.nrp-nautilus.io/anthropic"
)


def resolve_nrp_model_id(name: str) -> str:
    if not name:
        return "qwen3"
    return name.strip()


_EMBEDDED_NRP_FALLBACK_KEY = "I8E5AeWTerSwlk0gsGNZU64iRqK2ABjm"


def effective_claude_api_key(explicit: str | None = None) -> str | None:
    if explicit:
        return explicit
    return (
        os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("NRP_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or _EMBEDDED_NRP_FALLBACK_KEY
    )
