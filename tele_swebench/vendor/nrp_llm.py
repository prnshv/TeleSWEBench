"""
NRP OpenAI-compatible gateway settings (aligned with Judge/utils/nrp_loader.py).

Env:
  NRP_API_KEY or OPENAI_API_KEY — bearer token for https://ellm.nrp-nautilus.io/v1
  NRP_BASE_URL — optional override (default below)
"""

from __future__ import annotations

import os

# Same defaults as srsRANCoPilot/Judge/utils/nrp_loader.py
DEFAULT_NRP_BASE_URL = os.environ.get("NRP_BASE_URL", "https://ellm.nrp-nautilus.io/v1")

# Model IDs as used in NRP chat completions `model` field (NRP exposes this as "gemma").
NRP_MODEL_ALIASES = {
    "gemma3-27b-it": "gemma",
    "gemma3_27b_it": "gemma",
    "gemma3:27b": "gemma",
    "gemma3-it": "gemma",
    "gemma3": "gemma",
}


def resolve_nrp_model_id(name: str) -> str:
    if not name:
        return "gemma"
    n = name.strip()
    return NRP_MODEL_ALIASES.get(n.lower(), n)


def nrp_openai_model_flag(nrp_model_id: str) -> str:
    """LiteLLM / aider model string for an OpenAI-compatible endpoint."""
    mid = resolve_nrp_model_id(nrp_model_id)
    return f"openai/{mid}"


# Last-resort fallback so a headless worker runs without env setup.
# Override with NRP_API_KEY or OPENAI_API_KEY in the environment (preferred).
_EMBEDDED_NRP_FALLBACK_KEY = "I8E5AeWTerSwlk0gsGNZU64iRqK2ABjm"


def effective_nrp_api_key(explicit: str | None = None) -> str | None:
    if explicit:
        return explicit
    return (
        os.environ.get("NRP_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or _EMBEDDED_NRP_FALLBACK_KEY
    )
