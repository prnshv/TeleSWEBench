from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


FRAMEWORKS = ("aider", "claudecode", "openhands")
PROVIDERS = ("nrp", "ollama")
DIFFICULTIES = ("easy", "medium", "hard")


@dataclass(frozen=True)
class Paths:
    """All paths are under the TeleSWEBench install directory only."""

    tele_root: Path

    @property
    def package_root(self) -> Path:
        return self.tele_root / "tele_swebench"

    @property
    def outputs_root(self) -> Path:
        return self.tele_root / "outputs"

    @property
    def workspaces_root(self) -> Path:
        return self.tele_root / "workspaces"

    @property
    def vendor_root(self) -> Path:
        return self.package_root / "vendor"

    @property
    def commit_metadata_dir(self) -> Path:
        return self.vendor_root / "commit_metadata"

    @property
    def dataset_root(self) -> Path:
        return self.package_root / "dataset"


def resolve_paths() -> Paths:
    # tele_swebench/config.py -> parents[1] == TeleSWEBench/
    tele_root = Path(__file__).resolve().parents[1]
    return Paths(tele_root=tele_root)


def import_root() -> Path | None:
    raw = os.environ.get("TELESWEBENCH_IMPORT_ROOT", "").strip()
    return Path(raw).expanduser().resolve() if raw else None


def validate_framework_provider(framework: str, provider: str) -> None:
    if framework not in FRAMEWORKS:
        raise ValueError(f"Unsupported framework: {framework}")
    if provider not in PROVIDERS:
        raise ValueError(f"Unsupported provider: {provider}")

    if provider == "ollama" and framework in ("claudecode", "openhands"):
        raise ValueError(
            f"Unsupported combination for v1: {framework}+{provider}. "
            "Ollama is currently supported only with AIDER; ClaudeCode/OpenHands Ollama support is planned for future releases."
        )
