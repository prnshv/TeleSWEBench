# Framework runners

Execution is **in-process** via `tele_swebench/engine/experiment_core.py` (vendored AIDER experiment logic) with `engine_scope()` pinning all outputs to the current run directory.

- **AIDER** — default `RUN_AGENT` = `run_aider`
- **ClaudeCode** — `RUN_AGENT` = Claude Code CLI (`tele_swebench/engine/agents.py`)
- **OpenHands** — `RUN_AGENT` = OpenHands CLI

Model/provider flags match the previous NeurIPS workers (`nrp_llm`, `claude_nrp_llm`, `ollama_llm` under `tele_swebench/vendor/`).
