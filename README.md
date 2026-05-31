# TeleSWEBench

TeleSWEBench is a benchmark for **734 telecom software-engineering tasks** across three difficulty levels easy, medium, and hard. It runs agent-assisted coding tools against the srsRAN Project codebase, records patches and logs, and supports **localization** metrics plus **TeleJudge** scoring.

## Quick start

```bash
git clone https://github.com/prnshv/TeleSWEBench.git
cd TeleSWEBench
python3 -m pip install -e .
python3 -m tele_swebench.cli doctor --strict
```

If `doctor --strict` succeeds, the bundled tasks and metadata are ready to use.

## What’s included

| | Location |
|---|----------|
| 734 tasks | `tele_swebench/dataset/bundled/` (`easy.json`, `medium.json`, `hard.json`) |
| Ground-truth metadata | `tele_swebench/vendor/commit_metadata/` |
| CLI entry point | `python3 -m tele_swebench.cli` (or `TeleSWEBench` after install) |

## Concepts

- **ASE tool** = an **ASE framework** (how the agent is run) plus a **model** served by a **provider**.
- **Frameworks:** AIDER, ClaudeCode, OpenHands  
- **Providers:** NRP, Ollama  

**Supported in v1:** AIDER + NRP, AIDER + Ollama, ClaudeCode + NRP, OpenHands + NRP. ClaudeCode + Ollama and OpenHands + Ollama are planned for a later release.

## Requirements

- Python 3.10+
- `git`
- The framework you choose on your machine (e.g. `aider`, or set `TELESWEBENCH_AIDER_BIN` to its path; ClaudeCode / OpenHands need their respective CLIs where applicable)
- API access for your provider (NRP and/or Ollama as configured for your environment)
- Network access to download per-task source archives and to call model APIs

Configure API keys and base URLs via environment variables; do not commit secrets into this repo.

## Run a benchmark

Example:

```bash
python3 -m tele_swebench.cli run \
  --framework aider --provider nrp --model qwen3 \
  --difficulties easy --limit 5
```

Each run creates a directory under `outputs/runs/<run_id>/` with:

- `experiments/` — JSON records for evaluation (including full ground truth for comparison)
- `results/` — patches when enabled
- `workspaces/` — checked-out code and agent workspace per task
- `logs/` — runner and test logs

## Evaluation

```bash
python3 -m tele_swebench.cli eval --directory outputs/runs/<run_id> --mode localization
python3 -m tele_swebench.cli eval --directory outputs/runs/<run_id> --mode telejudge --copilot-model qwen3
```

TeleJudge uses Ollama (or your configured OpenAI-compatible endpoint); see [docs/evaluation.md](docs/evaluation.md).

## Executable tests

By default, the suite runs `ctest_r_exact` from each task inside that task’s workspace.

Before running `ctest`, TeleSWEBench now performs an automatic prepare step:

- Runs `cmake -S . -B build` when the workspace has no complete build tree.
- Overlays the `tests/` tree from each task's `after_archive_path` so test execution uses post-fix test cases.
- Builds only the test targets inferred from exact `ctest -R "^...$"` patterns when possible.
- Falls back to a full `cmake --build build` if target-specific build fails.

By default, tests are only executed when the assistant-changed file set exactly matches the task's full ground-truth file set (`ground_truth.files`). If files do not match, the task is marked as `skipped` in `logs/tests.json`. Use `--force-tests` to run tests for every task regardless of file-match status.

### Build/test prerequisites

The workspace still needs C/C++ dependencies available to CMake (for example, `GTest` for many unit-test targets).

- System-level install (recommended): install required dev packages via your OS package manager.
- User-space install (no sudo): install dependencies under a local prefix and point CMake to it.

## More documentation

- [docs/framework-adapters.md](docs/framework-adapters.md) — frameworks, providers, environment setup  
- [docs/testcases.md](docs/testcases.md) — `ctest_r_exact` and the bundled task format  
- [docs/evaluation.md](docs/evaluation.md) — localization and TeleJudge  
