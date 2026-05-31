# Evaluation

All evaluation code is vendored under `tele_swebench/eval/` (localization metrics + TeleJudge Ollama).

```bash
TeleSWEBench eval --directory outputs/runs/<run_id> --mode localization|telejudge|all
```

Inputs are read only from `<run_id>/experiments/`. Outputs go to `<run_id>/evaluation/`.
