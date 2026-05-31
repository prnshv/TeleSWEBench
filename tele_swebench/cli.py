from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .artifacts import create_run_artifacts, write_json
from .config import DIFFICULTIES, resolve_paths, validate_framework_provider
from .dataset import bundle_dataset, load_questions, verify_dataset
from .eval_runner import run_localization, run_telejudge, write_eval_report
from .isolation import assert_path_inside
from .local_runner import run_benchmark
from .tests_runner import run_question_tests
from .vendor_sync import sync_commit_metadata, verify_commit_metadata


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="TeleSWEBench", description="TeleSWEBench standalone benchmark framework")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run ASE tool benchmark (artifacts only under TeleSWEBench/)")
    run.add_argument("--framework", required=True, choices=["aider", "claudecode", "openhands"])
    run.add_argument("--provider", required=True, choices=["nrp", "ollama"])
    run.add_argument("--model", required=True, help="Model id for provider/framework")
    run.add_argument("--difficulties", default="easy,medium,hard")
    run.add_argument("--limit", type=int, default=0)
    run.add_argument("--timeout", type=int, default=3600)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--skip-tests", action="store_true")
    run.add_argument(
        "--force-tests",
        action="store_true",
        help="Run tests for every question even when changed files do not match full ground-truth files.",
    )
    run.add_argument("--skip-patch-files", action="store_true")
    run.add_argument("--dry-run", action="store_true")

    ev = sub.add_parser("eval", help="Run localization / TeleJudge on a run directory")
    ev.add_argument("--directory", required=True, help="Run directory under outputs/runs/")
    ev.add_argument("--mode", required=True, choices=["localization", "telejudge", "all"])
    ev.add_argument("--judge-model", default="gemma4:31b")
    ev.add_argument("--copilot-model", default="", help="Subfolder under experiments/aider/")

    data = sub.add_parser("dataset", help="Dataset operations")
    data_sub = data.add_subparsers(dest="dataset_cmd", required=True)
    data_sub.add_parser("bundle")
    data_sub.add_parser("verify")

    vendor = sub.add_parser("vendor", help="Vendor bundled metadata into TeleSWEBench")
    vendor_sub = vendor.add_subparsers(dest="vendor_cmd", required=True)
    vendor_sub.add_parser("sync-commit-metadata")
    vendor_sub.add_parser("verify-commit-metadata")

    doc = sub.add_parser("doctor", help="Sanity checks")
    doc.add_argument("--strict", action="store_true", help="Assert dataset + commit metadata are present")
    return p


def cmd_run(args: argparse.Namespace) -> int:
    paths = resolve_paths()
    validate_framework_provider(args.framework, args.provider)
    difficulties = [d.strip() for d in args.difficulties.split(",") if d.strip()]
    for d in difficulties:
        if d not in DIFFICULTIES:
            raise ValueError(f"Invalid difficulty: {d}")

    manifest = verify_dataset(paths)
    artifacts = create_run_artifacts(paths.outputs_root, args.framework, args.provider, args.model)
    assert_path_inside(paths.tele_root, artifacts.run_dir, name="run_dir")

    questions = load_questions(paths, difficulties)
    if args.limit > 0:
        questions = questions[: args.limit]

    run_cfg = {
        "framework": args.framework,
        "provider": args.provider,
        "model": args.model,
        "difficulties": difficulties,
        "limit": args.limit,
        "timeout": args.timeout,
        "resume": args.resume,
        "skip_tests": args.skip_tests,
        "force_tests": args.force_tests,
        "skip_patch_files": args.skip_patch_files,
        "dry_run": args.dry_run,
        "ground_truth_mode": "full",
    }
    write_json(artifacts.config_path, run_cfg)
    write_json(
        artifacts.manifest_path,
        {
            "dataset_manifest": manifest,
            "selected_questions": len(questions),
            "selected_difficulties": difficulties,
        },
    )

    runner_result = run_benchmark(
        paths,
        artifacts,
        framework=args.framework,
        provider=args.provider,
        model=args.model,
        difficulties=difficulties,
        questions=questions,
        limit=args.limit,
        timeout=args.timeout,
        resume=args.resume,
        skip_patch_files=args.skip_patch_files,
        dry_run=args.dry_run,
        log_file=artifacts.logs_dir / "runner.log",
    )

    tests_result: dict = {"status": "skipped"}
    if not args.skip_tests and not args.dry_run:
        tests_result = run_question_tests(
            paths,
            artifacts.run_dir,
            questions,
            timeout_sec=min(args.timeout, 1800) if args.timeout > 0 else 1800,
            force_test_all=args.force_tests,
        )
        write_json(artifacts.logs_dir / "tests.json", tests_result)

    summary = {
        "run_dir": str(artifacts.run_dir),
        "runner": runner_result,
        "tests": tests_result,
    }
    write_json(artifacts.summary_path, summary)
    print(json.dumps(summary, indent=2))
    st = runner_result.get("status")
    return 0 if st in ("ok", "dry_run") else 1


def cmd_eval(args: argparse.Namespace) -> int:
    paths = resolve_paths()
    run_dir = Path(args.directory).expanduser().resolve()
    run_dir = assert_path_inside(paths.tele_root, run_dir, name="--directory")
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    out = {}
    if args.mode in ("localization", "all"):
        out["localization"] = run_localization(run_dir / "experiments")
    if args.mode in ("telejudge", "all"):
        copilot_model = args.copilot_model
        if not copilot_model:
            cfg_path = run_dir / "config.json"
            if cfg_path.is_file():
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                copilot_model = cfg.get("model", "")
        if not copilot_model:
            raise ValueError("Need --copilot-model (or model in run config) for telejudge mode.")
        out["telejudge"] = run_telejudge(
            experiments_dir=run_dir / "experiments",
            output_dir=run_dir / "evaluation" / "telejudge_outputs",
            logs_dir=run_dir / "logs" / "telejudge",
            copilot_model=copilot_model,
            judge_model=args.judge_model,
            resume=True,
        )

    write_eval_report(run_dir / "evaluation" / "report.json", out)
    print(json.dumps(out, indent=2))
    ok = all(v.get("status") == "ok" for v in out.values())
    return 0 if ok else 1


def cmd_dataset(args: argparse.Namespace) -> int:
    paths = resolve_paths()
    if args.dataset_cmd == "bundle":
        print(json.dumps(bundle_dataset(paths), indent=2))
        return 0
    if args.dataset_cmd == "verify":
        print(json.dumps(verify_dataset(paths), indent=2))
        return 0
    raise ValueError("Unknown dataset command")


def cmd_vendor(args: argparse.Namespace) -> int:
    paths = resolve_paths()
    if args.vendor_cmd == "sync-commit-metadata":
        print(json.dumps(sync_commit_metadata(paths), indent=2))
        return 0
    if args.vendor_cmd == "verify-commit-metadata":
        print(json.dumps(verify_commit_metadata(paths), indent=2))
        return 0
    raise ValueError("Unknown vendor command")


def cmd_doctor(args: argparse.Namespace) -> int:
    paths = resolve_paths()
    payload: dict = {
        "tele_root": str(paths.tele_root),
        "dataset_ok": False,
        "commit_metadata_ok": False,
    }
    try:
        verify_dataset(paths)
        payload["dataset_ok"] = True
    except Exception as e:
        payload["dataset_error"] = str(e)
    try:
        v = verify_commit_metadata(paths)
        payload["commit_metadata_ok"] = bool(v.get("is_complete"))
        payload["commit_metadata_files"] = v.get("json_files")
        payload["required_unique_commits"] = v.get("required_unique_commits")
        payload["missing_required_files"] = v.get("missing_required_files")
    except Exception as e:
        payload["commit_metadata_error"] = str(e)

    if args.strict and (not payload["dataset_ok"] or not payload["commit_metadata_ok"]):
        print(json.dumps(payload, indent=2))
        return 1
    print(json.dumps(payload, indent=2))
    return 0


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        if args.cmd == "run":
            code = cmd_run(args)
        elif args.cmd == "eval":
            code = cmd_eval(args)
        elif args.cmd == "dataset":
            code = cmd_dataset(args)
        elif args.cmd == "vendor":
            code = cmd_vendor(args)
        elif args.cmd == "doctor":
            code = cmd_doctor(args)
        else:
            raise ValueError(f"Unknown command: {args.cmd}")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        code = 1
    raise SystemExit(code)


if __name__ == "__main__":
    main()
