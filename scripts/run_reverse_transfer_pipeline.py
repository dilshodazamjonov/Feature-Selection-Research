from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
CLI_SCRIPT = "scripts/run_corrected_lendingclub_to_homecredit_transfer.py"
CONFIG_DIR = "configs/corrected_lendingclub_to_homecredit"
OUTPUT_DIR = "results/corrected_lendingclub_to_homecredit_transfer"
LOG_ROOT = "logs/corrected_lendingclub_to_homecredit_transfer"
SEEDS = "11,22,33,44,55"
MODELS = "lr,catboost"
REGISTRY_GATE_FAILURE_EXIT_CODE = 4
FAILURE_TAIL_LINE_COUNT = 30


@dataclass(frozen=True)
class Stage:
    name: str
    log_name: str
    command: tuple[str, ...]


@dataclass(frozen=True)
class StageResult:
    stage: Stage
    exit_code: int
    log_path: Path
    combined_output: bytes
    elapsed_seconds: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the fixed corrected LendingClub v2 to Home Credit "
            "reverse-transfer stages sequentially."
        )
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--dry-run-only",
        action="store_true",
        help="Run only the approved preflight dry-run.",
    )
    modes.add_argument(
        "--skip-registry-commit",
        action="store_true",
        help="Run through registry validation but do not commit registries.",
    )
    return parser


def stage_command(stage: str, *, dry_run: bool = False) -> tuple[str, ...]:
    command = [
        sys.executable,
        str(CLI_SCRIPT),
        "--stage",
        stage,
        "--config-dir",
        str(CONFIG_DIR),
        "--output-dir",
        str(OUTPUT_DIR),
        "--seeds",
        SEEDS,
        "--models",
        MODELS,
    ]
    if dry_run:
        command.append("--dry-run")
    return tuple(command)


def build_stages() -> tuple[Stage, ...]:
    return (
        Stage(
            "01_preflight",
            "01_preflight.log",
            stage_command("all", dry_run=True),
        ),
        Stage("02_prepare", "02_prepare.log", stage_command("prepare")),
        Stage("03_train", "03_train.log", stage_command("train")),
        Stage("04_project", "04_project.log", stage_command("project")),
        Stage("05_evaluate", "05_evaluate.log", stage_command("evaluate")),
        Stage(
            "06_register_dry_run",
            "06_register_dry_run.log",
            stage_command("register", dry_run=True),
        ),
        Stage(
            "07_register_commit",
            "07_register_commit.log",
            stage_command("register"),
        ),
    )


def validate_prerequisites() -> None:
    cli_path = ROOT / CLI_SCRIPT
    config_path = ROOT / CONFIG_DIR
    if not cli_path.is_file():
        raise FileNotFoundError(f"Underlying stage CLI is missing: {cli_path}")
    if not config_path.is_dir():
        raise FileNotFoundError(
            f"Reverse-transfer configuration directory is missing: {config_path}"
        )
    if not Path(sys.executable).is_file():
        raise FileNotFoundError(
            f"Active Python interpreter is missing: {sys.executable}"
        )


def format_command(command: tuple[str, ...]) -> str:
    return subprocess.list2cmdline(list(command))


def write_terminal_bytes(data: bytes) -> None:
    terminal = getattr(sys.stdout, "buffer", None)
    if terminal is not None:
        terminal.write(data)
        terminal.flush()
    else:
        sys.stdout.write(data.decode(errors="replace"))
        sys.stdout.flush()


def run_stage(stage: Stage, log_directory: Path) -> StageResult:
    log_path = (log_directory / stage.log_name).resolve()
    mirror_output = stage.name == "01_preflight"
    print(f"\n[{stage.name}] Starting...", flush=True)
    print(f"[{stage.name}] Command: {format_command(stage.command)}", flush=True)
    print(f"[{stage.name}] Log: {log_path}", flush=True)

    captured = bytearray()
    exit_code = 127
    started = time.monotonic()
    with log_path.open("wb") as log_file:
        try:
            process = subprocess.Popen(
                list(stage.command),
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,
                shell=False,
            )
            assert process.stdout is not None
            while True:
                chunk = process.stdout.read(8192)
                if not chunk:
                    break
                captured.extend(chunk)
                log_file.write(chunk)
                log_file.flush()
                if mirror_output:
                    write_terminal_bytes(chunk)
            exit_code = process.wait()
        except OSError:
            failure = traceback.format_exc().encode("utf-8", errors="replace")
            captured.extend(failure)
            log_file.write(failure)
            log_file.flush()
            if mirror_output:
                write_terminal_bytes(failure)

    return StageResult(
        stage=stage,
        exit_code=exit_code,
        log_path=log_path,
        combined_output=bytes(captured),
        elapsed_seconds=time.monotonic() - started,
    )


def registry_dry_run_approval(combined_output: bytes) -> tuple[bool, str]:
    text = combined_output.decode("utf-8", errors="replace")
    lowered = text.lower()
    failure_markers = (
        "hash mismatch",
        "invalid schema",
        "schema validation failed",
        "failed referential integrity",
        "referential integrity failed",
    )
    marker = next((value for value in failure_markers if value in lowered), None)
    if marker is not None:
        return False, f"registry validation output contains {marker!r}"

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return False, f"registry dry-run output is not valid JSON: {exc}"

    validation = payload.get("registry_dry_run")
    if not isinstance(validation, dict):
        return False, "registry dry-run output omitted registry validation results"

    outcome = validation.get("transaction_outcome")
    missing_artifacts = validation.get("missing_artifacts") or []
    if outcome == "CONFLICT":
        return False, "registry dry-run reported CONFLICT"
    if missing_artifacts:
        return False, "registry dry-run reported missing artifacts"
    if validation.get("writes_performed") is not False:
        return False, "registry dry-run did not explicitly confirm zero writes"
    if validation.get("success_transaction_manifest_written") is not False:
        return False, "registry dry-run unexpectedly wrote a transaction manifest"
    if outcome not in {"NEW_TRANSACTION", "IDEMPOTENT_NO_OP"}:
        return False, f"registry dry-run outcome is not acceptable: {outcome!r}"
    return True, str(outcome)


def append_gate_failure(log_path: Path, reason: str) -> None:
    message = f"\nRunner registry gate failure: {reason}\n".encode("utf-8")
    with log_path.open("ab") as log_file:
        log_file.write(message)
    write_terminal_bytes(message)


def last_meaningful_error(combined_output: bytes) -> str:
    lines = [
        line.strip()
        for line in combined_output.decode("utf-8", errors="replace").splitlines()
        if line.strip()
    ]
    if not lines:
        return "No error output was captured."
    error_tokens = ("error", "exception", "failed", "failure", "traceback")
    for line in reversed(lines):
        lowered = line.lower()
        if any(token in lowered for token in error_tokens):
            return line
    return lines[-1]


def print_log_tail(log_path: Path, *, line_count: int) -> None:
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    print(f"--- Final {line_count} log lines ---", file=sys.stderr)
    for line in lines[-line_count:]:
        print(line, file=sys.stderr)
    print("--- End log tail ---", file=sys.stderr)


def print_pipeline_failure(
    result: StageResult,
    exit_code: int,
    *,
    reason: str | None = None,
) -> None:
    failure_reason = reason or last_meaningful_error(result.combined_output)
    print("Pipeline failed", file=sys.stderr)
    print(f"Failed stage: {result.stage.name}", file=sys.stderr)
    print(f"Exit code: {exit_code}", file=sys.stderr)
    print(f"Reason: {failure_reason}", file=sys.stderr)
    print(f"Full log: {result.log_path}", file=sys.stderr)
    print_log_tail(result.log_path, line_count=FAILURE_TAIL_LINE_COUNT)
    print(
        "Next step: inspect and repair the failed stage before rerunning.",
        file=sys.stderr,
    )


def execute_stages(stages: tuple[Stage, ...], log_directory: Path) -> int:
    for stage in stages:
        result = run_stage(stage, log_directory)
        if result.exit_code != 0:
            print_pipeline_failure(result, result.exit_code)
            return result.exit_code

        if stage.name == "06_register_dry_run":
            approved, detail = registry_dry_run_approval(result.combined_output)
            if not approved:
                append_gate_failure(result.log_path, detail)
                print_pipeline_failure(
                    result,
                    REGISTRY_GATE_FAILURE_EXIT_CODE,
                    reason=detail,
                )
                return REGISTRY_GATE_FAILURE_EXIT_CODE
            print(f"Registry dry-run approved: {detail}", flush=True)
        print(
            f"[{stage.name}] Completed successfully in "
            f"{result.elapsed_seconds:.1f} seconds.",
            flush=True,
        )
    return 0


def select_stages(
    stages: tuple[Stage, ...],
    *,
    dry_run_only: bool,
    skip_registry_commit: bool,
) -> tuple[Stage, ...]:
    if dry_run_only:
        return stages[:1]
    if skip_registry_commit:
        return stages[:6]
    return stages


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        validate_prerequisites()
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    all_stages = build_stages()
    selected_stages = select_stages(
        all_stages,
        dry_run_only=args.dry_run_only,
        skip_registry_commit=args.skip_registry_commit,
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    log_directory = (ROOT / LOG_ROOT / timestamp).resolve()
    log_directory.mkdir(parents=True, exist_ok=False)

    exit_code = execute_stages(selected_stages, log_directory)
    if exit_code != 0:
        return exit_code

    if args.dry_run_only:
        print("Preflight dry-run completed successfully.")
    elif args.skip_registry_commit:
        print(
            "Reverse-transfer stages and registry dry-run completed successfully; "
            "registry commit was skipped."
        )
    else:
        print("Reverse-transfer pipeline completed successfully.")
        print("Next step: Prompt 3 read-only post-run audit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
