from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from credit_risk_fs.clip.statistical_schema_v2 import DESCRIPTOR_COLUMNS_V2  # noqa: E402
from credit_risk_fs.clip.v1_freeze import verify_freeze_package  # noqa: E402
from credit_risk_fs.utils.hashing import sha256_file  # noqa: E402
from credit_risk_fs.utils.io import read_json, write_json  # noqa: E402


OUTPUT_ROOT = Path("results/clip_v2")
ARCHIVE_ROOT = Path("results/clip_v2_archives")
STATE_PATH = OUTPUT_ROOT / "pipeline_state.json"
LOG_PATH = OUTPUT_ROOT / "pipeline_execution.log"
LOCK_PATH = OUTPUT_ROOT / ".pipeline.lock"
STAGE_STATUSES = {"not_started", "running", "complete_valid", "interrupted", "failed", "stale"}
EVALUATION_SPECS = [
    ("homecredit", "lr", "clip_v2"),
    ("homecredit", "lr", "clip_v2_then_mrmr"),
    ("homecredit", "catboost", "clip_v2"),
    ("homecredit", "catboost", "clip_v2_then_mrmr"),
    ("lendingclub_v2", "lr", "clip_v2"),
    ("lendingclub_v2", "lr", "clip_v2_then_mrmr"),
    ("lendingclub_v2", "catboost", "clip_v2"),
    ("lendingclub_v2", "catboost", "clip_v2_then_mrmr"),
]


def stages() -> list[dict[str, Any]]:
    py = sys.executable
    return [
        {"name": "preflight", "commands": [], "validator": validate_preflight},
        {
            "name": "statistical_view",
            "commands": [[py, "scripts/build_clip_v2_statistical_view.py", "--execute"]],
            "validator": validate_statistical_view,
        },
        {
            "name": "contrastive_data",
            "commands": [[py, "scripts/build_clip_v2_contrastive_data.py", "--execute"]],
            "validator": validate_contrastive_data,
        },
        {
            "name": "training_smoke",
            "commands": [
                [py, "scripts/train_clip_v2_encoder.py", "--dry-run"],
                [py, "scripts/train_clip_v2_encoder.py", "--seed", "11", "--smoke-test", "--execute"],
            ],
            "validator": validate_training_smoke,
        },
        {
            "name": "training",
            "commands": [[py, "scripts/train_clip_v2_encoder.py", "--all-seeds", "--execute"]],
            "validator": validate_training,
        },
        {
            "name": "selector_integration",
            "commands": [[py, "scripts/validate_clip_v2_selector_integration.py", "--execute"]],
            "validator": validate_selector_integration,
        },
        {
            "name": "downstream_evaluation",
            "commands": [
                [
                    py,
                    "scripts/run_clip_v2_final_evaluation.py",
                    "--dataset",
                    dataset,
                    "--model",
                    model,
                    "--selector",
                    selector,
                    "--execute",
                ]
                for dataset, model, selector in EVALUATION_SPECS
            ],
            "validator": validate_downstream_evaluation,
        },
        {
            "name": "aggregate_rebuild",
            "commands": [[py, "scripts/rebuild_clip_v2_evaluation_aggregates.py", "--execute"]],
            "validator": validate_aggregates,
        },
        {
            "name": "final_analysis",
            "commands": [[py, "scripts/build_clip_v2_final_analysis.py", "--execute"]],
            "validator": validate_analysis,
        },
        {
            "name": "tests",
            "commands": [[py, "-m", "pytest", "tests/clip", "-q"], [py, "-m", "pytest", "tests", "-q"]],
            "validator": validate_tests,
        },
        {
            "name": "final_audit",
            "commands": [[py, "scripts/audit_clip_v2.py", "--execute"]],
            "validator": validate_final_audit,
        },
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Orchestrate the complete CLIP-v2 pipeline.")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--fresh-start", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--from-stage", choices=[stage["name"] for stage in stages()])
    parser.add_argument("--to-stage", choices=[stage["name"] for stage in stages()])
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = select_stages(args)
    if args.status:
        print(json.dumps(status_payload(), indent=2, default=str))
        return 0
    if args.plan or not args.execute:
        print(json.dumps(plan_payload(selected, args), indent=2, default=str))
        return 0
    lock = None
    try:
        if args.fresh_start:
            archive = fresh_start_reset(execute=True)
        else:
            archive = None
        lock = acquire_lock()
        return run_selected_stages(selected, archive_path=archive)
    except KeyboardInterrupt:
        release_lock(lock)
        print("Interrupted. Preview resume with:")
        print("uv run python scripts/run_clip_v2_pipeline.py --resume")
        return 130
    finally:
        release_lock(lock)


def select_stages(args: argparse.Namespace) -> list[dict[str, Any]]:
    all_stages = stages()
    names = [stage["name"] for stage in all_stages]
    if args.resume:
        state = load_state()
        return [stage for stage in all_stages if state["stages"].get(stage["name"], {}).get("status") != "complete_valid"]
    start = names.index(args.from_stage) if args.from_stage else 0
    end = names.index(args.to_stage) if args.to_stage else len(names) - 1
    if start > end:
        raise SystemExit("--from-stage must not come after --to-stage")
    return all_stages[start : end + 1]


def plan_payload(selected: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    lock = read_lock()
    fresh_archive_preview = None
    if args.fresh_start:
        fresh_archive_preview = preview_archive()
    return {
        "mode": "plan",
        "execute": False,
        "fresh_start": bool(args.fresh_start),
        "resume": bool(args.resume),
        "selected_stage_count": len(selected),
        "stage_order": [stage["name"] for stage in selected],
        "commands": {stage["name"]: [" ".join(command) for command in stage["commands"]] for stage in selected},
        "lock": lock,
        "fresh_start_archive_preview": fresh_archive_preview,
        "real_work_requires_execute": True,
    }


def status_payload() -> dict[str, Any]:
    state = load_state()
    return {
        "mode": "status",
        "output_root": str(OUTPUT_ROOT).replace("\\", "/"),
        "state_path": str(STATE_PATH).replace("\\", "/"),
        "log_path": str(LOG_PATH).replace("\\", "/"),
        "lock": read_lock(),
        "stages": state["stages"],
    }


def run_selected_stages(selected: list[dict[str, Any]], *, archive_path: str | None) -> int:
    state = load_state()
    total = len(selected)
    total_started = time.time()
    for index, stage in enumerate(selected, start=1):
        name = stage["name"]
        if state["stages"].get(name, {}).get("status") == "complete_valid":
            log(f"[{index}/{total}] {name}: skip complete_valid")
            continue
        stage_started = time.time()
        mark_stage(state, name, status="running", started=time_now(), command=stage["commands"], archive_path=archive_path)
        write_state(state)
        log_progress(index, total, name, "start", stage_started, total_started)
        try:
            if not stage["commands"]:
                stage["validator"]()
            for command in stage["commands"]:
                run_child(command, stage_name=name)
            stage["validator"]()
            mark_stage(
                state,
                name,
                status="complete_valid",
                finished=time_now(),
                elapsed_seconds=time.time() - stage_started,
                exit_code=0,
                output_hashes=stage_hashes(name),
                completion_marker=f"{name}:validated",
                resume_eligibility=True,
            )
            write_state(state)
            log_progress(index, total, name, "complete_valid", stage_started, total_started)
        except KeyboardInterrupt:
            mark_stage(state, name, status="interrupted", finished=time_now(), elapsed_seconds=time.time() - stage_started, failure_reason="KeyboardInterrupt")
            write_state(state)
            raise
        except Exception as exc:
            mark_stage(state, name, status="failed", finished=time_now(), elapsed_seconds=time.time() - stage_started, failure_reason=str(exc), resume_eligibility=True)
            write_state(state)
            log(f"{name}: failed: {exc}")
            print(f"Stage failed: {name}. Preview resume with: uv run python scripts/run_clip_v2_pipeline.py --resume")
            return 1
    return 0


def run_child(command: list[str], *, stage_name: str) -> None:
    log(f"{stage_name}: command start: {' '.join(command)}")
    process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        assert process.stdout is not None
        for line in process.stdout:
            text = line.rstrip()
            log(f"{stage_name}: {text}")
            print(text, flush=True)
        exit_code = process.wait()
    except KeyboardInterrupt:
        process.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.terminate()
        raise
    if exit_code != 0:
        raise RuntimeError(f"command failed with exit code {exit_code}: {' '.join(command)}")
    log(f"{stage_name}: command exit 0")


def validate_preflight() -> None:
    freeze = verify_freeze_package()
    if freeze["status"] != "passed":
        raise RuntimeError("CLIP-v1 freeze verification failed")
    required = [
        "configs/clip_v2/statistical_view.yaml",
        "configs/clip_v2/contrastive_data.yaml",
        "configs/clip_v2/training.yaml",
        "configs/clip_v2/selector.yaml",
        "scripts/build_clip_v2_statistical_view.py",
        "scripts/run_clip_v2_final_evaluation.py",
    ]
    missing = [path for path in required if not Path(path).exists()]
    if missing:
        raise RuntimeError(f"missing required files: {missing}")
    if len(DESCRIPTOR_COLUMNS_V2) != 13:
        raise RuntimeError("CLIP-v2 schema dimension is not 13")
    if not Path("data/homecredit").exists() or not Path("data/lendingclub_v2").exists():
        raise RuntimeError("dataset roots are missing")


def validate_statistical_view() -> None:
    order = read_json(OUTPUT_ROOT / "statistical_view" / "statistical_feature_order.json")
    pre = read_json(OUTPUT_ROOT / "statistical_view" / "statistical_preprocessor.json")
    if int(order.get("vector_dimension", -1)) != 13:
        raise RuntimeError("statistical view dimension mismatch")
    if order.get("field_order") != DESCRIPTOR_COLUMNS_V2:
        raise RuntimeError("statistical descriptor order mismatch")
    if pre.get("fit_dataset") != "homecredit" or pre.get("fit_split") != "train":
        raise RuntimeError("statistical scaler was not fitted on Home Credit train only")
    for path in [OUTPUT_ROOT / "statistical_view" / "homecredit_statistical_vectors.parquet", OUTPUT_ROOT / "statistical_view" / "lendingclub_v2_statistical_vectors.parquet"]:
        if not path.exists():
            raise RuntimeError(f"missing statistical vectors: {path}")


def validate_contrastive_data() -> None:
    manifest = read_json(OUTPUT_ROOT / "contrastive_data" / "contrastive_pair_manifest.json")
    schema = read_json(OUTPUT_ROOT / "contrastive_data" / "contrastive_tensor_schema.json")
    if int(schema.get("statistical_vector_dimension", -1)) != 13:
        raise RuntimeError("contrastive tensor schema dimension mismatch")
    counts = manifest.get("pair_counts", {})
    if min(int(counts.get(key, 0)) for key in ["homecredit_train_positive", "homecredit_validation_positive", "lendingclub_v2_external_positive"]) <= 0:
        raise RuntimeError("contrastive pair counts are incomplete")


def validate_training_smoke() -> None:
    if not (OUTPUT_ROOT / "training" / "smoke_test").exists():
        raise RuntimeError("smoke-test output directory missing")


def validate_training() -> None:
    selection = read_json(OUTPUT_ROOT / "training" / "model_selection_manifest.json")
    if selection.get("lendingclub_v2_used_for_selection"):
        raise RuntimeError("LendingClub v2 influenced CLIP-v2 model selection")
    if not (OUTPUT_ROOT / "training" / "selected_checkpoint.pt").exists():
        raise RuntimeError("selected checkpoint missing")
    seeds = selection.get("all_seed_results", [])
    if len(seeds) < 5:
        raise RuntimeError("not all configured seeds completed")


def validate_selector_integration() -> None:
    for name in ["homecredit_clip_v2_scores.csv", "lendingclub_v2_clip_v2_scores.csv"]:
        if not (OUTPUT_ROOT / "selector_integration" / name).exists():
            raise RuntimeError(f"selector cache missing: {name}")


def validate_downstream_evaluation() -> None:
    for dataset, model, selector in EVALUATION_SPECS:
        run_id = f"{dataset}_{model}_{selector}"
        if not (OUTPUT_ROOT / "final_evaluation" / "runs" / run_id / "RUN_COMPLETE.json").exists():
            raise RuntimeError(f"missing completed evaluation run: {run_id}")
        if not (OUTPUT_ROOT / "final_evaluation" / "predictions" / f"{run_id}.parquet").exists():
            raise RuntimeError(f"missing prediction file: {run_id}")


def validate_aggregates() -> None:
    validation = read_json(OUTPUT_ROOT / "final_evaluation" / "aggregate_validation.json")
    if validation.get("complete") is not True or int(validation.get("run_count", 0)) != 8:
        raise RuntimeError("aggregate validation is incomplete")


def validate_analysis() -> None:
    summary = read_json(OUTPUT_ROOT / "final_analysis" / "analysis_summary.json")
    if summary.get("status") != "complete":
        raise RuntimeError("analysis summary is not complete")
    for path in ["results/clip_v2/final_analysis/selected_feature_semantic_map_data.csv", "results/clip_v2/final_analysis/plots/06_clip_v1_vs_v2_feature_semantic_map.png"]:
        if not Path(path).exists():
            raise RuntimeError(f"semantic map artifact missing: {path}")


def validate_tests() -> None:
    return None


def validate_final_audit() -> None:
    # The child command is the authoritative final gate.
    return None


def fresh_start_reset(*, execute: bool) -> str | None:
    if active_clip_v2_processes():
        raise RuntimeError("active CLIP-v2 process detected; refusing fresh-start reset")
    preview = preview_archive()
    if not execute:
        return None
    if has_complete_audited_study():
        raise RuntimeError("complete audited CLIP-v2 study exists; refusing fresh-start reset")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    archive = ARCHIVE_ROOT / timestamp
    archive.mkdir(parents=True, exist_ok=True)
    entries = archive_generated_outputs(archive)
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    create_clean_output_root()
    write_json(
        archive / "reset_manifest.json",
        {
            "created_at": time_now(),
            "archive_path": str(archive).replace("\\", "/"),
            "archived_file_count": len(entries),
            "entries": entries,
        },
    )
    return str(archive).replace("\\", "/")


def preview_archive() -> dict[str, Any]:
    files = []
    if OUTPUT_ROOT.exists():
        files.extend(path for path in OUTPUT_ROOT.rglob("*") if path.is_file())
    files.extend(Path("reports").glob("clip_v2_*"))
    return {"file_count": len(files), "paths": [str(path).replace("\\", "/") for path in files[:20]]}


def archive_generated_outputs(archive: Path) -> list[dict[str, Any]]:
    entries = []
    if OUTPUT_ROOT.exists():
        output_files = [path for path in OUTPUT_ROOT.rglob("*") if path.is_file()]
        output_original_paths = {path.relative_to(OUTPUT_ROOT): path for path in output_files}
        destination = archive / "results_clip_v2"
        shutil.move(str(OUTPUT_ROOT), str(destination))
        for relative_path, original_path in output_original_paths.items():
            archived_path = destination / relative_path
            if archived_path.is_file():
                entries.append(archive_entry(archived_path, destination, original_path=original_path))
    report_files = list(Path("reports").glob("clip_v2_*"))
    if report_files:
        report_dest = archive / "reports"
        report_dest.mkdir(parents=True, exist_ok=True)
        for path in report_files:
            original_path = path
            target = report_dest / path.name
            shutil.move(str(path), str(target))
            entries.append(archive_entry(target, report_dest, original_path=original_path))
    return entries


def archive_entry(path: Path, archive_base: Path, *, original_path: Path) -> dict[str, Any]:
    return {
        "original_relative_path": str(original_path).replace("\\", "/"),
        "archive_relative_path": str(path.relative_to(archive_base)).replace("\\", "/"),
        "file_size": path.stat().st_size,
        "modification_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(path.stat().st_mtime)),
        "sha256": sha256_file(path),
        "reason_archived": "fresh-start reset of partial CLIP-v2 generated output",
        "appeared_complete_or_partial": "completion_marker_or_summary_present" if path.name in {"RUN_COMPLETE.json", "aggregate_validation.json", "analysis_summary.json"} else "partial_or_intermediate_artifact",
    }


def create_clean_output_root() -> None:
    for directory in ["statistical_view", "contrastive_data", "training", "selector_integration", "final_evaluation", "final_analysis", "audit"]:
        (OUTPUT_ROOT / directory).mkdir(parents=True, exist_ok=True)


def active_clip_v2_processes() -> list[dict[str, Any]]:
    if os.name != "nt":
        return []
    script = "Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,Name,CommandLine | ConvertTo-Json -Depth 2"
    result = subprocess.run(["powershell", "-NoProfile", "-Command", script], capture_output=True, text=True, check=False)
    if result.returncode != 0 or not result.stdout.strip():
        return []
    data = json.loads(result.stdout)
    rows = data if isinstance(data, list) else [data]
    ignored_pids = current_process_tree_pids(rows)
    return [row for row in rows if is_active_clip_v2_process(row, ignored_pids)]


def current_process_tree_pids(rows: list[dict[str, Any]]) -> set[int]:
    parent_by_pid = {
        int(row.get("ProcessId", 0)): int(row.get("ParentProcessId", 0) or 0)
        for row in rows
        if row.get("ProcessId") is not None
    }
    ignored = {os.getpid()}
    pid = os.getpid()
    while parent_by_pid.get(pid):
        pid = parent_by_pid[pid]
        if pid in ignored:
            break
        ignored.add(pid)
    return ignored


def is_active_clip_v2_process(row: dict[str, Any], ignored_pids: set[int]) -> bool:
    pid = int(row.get("ProcessId", 0) or 0)
    if pid in ignored_pids:
        return False
    command = str(row.get("CommandLine") or "").lower().replace("\\", "/")
    if "get-ciminstance win32_process" in command:
        return False
    if "scripts/run_clip_v2_pipeline.py" in command:
        return "--execute" in command
    active_scripts = [
        "scripts/build_clip_v2_statistical_view.py",
        "scripts/build_clip_v2_contrastive_data.py",
        "scripts/train_clip_v2_encoder.py",
        "scripts/validate_clip_v2_selector_integration.py",
        "scripts/run_clip_v2_final_evaluation.py",
        "scripts/rebuild_clip_v2_evaluation_aggregates.py",
        "scripts/build_clip_v2_final_analysis.py",
        "scripts/audit_clip_v2.py",
    ]
    return any(script in command for script in active_scripts)


def has_complete_audited_study() -> bool:
    summary = OUTPUT_ROOT / "final_analysis" / "analysis_summary.json"
    audit_reports = Path("reports/clip_v2_reproducibility_manifest.json")
    return summary.exists() and audit_reports.exists()


def acquire_lock() -> dict[str, Any]:
    create_clean_output_root()
    existing = read_lock()
    if existing and existing.get("active"):
        raise RuntimeError(f"active pipeline lock exists: {existing}")
    lock = {"pid": os.getpid(), "started_at": time_now(), "command": " ".join(sys.argv), "cwd": str(ROOT)}
    write_json(LOCK_PATH, lock)
    return lock


def read_lock() -> dict[str, Any] | None:
    if not LOCK_PATH.exists():
        return None
    payload = read_json(LOCK_PATH)
    payload["active"] = is_pid_active(int(payload.get("pid", -1)))
    return payload


def is_pid_active(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def release_lock(lock: dict[str, Any] | None) -> None:
    if lock and LOCK_PATH.exists():
        current = read_json(LOCK_PATH)
        if int(current.get("pid", -1)) == os.getpid():
            LOCK_PATH.unlink()


def load_state() -> dict[str, Any]:
    names = [stage["name"] for stage in stages()]
    if STATE_PATH.exists():
        state = read_json(STATE_PATH)
    else:
        state = {"created_at": time_now(), "stages": {}}
    for name in names:
        state["stages"].setdefault(name, {"status": "not_started", "resume_eligibility": True})
    return state


def mark_stage(state: dict[str, Any], name: str, **updates: Any) -> None:
    stage = dict(state["stages"].get(name, {}))
    if updates.get("status") in {"running", "complete_valid"}:
        stage.pop("failure_reason", None)
    stage.update(updates)
    if stage["status"] not in STAGE_STATUSES:
        raise RuntimeError(f"invalid stage status: {stage['status']}")
    state["stages"][name] = stage


def write_state(state: dict[str, Any]) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_json(STATE_PATH, state)


def stage_hashes(name: str) -> dict[str, str]:
    roots = {
        "statistical_view": OUTPUT_ROOT / "statistical_view",
        "contrastive_data": OUTPUT_ROOT / "contrastive_data",
        "training": OUTPUT_ROOT / "training",
        "selector_integration": OUTPUT_ROOT / "selector_integration",
        "downstream_evaluation": OUTPUT_ROOT / "final_evaluation",
        "aggregate_rebuild": OUTPUT_ROOT / "final_evaluation",
        "final_analysis": OUTPUT_ROOT / "final_analysis",
    }
    root = roots.get(name)
    if root is None or not root.exists():
        return {}
    return {str(path).replace("\\", "/"): sha256_file(path) for path in sorted(root.rglob("*")) if path.is_file()}


def time_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log(message: str) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"{time_now()} {message}\n")
        handle.flush()


def log_progress(index: int, total: int, name: str, status: str, stage_started: float, total_started: float) -> None:
    message = (
        f"[{index}/{total}] stage={name} status={status} "
        f"stage_elapsed={time.time() - stage_started:.1f}s total_elapsed={time.time() - total_started:.1f}s "
        f"log={LOG_PATH.as_posix()}"
    )
    log(message)
    print(message, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
