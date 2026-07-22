from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from credit_risk_fs.experiments.atomic_io import write_json_atomic  # noqa: E402
from credit_risk_fs.experiments.resource_policy import (  # noqa: E402
    DEFAULT_POLICY_PATH,
    detect_hardware,
    estimate_run_size,
    load_execution_policy,
    resolve_execution_policy,
    run_preflight,
)
from credit_risk_fs.experiments.result_paths import resolve_results_root  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Low-cost resource-safe execution preflight.")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--temp-root", type=Path, default=None)
    parser.add_argument("--accelerator", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--allow-gpu-without-telemetry", action="store_true")
    parser.add_argument("--requested-run-directory", type=Path, default=None)
    parser.add_argument("--input-rows", type=int, default=None)
    parser.add_argument(
        "--column-dtype-bytes",
        default=None,
        help="Comma-separated declared byte widths, for example 8,8,4,1.",
    )
    parser.add_argument("--method-memory-multiplier", type=float, default=None)
    parser.add_argument("--prediction-rows", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("cleanup/audits/resource_safe_execution/hardware_preflight.json"),
    )
    return parser


def _print_summary(report: dict[str, object]) -> None:
    detected = report["detected_capacity"]
    resolved = report["resolved_policy"]
    gpu = detected["gpu"]
    print(f"Preflight: {str(report['status']).upper()}")
    print(
        f"CPU: {detected['physical_cpu_count']} physical / "
        f"{detected['logical_cpu_count']} logical"
    )
    print(
        f"RAM: {detected['total_ram_gb']:.2f} GiB total / "
        f"{detected['available_ram_gb']:.2f} GiB available; "
        f"process warn/abort={resolved['memory']['warn_process_tree_rss_gb']:.2f}/"
        f"{resolved['memory']['abort_process_tree_rss_gb']:.2f} GiB"
    )
    if gpu["available"]:
        print(
            f"GPU: {gpu['name'] or 'detected'}; "
            f"VRAM total/free={gpu['total_vram_gb']}/{gpu['free_vram_gb']} GiB; "
            f"process telemetry={gpu['process_telemetry_available']}"
        )
    else:
        print(f"GPU: unavailable; telemetry error={gpu['error']}")
    print(
        f"Disk free: results={detected['results_free_disk_gb']:.2f} GiB, "
        f"temp={detected['temp_free_disk_gb']:.2f} GiB"
    )
    parallel = resolved["parallelism"]
    print(
        "Parallelism: "
        f"runs={parallel['concurrent_experiment_runs']}, "
        f"folds={parallel['concurrent_folds']}, "
        f"data_workers={parallel['data_loader_workers']}, "
        f"estimator_threads={parallel['estimator_threads']}, "
        f"nested={parallel['allow_nested_parallelism']}"
    )
    if report["blocking_reasons"]:
        print("Blocking checks: " + ", ".join(report["blocking_reasons"]))


def main(argv: list[str] | None = None) -> int:
    import psutil

    process = psutil.Process()
    rss_start = int(process.memory_info().rss)
    args = build_parser().parse_args(argv)
    root = args.root.resolve()
    active = resolve_results_root(root, args.results_root)
    temp = (args.temp_root or Path(tempfile.gettempdir())).resolve()
    configured = load_execution_policy(root, args.config)
    capacity = detect_hardware(active, temp)
    resolved = resolve_execution_policy(configured, capacity)
    estimate = None
    if args.input_rows is not None or args.column_dtype_bytes is not None:
        if args.input_rows is None or args.column_dtype_bytes is None:
            raise SystemExit("--input-rows and --column-dtype-bytes must be supplied together")
        widths = [int(item.strip()) for item in args.column_dtype_bytes.split(",") if item.strip()]
        estimate = estimate_run_size(
            row_count=args.input_rows,
            column_dtype_bytes=widths,
            method_memory_multiplier=args.method_memory_multiplier,
            prediction_row_count=args.prediction_rows,
            policy=resolved,
        )
    report = run_preflight(
        repository_root=root,
        config_path=args.config,
        results_root=args.results_root,
        temp_root=temp,
        requested_accelerator=args.accelerator,
        allow_gpu_without_telemetry=args.allow_gpu_without_telemetry,
        estimate=estimate,
        requested_run_directory=args.requested_run_directory,
        capacity=capacity,
    )
    from credit_risk_fs.experiments.tracking import _peak_ram_mb

    peak_ram_mb, peak_source = _peak_ram_mb()
    rss_end = int(process.memory_info().rss)
    report["preflight_resource_usage"] = {
        "rss_start_bytes": rss_start,
        "rss_end_bytes": rss_end,
        "peak_rss_bytes": (
            int(peak_ram_mb * 1024 * 1024) if peak_ram_mb is not None else max(rss_start, rss_end)
        ),
        "peak_measurement": peak_source,
    }
    output = args.output if args.output.is_absolute() else root / args.output
    write_json_atomic(output, report)
    _print_summary(report)
    print(f"Report: {output.resolve()}")
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
