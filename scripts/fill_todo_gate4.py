#!/usr/bin/env python
"""Fill the Gate-4 ``todo/`` CSV skeletons (steps 37, 38, 39a, 39b, 40, 43, 44, 45, 47) into ``todo_done/``.

    uv run --no-sync python scripts/fill_todo_gate4.py --clean                 # every step
    uv run --no-sync python scripts/fill_todo_gate4.py --steps 40,45           # only those two
    uv run --no-sync python scripts/fill_todo_gate4.py --dry-run               # plan + time estimates

Every step resolves its own inputs (cached LLM rankings, stored selector refits, new
ranking calls, backbone refits), so any subset of ``--steps`` runs on its own.  All
expensive work is checkpointed under ``--work-dir`` and the run is resumable.  The
row and column layout of every skeleton is preserved; only blank cells are filled.
``fill_manifest.json`` / ``fill_log.md`` in the output folder record the source of
every cell and the reason for every blank.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for candidate in (REPO_ROOT, REPO_ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from scripts.todo_fill.common import Manifest, Skeleton, configure_log, load_dotenv_if_present, log  # noqa: E402
from scripts.todo_fill.gate4 import config  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--todo-dir", default="todo", help="folder with the CSV skeletons (default: todo)")
    parser.add_argument("--output-dir", default="todo_done", help="folder that receives the filled CSVs (default: todo_done)")
    parser.add_argument("--work-dir", default="outputs/todo_fill", help="checkpoints, cached frames, rankings, fits (default: outputs/todo_fill, git-ignored)")
    parser.add_argument("--sidecar-dir", default="", help="where the manifest, log, contract issues and per-step sidecars go (default: --output-dir); set it to keep the output folder to the deliverable CSVs alone")
    parser.add_argument("--steps", default="all", help=f"comma list of steps to run, e.g. 40,45 (default: all = {','.join(config.ALL_STEPS)})")
    parser.add_argument("--jobs", type=int, default=6, help="parallel worker processes for backbone refits; -1 = all CPU cores (default: 6)")
    parser.add_argument("--third-jobs", type=int, default=3, help="parallel workers for third-dataset refits, which need more RAM each; -1 = all cores (default: 3)")
    parser.add_argument("--clean", action="store_true", help="delete everything inside --output-dir before running")
    parser.add_argument("--candidates", default="cache", choices=["cache", "universe"], help="LLM candidate set per partition: the cached run's list (default) or the full universe")
    parser.add_argument("--third-refits", default="run", choices=["run", "skip"], help="full-universe Stability 2024 selector refits (RAM-guarded; default run)")
    parser.add_argument("--appendix-e", default="~/projects/article/appendix_e_subsets.csv", help="the paper's 12 headline subsets; used in place of a local refit wherever a cell matches (empty string disables)")
    parser.add_argument("--no-api", action="store_true", help="never call the OpenAI API (cells needing new rankings stay blank)")
    parser.add_argument("--retry-failed-calls", type=int, default=0, help="extra re-issues of a ranking call that exhausted the frozen three attempts; 0 (default) keeps the protocol exact and leaves those cells blank")
    parser.add_argument("--tier", type=int, default=3, choices=[0, 1, 2, 3], help="maximum selector cost tier to refit (default: 3 = heavy)")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and running-time estimates; run nothing")
    return parser


def _restore_prefilled(skeleton: Skeleton, prefilled: list[tuple[int, str, str]], manifest: Manifest, step: str) -> None:
    """Put back every cell that arrived already filled, and record where the run disagreed.

    The skeletons carry values from the paper.  A builder is free to compute the same
    quantity, but the delivered file keeps the value that was handed to it.
    """

    for index, column, original in prefilled:
        if index >= len(skeleton.rows):
            continue
        row = skeleton.rows[index]
        computed = row.get(column, "")
        if computed != original:
            row[column] = original
            if computed != "":
                manifest.note(step, f"row {index + 1} {column}: kept the pre-filled {original!r}; this run computed {computed!r}")


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cores = os.cpu_count() or 1
    args.jobs = cores if args.jobs < 1 else args.jobs
    args.third_jobs = cores if args.third_jobs < 1 else args.third_jobs
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    todo_dir, out_dir, work_dir = _resolve(args.todo_dir), _resolve(args.output_dir), _resolve(args.work_dir)
    sidecar_dir = _resolve(args.sidecar_dir) if args.sidecar_dir else out_dir
    steps = list(config.ALL_STEPS) if args.steps.strip().lower() == "all" else [item.strip() for item in args.steps.split(",") if item.strip()]
    unknown = [step for step in steps if step not in config.STEP_FILE]
    if unknown:
        build_parser().error(f"unknown step(s) {unknown}; choose from {list(config.ALL_STEPS)}")
    steps = [step for step in config.ALL_STEPS if step in set(steps)]  # TODO.md order

    if args.dry_run:
        from scripts.todo_fill.gate4.estimates import estimate

        print(f"{'step':5s} {'file':30s} {'API calls':>9s} {'fits':>5s} {'est. minutes':>12s}  note")
        total = 0.0
        for step in steps:
            item = estimate(step, args.jobs, args.third_refits)
            total += float(item["total_minutes"])
            print(f"{step:5s} {config.STEP_FILE[step]:30s} {item['api_calls']:>9d} {item['backbone_fits']:>5d} {float(item['total_minutes']):>12.0f}  {item['note']}")
        print(f"{'':5s} {'total':30s} {'':>9s} {'':>5s} {total:>12.0f}  (jobs={args.jobs}; estimates from unit costs measured on this machine)")
        return 0

    if args.clean and out_dir.exists():
        for child in out_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        print(f"cleaned {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    configure_log(work_dir / "fill_todo_gate4.log")
    load_dotenv_if_present()
    manifest = Manifest(output_dir=sidecar_dir)
    manifest.settings = {**vars(args), "repo_root": str(REPO_ROOT), "python": sys.version.split()[0], "cwd": os.getcwd(), "steps_resolved": steps}
    log(f"fill_todo_gate4: todo={todo_dir} out={out_dir} work={work_dir} steps={steps} jobs={args.jobs}")

    skeletons: dict[str, Skeleton] = {}
    targets: dict[str, list[str]] = {}   # the blank columns each skeleton asks this run to fill
    prefilled: dict[str, list[tuple[int, str, str]]] = {}  # cells that arrived filled and must survive
    for name, step in config.FILES.items():
        path = todo_dir / name
        if path.exists():
            skeleton = Skeleton.read(path)
            skeletons[name] = skeleton
            targets[name] = [column for column in skeleton.columns if any(row.get(column, "") == "" for row in skeleton.rows)]
            prefilled[name] = [(index, column, row[column]) for index, row in enumerate(skeleton.rows) for column in skeleton.columns if row.get(column, "") != ""]
            if step in steps or not (out_dir / name).exists():
                skeleton.write(out_dir / name)  # the output folder always holds every skeleton
            if step not in steps:
                # keep a previously filled file untouched and report its fill rate
                existing = Skeleton.read(out_dir / name)
                filled, total = existing.fill_rate(targets[name] or config.VALUE_COLUMNS.get(name, []))
                manifest.csv_summary[name] = {"filled": filled, "total": total, "status": "excluded (--steps); kept from the previous run" if filled else "excluded (--steps)"}
        elif step in steps:
            manifest.skip(step, {"file": name}, "skeleton missing from the todo folder")

    from scripts.todo_fill.data import DataContext
    from scripts.todo_fill.gate4.fits import FitExecutor
    from scripts.todo_fill.gate4.llm import Ranker
    from scripts.todo_fill.gate4.records import RecordFactory
    from scripts.todo_fill.gate4.steps import BUILDERS, Context
    from scripts.todo_fill.gate4.subsets import SubsetResolver
    from scripts.todo_fill.selections import SelectionEngine, SelectionStore

    data = DataContext(work_dir)
    engine = SelectionEngine(store=SelectionStore(work_dir), data=data, manifest=manifest, max_tier=args.tier, third_ranking=None, retry_skipped=False)
    ranker = Ranker(work_dir, manifest, enabled=False if args.no_api else None, retry_failed=args.retry_failed_calls)
    records = RecordFactory(data, work_dir, manifest, candidate_source=args.candidates)
    subsets = SubsetResolver(data=data, engine=engine, ranker=ranker, records=records, manifest=manifest, work_dir=work_dir, third_refits=args.third_refits, appendix_e=args.appendix_e or None)
    executor = FitExecutor(work_dir=work_dir, manifest=manifest, jobs=args.jobs, third_jobs=args.third_jobs)
    ctx = Context(data=data, engine=engine, executor=executor, ranker=ranker, records=records, subsets=subsets, manifest=manifest, work_dir=work_dir, out_dir=out_dir, options=vars(args), sidecar_dir=sidecar_dir)
    if not ranker.enabled:
        manifest.note("setup", "API calls disabled (no OPENAI_API_KEY or --no-api): steps 44/45/47 and the Stability 2024 rows of 39a/39b keep their blanks")

    for step in steps:
        name = config.STEP_FILE[step]
        skeleton = skeletons.get(name)
        if skeleton is None:
            continue
        started = time.perf_counter()
        log(f"==== step {step}: {name}")
        try:
            BUILDERS[step](skeleton, ctx)
        except KeyboardInterrupt:
            log("interrupted; writing what is available")
            skeleton.write(out_dir / name)
            manifest.write()
            return 130
        except Exception as exc:  # noqa: BLE001
            manifest.note(step, f"{name} failed: {type(exc).__name__}: {exc}\n{traceback.format_exc()}")
        _restore_prefilled(skeleton, prefilled.get(name, []), manifest, step)
        skeleton.write(out_dir / name)
        filled, total = skeleton.fill_rate(targets.get(name) or config.VALUE_COLUMNS.get(name, []))
        manifest.csv_summary[name] = {"filled": filled, "total": total, "status": "complete" if filled == total and total else ("partial" if filled else "empty")}
        manifest.timing(f"step:{step}", time.perf_counter() - started)
        log(f"wrote {name}: {filled}/{total} value cells filled in {(time.perf_counter() - started) / 60:.1f} min")
        manifest.write()
        engine.release()
    issues = ranker.write_contract_issues(sidecar_dir / "contract_issues.csv")
    manifest.note("setup", f"ranking calls made in this run: {ranker.calls_made}; calls that failed the frozen contract: {ranker.calls_failed}" + (f"; {issues} attempt rows in contract_issues.csv" if issues else ""))
    if ranker.calls_failed:
        manifest.note("setup", f"{ranker.calls_failed} ranking call(s) exhausted the three frozen attempts; their cells are blank. Re-running fills them if the model complies, or use --retry-failed-calls N (a disclosed deviation from the frozen protocol).")
    manifest.write()
    log(f"done; manifest at {out_dir / 'fill_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
