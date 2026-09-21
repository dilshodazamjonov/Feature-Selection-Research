#!/usr/bin/env python
"""Fill the ``todo/`` CSV skeletons into ``todo_done/``.

    uv run python scripts/fill_todo.py
    uv run python scripts/fill_todo.py --steps copy,01,06,09,17   # without 08/10/11/12

The command copies the three computationally expensive files verbatim
(``15_cohort.csv``, ``16_l1.csv``, ``21_brier.csv``), then fills the other eight
from (1) frozen evidence in the audit layer, (2) the cached LLM rankings, and
(3) selector refits under the frozen protocols where nothing frozen exists.
Everything is resumable: selector fits, LLM responses and model fits are
checkpointed under the work directory.  ``fill_manifest.json`` and
``fill_log.md`` in the output folder record the source of every cell and the
reason for every blank.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for candidate in (REPO_ROOT, REPO_ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from scripts.todo_fill.common import (  # noqa: E402
    BUDGETS,
    COPY_VERBATIM,
    DATASETS,
    HOMECREDIT,
    LENDINGCLUB,
    THIRD,
    Manifest,
    Skeleton,
    configure_log,
    load_dotenv_if_present,
    log,
)

COMPUTED_FILES = ("08_boruta.csv", "10_stability.csv", "11_subsets.csv", "09_psi.csv", "12_overlap.csv", "06_leakage.csv", "17_mrmr.csv", "01_obfuscation.csv")
VALUE_COLUMNS = {
    "01_obfuscation.csv": ["ho_auc", "fold1_auc", "fold2_auc", "fold3_auc", "fold4_auc", "fold5_auc"],
    "06_leakage.csv": ["feature", "in_candidate_set_373", "in_headline_subset"],
    "08_boruta.csv": ["n_selected"],
    "09_psi.csv": ["psi_mean", "psi_median", "psi_max", "psi_max_feature"],
    "10_stability.csv": ["d", "nogueira", "jaccard_mean", "jaccard_min", "jaccard_max"],
    "11_subsets.csv": ["feature"],
    "12_overlap.csv": ["n_features", "n_sharing_base_column", "n_distinct_base_columns"],
    "17_mrmr.csv": ["seconds"],
}
STEP_OF = {name: name.split("_")[0] for name in COMPUTED_FILES}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--todo-dir", default="todo", help="folder with the CSV skeletons (default: todo)")
    parser.add_argument("--output-dir", default="todo_done", help="folder that receives the filled CSVs (default: todo_done)")
    parser.add_argument("--work-dir", default="outputs/todo_fill", help="checkpoints, cached frames, logs (default: outputs/todo_fill, git-ignored)")
    parser.add_argument("--datasets", default=",".join(DATASETS), help="comma list of datasets whose selector refits may run (default: all three)")
    parser.add_argument("--steps", default="all", help="comma list of steps to build: copy,01,06,08,09,10,11,12,17 (default: all)")
    parser.add_argument("--tier", type=int, default=3, choices=[0, 1, 2, 3], help="maximum selector cost tier to refit: 0 cache/analytic only, 1 light, 2 medium, 3 heavy (default: 3)")
    parser.add_argument("--b6", default="auto", choices=["auto", "run", "skip"], help="step 1 obfuscation run: auto = run when OPENAI_API_KEY is set (default)")
    parser.add_argument("--b6-candidates", default="cache", choices=["cache", "universe"], help="step 1 candidate set per partition: the cached run's fold-local list (default) or the full 529 universe")
    parser.add_argument("--third-matrix", default="auto", choices=["auto", "skip"], help="build the third-dataset matrix when needed (default) or skip everything that needs it")
    parser.add_argument("--third-refits", default="skip", choices=["skip", "run"], help="refit third-dataset selectors for steps 08/10/11 (default: skip; folds 4-5 exceed 19 GiB RAM, so step 10 rows would stay blank anyway)")
    parser.add_argument("--retry-skipped", action="store_true", help="retry selector fits that a previous run marked as skipped")
    parser.add_argument("--dry-run", action="store_true", help="plan only: report frozen coverage and the refits that would run")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    todo_dir = (REPO_ROOT / args.todo_dir).resolve() if not Path(args.todo_dir).is_absolute() else Path(args.todo_dir)
    out_dir = (REPO_ROOT / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    work_dir = (REPO_ROOT / args.work_dir).resolve() if not Path(args.work_dir).is_absolute() else Path(args.work_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    configure_log(work_dir / "fill_todo.log")
    load_dotenv_if_present()
    known_steps = {"copy", *STEP_OF.values()}
    steps = {item.strip() for item in args.steps.split(",") if item.strip()} if args.steps != "all" else set(known_steps)
    unknown = sorted(steps - known_steps)
    if unknown:
        build_parser().error(f"unknown step(s) {unknown}; choose from {sorted(known_steps)}")
    datasets = tuple(item.strip() for item in args.datasets.split(",") if item.strip())
    manifest = Manifest(output_dir=out_dir)
    manifest.settings = {**vars(args), "repo_root": str(REPO_ROOT), "python": sys.version.split()[0], "cwd": os.getcwd()}
    log(f"fill_todo: todo={todo_dir} out={out_dir} work={work_dir} steps={sorted(steps)} datasets={datasets} tier={args.tier}")

    skeletons: dict[str, Skeleton] = {}
    for name in COMPUTED_FILES:
        path = todo_dir / name
        if path.exists():
            skeletons[name] = Skeleton.read(path)
            skeletons[name].write(out_dir / name)  # the output folder always holds every file
        else:
            manifest.skip(STEP_OF[name], {"file": name}, "skeleton missing from the todo folder")

    from scripts.todo_fill import steps as builders

    if "copy" in steps:
        builders.copy_verbatim(todo_dir, out_dir, manifest)

    # --------------------------------------------------- cached LLM ranking pins
    from scripts.todo_fill import llm_cache

    pins: dict[tuple[str, int], dict] = {}
    if "10_stability.csv" in skeletons and "10" in steps:
        for row in skeletons["10_stability.csv"].rows:
            if row["selector"] != "Pure LLM" or row["dataset"] == THIRD or not row.get("nogueira"):
                continue
            k = BUDGETS[row["backbone"]]
            try:
                pin = llm_cache.pin_fold_files_to_targets(row["dataset"], k, target_nogueira=float(row["nogueira"]), target_jaccard_mean=float(row["jaccard_mean"]), d=int(float(row["d"])))
            except Exception as exc:  # noqa: BLE001
                manifest.note("10", f"could not pin cached fold files for {row['dataset']} K={k}: {exc}")
                continue
            pins[(row["dataset"], k)] = pin
            manifest.note("10", f"{row['dataset']} K={k}: cached fold files {'reproduce' if pin['matched'] else 'DO NOT reproduce'} the pre-filled Table 9 values ({pin['reproduced']})", chosen_files=pin["chosen_files"])
    manifest.settings["llm_cache_inventory"] = llm_cache.inventory()

    # ---------------------------------------------------------------- planning
    from scripts.todo_fill.data import DataContext
    from scripts.todo_fill.selections import SelectionEngine, SelectionStore

    headline_rows = skeletons["11_subsets.csv"].rows if "11_subsets.csv" in skeletons else []
    requests = builders.plan_requests({name: skel for name, skel in skeletons.items() if STEP_OF[name] in steps}, manifest, leakage_headline_rows=headline_rows)
    requests = [request for request in requests if request.dataset in datasets]
    if args.third_matrix == "skip" or args.third_refits == "skip":
        dropped = [request for request in requests if request.dataset == THIRD]
        requests = [request for request in requests if request.dataset != THIRD]
        if dropped:
            manifest.note("plan", f"{len(dropped)} third-dataset selector refits skipped (--third-refits {args.third_refits}, --third-matrix {args.third_matrix}); their cells stay blank")
    if args.dry_run:
        for request in requests:
            print(f"  refit needed: {request.dataset:34s} {request.backbone:8s} {request.label:24s} {request.partition}")
        print(f"{len(requests)} selector fits would run; frozen evidence covers the rest.")
        manifest.write()
        return 0

    data = DataContext(work_dir)
    engine = SelectionEngine(store=SelectionStore(work_dir), data=data, manifest=manifest, max_tier=args.tier, third_ranking=None, retry_skipped=args.retry_skipped)
    if any(request.dataset == THIRD for request in requests) or ("17" in steps and THIRD in datasets and args.third_matrix != "skip"):
        try:
            data.third().ensure_matrix()
        except Exception as exc:  # noqa: BLE001
            manifest.skip("data", {"dataset": THIRD}, f"third-dataset matrix unavailable: {type(exc).__name__}: {exc}")
            requests = [request for request in requests if request.dataset != THIRD]
    try:
        engine.run(requests)
    except KeyboardInterrupt:
        log("interrupted; writing what is available")
    except Exception as exc:  # noqa: BLE001
        manifest.note("selection", f"selector engine stopped early: {type(exc).__name__}: {exc}\n{traceback.format_exc()}")

    # ------------------------------------------------------------------ steps
    def build(name: str, fn) -> None:
        if name not in skeletons or STEP_OF[name] not in steps:
            return
        skeleton = skeletons[name]
        try:
            fn(skeleton)
        except Exception as exc:  # noqa: BLE001
            manifest.note(STEP_OF[name], f"{name} failed: {type(exc).__name__}: {exc}\n{traceback.format_exc()}")
        skeleton.write(out_dir / name)
        filled, total = skeleton.fill_rate(VALUE_COLUMNS[name])
        manifest.csv_summary[name] = {"filled": filled, "total": total, "status": "complete" if filled == total and total else ("partial" if filled else "empty")}
        log(f"wrote {name}: {filled}/{total} value cells filled")

    for name in COMPUTED_FILES:  # excluded steps keep their untouched skeleton in the output folder
        if name in skeletons and STEP_OF[name] not in steps:
            manifest.csv_summary[name] = {"filled": 0, "total": len(skeletons[name].rows), "status": "excluded (--steps)"}
    build("08_boruta.csv", lambda skel: builders.fill_08(skel, engine, manifest))
    build("10_stability.csv", lambda skel: builders.fill_10(skel, engine, manifest, pins))
    build("11_subsets.csv", lambda skel: builders.fill_11(skel, engine, manifest))
    build("09_psi.csv", lambda skel: builders.fill_09(skel, engine, manifest, out_dir))
    build("12_overlap.csv", lambda skel: builders.fill_12(skel, engine, manifest, out_dir))
    build("06_leakage.csv", lambda skel: builders.fill_06(skel, engine, manifest, headline_rows))
    if THIRD in datasets and args.third_matrix != "skip":
        build("17_mrmr.csv", lambda skel: builders.fill_17(skel, engine, manifest, None))
    elif "17_mrmr.csv" in skeletons and "17" in steps:
        manifest.skip("17", {"dataset": THIRD}, "third dataset excluded by --datasets/--third-matrix")
        manifest.csv_summary["17_mrmr.csv"] = {"filled": 0, "total": len(skeletons["17_mrmr.csv"].rows), "status": "skipped"}
    engine.release()

    if "01" in steps and "01_obfuscation.csv" in skeletons:
        run_b6 = args.b6 == "run" or (args.b6 == "auto" and bool(os.getenv("OPENAI_API_KEY")))
        if run_b6 and HOMECREDIT in datasets:
            from scripts.todo_fill.b6_two_arm import run_b6 as _run_b6

            build("01_obfuscation.csv", lambda skel: _run_b6(skeleton=skel, data=data, work_dir=work_dir, out_dir=out_dir, manifest=manifest, candidate_source=args.b6_candidates))
        else:
            reason = "OPENAI_API_KEY is not set; put it in .env and rerun with --steps 01" if not os.getenv("OPENAI_API_KEY") else "disabled by --b6/--datasets"
            manifest.skip("01", {"dataset": HOMECREDIT}, reason)
            manifest.csv_summary["01_obfuscation.csv"] = {"filled": 0, "total": len(skeletons["01_obfuscation.csv"].rows) * 6, "status": "skipped"}

    manifest.write()
    print()
    print(f"{'file':22s} {'filled':>8s} {'total':>8s}  status")
    for name, summary in sorted(manifest.csv_summary.items()):
        print(f"{name:22s} {summary.get('filled', ''):>8} {summary.get('total', ''):>8}  {summary.get('status', '')}")
    print(f"\nOutputs: {out_dir}\nProvenance: {out_dir / 'fill_manifest.json'} and {out_dir / 'fill_log.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
