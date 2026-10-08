#!/usr/bin/env python
"""Fill ``responserepeats_calls.csv``, ``poolcontrols_auc.csv`` and ``poolcontrols_delta.csv`` of
``todo/README_reruns.md`` into ``todo_done/``.

    uv run --no-sync python scripts/fill_todo_reruns.py --preflight     # light checks + time estimates; no API call, no fit
    uv run --no-sync python scripts/fill_todo_reruns.py --commit-plan   # commit rerun_plan.md, then run both reruns

``responserepeats_auc.csv`` is supplied already filled and is never opened, written, refitted or
otherwise touched: its name appears here only in ``PROTECTED_FILES``, whose size and modification
time (a stat, not a read) are recorded before the run and verified after it.

Rerun 1 (response repeats) issues 50 new full-DEV ranking calls (ten per call structure) under
the frozen target-free contract and records every call (validation, token usage summed over the
application attempts) in ``responserepeats_calls.csv``.  Rerun 2 (pool controls) builds, per case, the LLM
pool, the IV/WOE pool, five random pools and the whole candidate set, reduces each to K with
mRMR (timed, one selection at a time, nothing else running), refits the frozen backbone on full
DEV, scores HO once and pairs every pipeline against the LLM pool with the paper's bootstrap.

Protocol choices the README does not spell out (all repeated in ``rerun_settings.txt``):

* mRMR is the mutual-information implementation (``mrmr_mutual_information``: n_bins 10,
  objective mid), which the manuscript defines for stand-alone mRMR and for LLM then mRMR.
  The ``LLM then mRMR`` recipe of the Gate-4 fills (``scripts/todo_fill/selectors.py``) runs the
  superseded RF/correlation filter and is not used here.
* The mRMR stage reads the frozen selection encoding (one numeric column per original feature)
  on Home Credit and Stability 2024 and on every Full mRMR run, and the expanded one-hot matrix
  for the pooled LendingClub pipelines, as README_reruns.md prescribes.
* "The candidate set" (``--pool-universe``): ``baseline`` (default) is the universe the frozen
  stand-alone selectors searched (529 Home Credit, 675 LendingClub, the 1,068 availability-
  filtered Stability 2024 features); ``llm`` is the exact list offered to the LLM in the call
  behind the LLM pool (391 / 161 / 1,068).
* Stability 2024 uses the paper's headline cohort (DEV 2019-01-01..2020-01-31, 1,157,512 rows;
  HO 2020-02-01..2020-10-05, 369,147 rows).  The local protocol lock still carries the superseded
  2020-02-26 cutoff; ``--third-cohort lock`` restores it.
* The Stability LLM pool comes from the regenerated named ranking of the Gate-4 fills; the
  paper's cached Stability ranking is not on this machine.
* ``select_seconds`` = discretisation + MI relevance + greedy selection of the mRMR stage.  Data
  loading and the selection encoding are excluded and logged separately.  Discretisation is
  column-wise, so it is timed once per column and summed over the columns a pipeline uses.

Everything is checkpointed under ``--work-dir``, which has its own fit store so no Gate-4 fit of
the superseded Stability cohort can leak in; re-running the command resumes.  The row and column
layout of every skeleton is preserved and only blank cells are filled.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Sequence

#: README_reruns.md: "measured on one machine with four threads, one experiment at a time".
SELECT_THREADS = 4
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
for _name in THREAD_VARS:
    os.environ.setdefault(_name, str(SELECT_THREADS))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

REPO_ROOT = Path(__file__).resolve().parents[1]
for _candidate in (REPO_ROOT, REPO_ROOT / "src"):
    if str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.todo_fill.common import (  # noqa: E402
    BUDGETS,
    FULL_DEV,
    HOMECREDIT,
    LENDINGCLUB,
    LLM_POOL_BUDGET,
    THIRD,
    FillError,
    Manifest,
    Skeleton,
    available_ram_gb,
    configure_log,
    fmt,
    frame_gib,
    heartbeat,
    log,
    read_json,
    require_ram,
    sha256_file,
    slug,
    write_json,
)
from scripts.todo_fill.gate4.fits import FitExecutor, FitSpec  # noqa: E402
from scripts.todo_fill.gate4.llm import Ranker  # noqa: E402

# ----------------------------------------------------------------------------- constants

DATASETS: dict[str, str] = {"Home Credit": HOMECREDIT, "LendingClub v2": LENDINGCLUB, "Stability 2024": THIRD}
DATASET_LABELS: dict[str, str] = {value: key for key, value in DATASETS.items()}
BACKBONES: dict[str, str] = {"LR": "lr", "CatBoost": "catboost"}

CALLS_CSV = "responserepeats_calls.csv"
POOLS_CSV = "poolcontrols_auc.csv"
DELTAS_CSV = "poolcontrols_delta.csv"
#: The only files this script reads from ``--todo-dir`` and writes to ``--output-dir``.
#: file -> (columns, expected rows).  Columns must match exactly; rows are only reported.
SCHEMA: dict[str, tuple[list[str], int]] = {
    CALLS_CSV: (["dataset", "call_budget", "repeat", "validation_passed", "input_tokens", "output_tokens"], 50),
    POOLS_CSV: (["dataset", "backbone", "pipeline", "ho_auc", "select_seconds"], 48),
    DELTAS_CSV: (["dataset", "backbone", "comparison", "delta_auc", "ci_low", "ci_high"], 42),
}
VALUE_COLUMNS: dict[str, list[str]] = {
    CALLS_CSV: ["validation_passed", "input_tokens", "output_tokens"],
    POOLS_CSV: ["ho_auc", "select_seconds"],
    DELTAS_CSV: ["delta_auc", "ci_low", "ci_high"],
}
#: Supplied, already filled, and never touched: only hashed before and after the run.
PROTECTED_FILES: tuple[str, ...] = ("responserepeats_auc.csv",)

RR_STEP = "rr"
PC_STEP = "pc"
HC_CALL_BUDGET = 100
#: Extra rounds for calls that never reached validation (transport errors, not contract failures).
TRANSPORT_RETRY_ROUNDS = 3
RANDOM_SEEDS: tuple[int, ...] = (101, 102, 103, 104, 105)
POOL_PIPELINES: tuple[str, ...] = ("LLM pool", "IV pool", *(f"Random pool s{seed}" for seed in RANDOM_SEEDS), "Full mRMR")
#: The manuscript's information-value interval for the IV/WOE ranker.
IV_INTERVAL = (0.01, 0.50)
MATERIALITY = 0.010

#: Stability 2024 cohorts.  ``paper`` is the headline evaluation of the manuscript (Table
#: ``tab:datasets``); ``lock`` keeps the superseded cutoff of the local protocol lock.
THIRD_COHORTS: dict[str, dict[str, dict[str, Any]] | None] = {
    "paper": {
        "dev": {"date_min": "2019-01-01", "date_max": "2020-01-31", "rows": 1_157_512},
        "oot": {"date_min": "2020-02-01", "date_max": "2020-10-05", "rows": 369_147},
    },
    "lock": None,
}
PLAN_COMMIT_MESSAGE = (
    "Commit the rerun plan before any rerun call or refit\n\n"
    "Decision rules for the response repeats and pool controls of\n"
    "Feature-Selection-Research/todo/README_reruns.md, fixed before running.\n\n"
    "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>\n"
)


def _pipe(names: Iterable[str]) -> str:
    return "|".join(str(name) for name in names)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def call_name(dataset: str, budget: int | None, repeat: int) -> str:
    parts = [dataset, FULL_DEV]
    if budget is not None:
        parts.append(f"b{budget}")
    parts += [f"repeat{repeat:02d}", "named"]
    return "__".join(parts)


# ------------------------------------------------------------------------ Stability cohort


def use_third_cohort(name: str) -> None:
    """Point ``ThirdDataset`` (this process) at the chosen DEV/HO cutoff; folds are untouched."""

    from scripts.todo_fill import data as data_module

    cls = data_module.ThirdDataset
    if "_lock_boundaries" not in cls.__dict__:
        cls._lock_boundaries = cls.__dict__["boundaries"]
    spec = THIRD_COHORTS[name]
    if spec is None:
        cls.boundaries = cls._lock_boundaries
        return

    def boundaries(self: Any) -> dict[str, Any]:
        base = copy.deepcopy(self.lock["approved_protocol"]["split_and_fold_boundaries"])
        for part in ("dev", "oot"):
            base[part].update(spec[part])
        return base

    cls.boundaries = property(boundaries)


def run_fit_in_cohort(spec_payload: dict[str, Any], work_dir: str, cohort: str) -> dict[str, Any]:
    """Worker entry point: the Gate-4 frozen fit under the chosen Stability cohort."""

    use_third_cohort(cohort)
    from scripts.todo_fill.gate4.fits import run_fit

    return run_fit(spec_payload, work_dir)


@dataclass
class CohortFitExecutor(FitExecutor):
    """``FitExecutor`` whose spawned workers apply the Stability cohort before fitting."""

    cohort: str = "paper"

    def _run_batch(self, batch: list[FitSpec], workers: int, step: str) -> None:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor, as_completed

        ordered = sorted(batch, key=lambda spec: (spec.backbone != "catboost", spec.dataset, spec.label))
        for name in THREAD_VARS:
            os.environ[name] = "1"  # backbone fits are single-threaded, as in every Gate-4 run
        started = time.perf_counter()

        def report(done: int, spec: FitSpec) -> None:
            log(f"[{step}] fit {done}/{len(ordered)} finished: {spec.dataset}/{spec.backbone} {spec.label} ({(time.perf_counter() - started) / 60:.1f} min elapsed)")

        if workers == 1:
            for done, spec in enumerate(ordered, start=1):
                self._record(run_fit_in_cohort(spec.to_dict(), str(self.work_dir), self.cohort), spec, step)
                report(done, spec)
            return
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(run_fit_in_cohort, spec.to_dict(), str(self.work_dir), self.cohort): spec for spec in ordered}
            for done, future in enumerate(as_completed(futures), start=1):
                spec = futures[future]
                try:
                    payload = future.result()
                except Exception as exc:  # noqa: BLE001
                    payload = {"status": "failed", **spec.to_dict(), "fit_id": spec.fit_id, "reason": f"worker crashed: {type(exc).__name__}: {exc}"}
                self._record(payload, spec, step)
                report(done, spec)


# ---------------------------------------------------------------------------- ranking calls


class _UsageCompletions:
    def __init__(self, delegate: Any, sink: list[dict[str, Any]]) -> None:
        self._delegate = delegate
        self._sink = sink

    def create(self, **kwargs: Any) -> Any:
        response = self._delegate.create(**kwargs)
        usage = getattr(response, "usage", None)
        self._sink.append(
            {
                "response_id": getattr(response, "id", None),
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }
        )
        return response


class _UsageClient:
    """OpenAI client proxy that records the token usage of every attempt, accepted or not."""

    def __init__(self, delegate: Any, sink: list[dict[str, Any]]) -> None:
        self.chat = SimpleNamespace(completions=_UsageCompletions(delegate.chat.completions, sink))


class RepeatRanker(Ranker):
    """Gate-4 ranker plus per-attempt token usage and a final verdict for contract failures.

    A call that got a response and failed the strict contract in all three attempts is final
    (``validation_passed`` False) and is never re-issued, so a resumed run cannot redraw it.  A
    call that never reached validation (transport or guard error) is re-issued on the next run.
    """

    def __init__(self, work_dir: Path, manifest: Manifest, enabled: bool | None = None, reissue_failed: bool = False) -> None:
        super().__init__(work_dir, manifest, enabled=enabled, retry_failed=0)
        self.reissue_failed = reissue_failed
        self._usage: list[dict[str, Any]] = []

    def _openai(self):
        if self._client is None:
            from openai import OpenAI

            self._client = _UsageClient(OpenAI(api_key=os.getenv("OPENAI_API_KEY")), self._usage)
        return self._client

    def failed(self, step: str, name: str) -> dict[str, Any] | None:
        path = self.path(step, f"{name}.failed")
        return read_json(path) if path.exists() else None

    def rank(self, *, step: str, name: str, **kwargs: Any) -> dict[str, Any] | None:
        found = self.cached(step, name)
        if found is not None:
            return found
        if self.failed(step, name) is not None and not self.reissue_failed:
            return None
        self._usage.clear()
        result = super().rank(step=step, name=name, **kwargs)
        usage = [dict(item) for item in self._usage]
        if result is not None:
            result["usage_per_attempt"] = usage
            write_json(self.path(step, name), result)
            return result
        failed_path = self.path(step, f"{name}.failed")
        if failed_path.exists():
            record = read_json(failed_path)
            record["usage_per_attempt"] = usage
            answered = any(((attempt.get("response") or {}).get("id") or (attempt.get("response") or {}).get("raw_content")) for attempt in record.get("attempts", []))
            if answered:
                write_json(failed_path, record)
            else:
                failed_path.unlink()
                write_json(self.path(step, f"{name}.not_answered"), record)
                self.manifest.note(step, f"{name}: no response reached validation ({(record.get('errors') or ['unknown'])[-1]}); the call is re-issued on the next run")
        return None


# ------------------------------------------------------------------------------ mRMR, IV


def _compact_codes(codes: np.ndarray) -> np.ndarray:
    """Smallest integer dtype holding the codes; the MI estimator only compares labels."""

    if codes.size == 0:
        return codes.astype(np.int8)
    low, high = int(codes.min()), int(codes.max())
    for dtype in (np.int8, np.int16, np.int32):
        info = np.iinfo(dtype)
        if low >= info.min and high <= info.max:
            return codes.astype(dtype)
    return codes


def discretize(frame: pd.DataFrame, n_bins: int) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """The frozen MI-mRMR discretisation, one column at a time, with its per-column seconds."""

    from credit_risk_fs.selectors.lightweight.mi_mrmr import _discretize_column

    codes: dict[str, np.ndarray] = {}
    seconds: dict[str, float] = {}
    for name in frame.columns:
        started = time.perf_counter()
        codes[str(name)] = _compact_codes(_discretize_column(frame[name].reset_index(drop=True), n_bins))
        seconds[str(name)] = time.perf_counter() - started
    return codes, seconds


def mrmr_from_codes(codes: dict[str, np.ndarray], order: Sequence[str], y: pd.Series, k: int, settings: dict[str, Any]) -> tuple[list[str], float]:
    """Relevance + greedy of ``MutualInformationMRMRSelector`` over precomputed codes.

    Same estimator, argument order, tie rule and zero-relevance policy as the selector's own
    ``fit`` (the preflight checks equality); returns the first ``k`` names and the seconds.
    """

    from sklearn.metrics import mutual_info_score

    from credit_risk_fs.selectors.lightweight.mi_mrmr import MutualInformationMRMRSelector

    selector = MutualInformationMRMRSelector(k=int(k), **settings)
    target = np.asarray(pd.Series(y).reset_index(drop=True).to_numpy(), dtype="int64")
    order = [str(name) for name in order]
    started = time.perf_counter()
    relevance = {name: float(mutual_info_score(codes[name], target)) for name in order}
    selector._codes = {name: codes[name] for name in order}
    selector._mi_cache = {}
    ranking, _, _ = selector._rank_from_relevance(candidate_order=order, relevance=relevance, pair_mi=selector._pair_mi)
    return [str(name) for name in list(ranking)[: int(k)]], time.perf_counter() - started


def iv_scores(frame: pd.DataFrame, y: pd.Series, settings: dict[str, Any]) -> dict[str, float]:
    """Total information value per column (the frozen IV/WOE ranker; column-wise)."""

    from credit_risk_fs.selectors.lightweight.iv import InformationValueSelector

    selector = InformationValueSelector(k=None, **settings)
    selector.fit(frame, y)
    return {str(name): float(value) for name, value in (selector.iv_scores_ or {}).items()}


def iv_pool(scores: dict[str, float], candidates: Sequence[str], size: int) -> list[str]:
    from credit_risk_fs.selectors.lightweight.contract import rank_by_score

    ranked = rank_by_score(scores, candidate_order=list(candidates))
    eligible = [name for name in ranked if np.isfinite(scores[name]) and IV_INTERVAL[0] <= scores[name] <= IV_INTERVAL[1]]
    return eligible[:size]


def random_pool(candidates: Sequence[str], size: int, seed: int) -> list[str]:
    """``size`` names drawn uniformly without replacement, kept in universe order."""

    picks = np.sort(np.random.default_rng(seed).choice(len(candidates), size=min(size, len(candidates)), replace=False))
    return [candidates[int(index)] for index in picks]


def protected_stamps(folders: Iterable[Path]) -> dict[str, tuple[int, int] | None]:
    """(size, mtime_ns) of every ``PROTECTED_FILES`` copy in ``folders``: a stat, the file is never opened."""

    stamps: dict[str, tuple[int, int] | None] = {}
    for folder in folders:
        for name in PROTECTED_FILES:
            path = Path(folder) / name
            if path.exists():
                info = path.stat()
                stamps[str(path)] = (int(info.st_size), int(info.st_mtime_ns))
            else:
                stamps[str(path)] = None
    return stamps


def selector_settings(dataset: str, method: str, cls: type) -> dict[str, Any]:
    from scripts.todo_fill.selectors import _filtered, baseline_settings

    return _filtered(cls, baseline_settings(dataset, method))


def mi_settings(dataset: str) -> dict[str, Any]:
    from credit_risk_fs.selectors.lightweight.mi_mrmr import MutualInformationMRMRSelector

    return selector_settings(dataset, "mrmr_mutual_information", MutualInformationMRMRSelector)


def iv_settings(dataset: str) -> dict[str, Any]:
    from credit_risk_fs.selectors.lightweight.iv import InformationValueSelector

    return selector_settings(dataset, "iv_woe", InformationValueSelector)


# ------------------------------------------------------------------------------ the plan


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=check)


def plan_state(plan: Path) -> dict[str, Any]:
    state: dict[str, Any] = {"path": str(plan), "exists": plan.exists()}
    if not plan.exists():
        return state
    top = _git(plan.parent, "rev-parse", "--show-toplevel", check=False)
    if top.returncode != 0:
        state["repo"] = None
        return state
    repo = Path(top.stdout.strip())
    relative = plan.resolve().relative_to(repo.resolve()).as_posix()
    tracked = _git(repo, "ls-files", "--error-unmatch", "--", relative, check=False).returncode == 0
    dirty = bool(_git(repo, "status", "--porcelain", "--", relative, check=False).stdout.strip())
    commit = _git(repo, "log", "-1", "--format=%H", "--", relative, check=False).stdout.strip() if tracked else ""
    committed_at = _git(repo, "log", "-1", "--format=%cI", "--", relative, check=False).stdout.strip() if commit else ""
    branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD", check=False).stdout.strip()
    state.update(repo=str(repo), relative=relative, tracked=tracked, dirty=dirty, commit=commit, committed_at=committed_at, branch=branch, sha256=sha256_file(plan))
    return state


def ensure_plan_committed(plan: Path, commit: bool) -> dict[str, Any]:
    """README_reruns.md Step 0: nothing runs before ``rerun_plan.md`` is committed."""

    state = plan_state(plan)
    if not state["exists"]:
        raise SystemExit(f"rerun plan not found: {plan}; README_reruns.md Step 0 requires it before any run")
    if state.get("repo") is None:
        raise SystemExit(f"{plan} is not inside a git repository, so it cannot be committed")
    if state["tracked"] and not state["dirty"] and state["commit"]:
        return state
    if not commit:
        raise SystemExit(
            f"{plan} is {'modified since its last commit' if state['tracked'] else 'not committed'}; README_reruns.md Step 0 requires the "
            "plan to be committed before running. Re-run with --commit-plan (commits that file only) or commit it yourself."
        )
    repo, relative = Path(state["repo"]), state["relative"]
    try:
        _git(repo, "add", "--", relative)
        _git(repo, "commit", "--only", "-m", PLAN_COMMIT_MESSAGE, "--", relative)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"committing {relative} failed: {exc.stderr or exc.stdout}") from exc
    state = plan_state(plan)
    if not (state["tracked"] and not state["dirty"] and state["commit"]):
        raise SystemExit(f"{relative} is still not committed cleanly after the commit attempt")
    state["committed_by_this_run"] = True
    log(f"committed {relative} in {repo} on branch {state['branch']}: {state['commit']}")
    return state


# ------------------------------------------------------------------------------- setup


def read_skeletons(todo_dir: Path) -> dict[str, Skeleton]:
    skeletons: dict[str, Skeleton] = {}
    for name, (columns, rows) in SCHEMA.items():
        path = todo_dir / name
        if not path.exists():
            raise SystemExit(f"skeleton missing: {path}")
        skeleton = Skeleton.read(path)
        if skeleton.columns != columns:
            raise SystemExit(f"{name}: columns {skeleton.columns} differ from the expected {columns}")
        if len(skeleton.rows) != rows:
            log(f"note: {name} has {len(skeleton.rows)} rows (README layout has {rows}); every row present is filled")
        skeletons[name] = skeleton
    return skeletons


def prepare_work_dir(work: Path, shared: Path) -> None:
    """Own checkpoint root; the cached frames and the Stability matrix are shared by symlink."""

    if work.resolve() == shared.resolve():
        raise SystemExit("--work-dir must differ from --shared-work-dir (the reruns keep their own fit store)")
    work.mkdir(parents=True, exist_ok=True)
    for name in ("frames", "hcms2024_matrix"):
        target = (shared / name).resolve()
        link = work / name
        if not target.exists():
            raise SystemExit(f"shared cache missing: {target} (produced by the Gate-4 fills)")
        if link.is_symlink() or link.exists():
            continue
        link.symlink_to(target, target_is_directory=True)
    source = shared / "gate4" / "third_candidates.json"
    destination = work / "gate4" / "third_candidates.json"
    if source.exists() and not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _resolve(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


# ----------------------------------------------------------------------------- the run


class Reruns:
    def __init__(self, args: argparse.Namespace, skeletons: dict[str, Skeleton], manifest: Manifest, *, work_dir: Path, shared_dir: Path, out_dir: Path, sidecar_dir: Path, plan: dict[str, Any] | None) -> None:
        from scripts.todo_fill.data import DataContext
        from scripts.todo_fill.gate4.records import RecordFactory

        self.args = args
        self.skeletons = skeletons
        self.manifest = manifest
        self.work_dir = work_dir
        self.shared_dir = shared_dir
        self.out_dir = out_dir
        self.sidecar_dir = sidecar_dir
        self.plan = plan
        self.data = DataContext(work_dir)
        self.records = RecordFactory(self.data, work_dir, manifest, candidate_source="cache")
        self.ranker = RepeatRanker(work_dir, manifest, enabled=False if args.no_api else None, reissue_failed=args.reissue_failed_calls)
        self.executor = CohortFitExecutor(work_dir=work_dir, manifest=manifest, jobs=args.jobs, third_jobs=args.third_jobs, cohort=args.third_cohort)
        self.calls: dict[tuple[str, int | None, int], dict[str, Any]] = {}
        self.pc_sel: dict[tuple[str, str, str], dict[str, Any]] = {}
        self.inference: dict[tuple[str, str, str], dict[str, Any]] = {}
        self.dataset_info: dict[str, dict[str, Any]] = {}
        self._dev: dict[str, tuple[pd.DataFrame, pd.Series]] = {}
        self._parse_rows()

    # ------------------------------------------------------------------ skeleton rows
    def _parse_rows(self) -> None:
        self.call_items: list[tuple[str, int | None, int]] = []
        self.pc_items: list[tuple[str, str, str]] = []
        self.delta_items: list[tuple[str, str, str]] = []
        for row in self.skeletons[CALLS_CSV].rows:
            dataset = DATASETS.get(row["dataset"])
            try:
                budget, repeat = int(row["call_budget"]), int(row["repeat"])
            except ValueError:
                dataset = None
            if dataset is None or dataset == THIRD or (dataset == HOMECREDIT and budget != HC_CALL_BUDGET):
                self.manifest.skip(RR_STEP, row, "unrecognised call row")
                continue
            self.call_items.append((dataset, None if dataset == HOMECREDIT else budget, repeat))
        for row in self.skeletons[POOLS_CSV].rows:
            dataset, backbone = DATASETS.get(row["dataset"]), BACKBONES.get(row["backbone"])
            if not dataset or not backbone or row["pipeline"] not in POOL_PIPELINES:
                self.manifest.skip(PC_STEP, row, "unrecognised pool-control row")
                continue
            self.pc_items.append((dataset, backbone, row["pipeline"]))
        for row in self.skeletons[DELTAS_CSV].rows:
            dataset, backbone = DATASETS.get(row["dataset"]), BACKBONES.get(row["backbone"])
            head, _, other = row["comparison"].partition(" vs ")
            if not dataset or not backbone or head != "LLM pool" or other not in POOL_PIPELINES or other == "LLM pool":
                self.manifest.skip(PC_STEP, row, "unrecognised comparison row")
                continue
            self.delta_items.append((dataset, backbone, other))
            for pipeline in ("LLM pool", other):  # a delta needs both fits even if the AUC file lacks the row
                if (dataset, backbone, pipeline) not in self.pc_items:
                    self.pc_items.append((dataset, backbone, pipeline))

    # --------------------------------------------------------------------- data access
    def dev_all(self, dataset: str) -> tuple[pd.DataFrame, pd.Series]:
        if dataset not in self._dev:
            self._dev[dataset] = self.data.training_frame(dataset, FULL_DEV)
        return self._dev[dataset]

    def release(self, dataset: str) -> None:
        self._dev.pop(dataset, None)
        if dataset != THIRD:
            self.data.release_dev(dataset)
        gc.collect()

    def third_slice(self, columns: Sequence[str]) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
        third = self.data.third()
        date_min, date_max, rows = third.date_range(FULL_DEV)
        require_ram(frame_gib(rows, len(columns) + 5, 8) * 2.0, f"Stability 2024 full-DEV slice of {len(columns)} columns")
        frame = third.read_slice(date_min, date_max, list(columns))
        if len(frame) != rows:
            raise FillError(f"Stability 2024 DEV {date_min}..{date_max} has {len(frame)} rows; the cohort expects {rows}")
        y = frame["target"].astype("int8").reset_index(drop=True)
        ids = frame["case_id"].to_numpy()
        X = frame.loc[:, list(columns)].reset_index(drop=True)
        del frame
        gc.collect()
        return X, y, ids

    def candidates(self, dataset: str, backbone: str) -> list[str]:
        """The candidate set of ``--pool-universe``, in universe order."""

        if dataset == THIRD:
            return list(self.records.third_candidates())
        if self.args.pool_universe == "baseline":
            return list(self.records.universe(dataset))
        return list(self.records.candidate_set(dataset, FULL_DEV, None if dataset == HOMECREDIT else LLM_POOL_BUDGET[backbone]))

    def llm_pool_ranking(self, dataset: str, backbone: str) -> tuple[list[str] | None, str]:
        from scripts.todo_fill import llm_cache

        if dataset == HOMECREDIT:
            cached = llm_cache.ranking(HOMECREDIT, FULL_DEV)
            return list(cached.features), f"artifacts/llm_cache full-DEV ranking {cached.relative_path}"
        if dataset == LENDINGCLUB:
            budget = LLM_POOL_BUDGET[backbone]
            cached = llm_cache.ranking(LENDINGCLUB, FULL_DEV, budget=budget)
            if cached.feature_budget != budget:
                self.manifest.note(PC_STEP, f"LendingClub {backbone}: no cached full-DEV call with budget {budget}; used {cached.relative_path} (budget {cached.feature_budget})")
            return list(cached.features), f"artifacts/llm_cache full-DEV budget-{cached.feature_budget} call {cached.relative_path}"
        path = self.shared_dir / "gate4" / "rankings" / "shared" / "third_named_full_dev.json"
        if not path.exists():
            return None, f"missing: {path}"
        payload = read_json(path)
        return list(payload["ranking"]), f"regenerated Stability 2024 named ranking of the Gate-4 fills ({path.name}, response {payload.get('response_id')}); the paper's cached ranking is not on this machine"

    # ----------------------------------------------------------------- checkpoints
    def _sel_path(self, step: str, *parts: Any) -> Path:
        path = self.work_dir / "gate4" / "selections" / step / ("__".join(slug(str(part)) for part in parts if str(part)) + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _pc_path(self, dataset: str, backbone: str, pipeline: str) -> Path:
        return self._sel_path(PC_STEP, self.args.pool_universe, self.args.third_cohort if dataset == THIRD else "", dataset, backbone, pipeline)

    def _load_pc(self, dataset: str, backbone: str, pipeline: str) -> dict[str, Any] | None:
        key = (dataset, backbone, pipeline)
        if key not in self.pc_sel:
            path = self._pc_path(dataset, backbone, pipeline)
            if path.exists():
                self.pc_sel[key] = read_json(path)
        return self.pc_sel.get(key)

    # ================================================================ rerun 1: calls
    def _call_settled(self, item: tuple[str, int | None, int]) -> bool:
        """Accepted, or failed the strict contract (final); anything else still owes a call."""

        name = call_name(*item)
        return self.ranker.cached(RR_STEP, name) is not None or self.ranker.failed(RR_STEP, name) is not None

    def make_calls(self) -> None:
        structures = list(dict.fromkeys((dataset, budget) for dataset, budget, _ in self.call_items))
        repeats = sorted({repeat for _, _, repeat in self.call_items})
        order = [(dataset, budget, repeat) for repeat in repeats for dataset, budget in structures if (dataset, budget, repeat) in self.call_items]  # repeat-major
        prepared: dict[tuple[str, int | None], tuple[list[str], list[dict[str, Any]]]] = {}
        started = time.perf_counter()
        for round_index in range(TRANSPORT_RETRY_ROUNDS + 1):
            pending = [item for item in order if not self._call_settled(item)]
            if not pending or not self.ranker.enabled:
                break
            if round_index:
                log(f"[{RR_STEP}] {len(pending)} call(s) never reached validation; retry round {round_index}/{TRANSPORT_RETRY_ROUNDS} after {60 * round_index}s")
                time.sleep(60 * round_index)
            for dataset, budget, repeat in pending:
                if (dataset, budget) not in prepared:
                    names = self.records.candidate_set(dataset, FULL_DEV, budget)
                    prepared[(dataset, budget)] = (names, self.records.named_records(dataset, names))
                names, named = prepared[(dataset, budget)]
                self.ranker.rank(
                    step=RR_STEP,
                    name=call_name(dataset, budget, repeat),
                    dataset=dataset,
                    partition=FULL_DEV,
                    condition="named",
                    records=named,
                    ids=names,
                    universe=self.records.universe(dataset),
                    extra={"budget": budget, "repeat_id": repeat, "call_budget": budget or HC_CALL_BUDGET},
                )
        for item in order:
            result = self.ranker.cached(RR_STEP, call_name(*item))
            if result is not None:
                self.calls[item] = result
            elif not self._call_settled(item):
                self.manifest.skip(RR_STEP, {"call": call_name(*item)}, "the call never reached validation (API disabled or repeated transport errors); its row stays blank until the command is re-run")
        self.manifest.timing("calls", time.perf_counter() - started, n_calls=len(self.calls))
        for dataset, budget in structures:
            done = [self.calls[(dataset, budget, repeat)] for repeat in repeats if (dataset, budget, repeat) in self.calls]
            label = f"{DATASET_LABELS[dataset]} call budget {budget or HC_CALL_BUDGET}"
            if not done:
                continue
            prompts = sorted({str(call.get("prompt_sha256")) for call in done})
            distinct = len({tuple(call["ranking"]) for call in done})
            candidates = sorted({int(call.get("n_candidates") or 0) for call in done})
            self.manifest.note(RR_STEP, f"{label}: {len(done)} accepted calls over {candidates} candidates, prompt sha256 {[p[:12] for p in prompts]}, {distinct} distinct rankings of 100")
            reference = self.shared_dir / "gate4" / "rankings" / "45" / f"{call_name(dataset, budget, 1)}.json"
            if reference.exists():
                old = str(read_json(reference).get("prompt_sha256"))
                self.manifest.note(RR_STEP, f"{label}: prompt {'identical to' if prompts == [old] else 'DIFFERENT from'} the Gate-4 step-45 repeat call ({old[:12]})")

    # ============================================================ rerun 2: pool selections
    def pool_selections(self, dataset: str) -> None:
        wanted = [(backbone, pipeline) for d, backbone, pipeline in self.pc_items if d == dataset]
        pending = [(backbone, pipeline) for backbone, pipeline in wanted if self._load_pc(dataset, backbone, pipeline) is None]
        if not pending:
            return
        backbones = sorted({backbone for backbone, _ in wanted})
        candidates = {backbone: self.candidates(dataset, backbone) for backbone in backbones}
        union = list(dict.fromkeys(name for backbone in backbones for name in candidates[backbone]))
        settings = mi_settings(dataset)
        info = self.dataset_info.setdefault(dataset, {})
        info.update({"n_candidates": {backbone: len(candidates[backbone]) for backbone in backbones}, "pool_universe": self.args.pool_universe})

        # -- IV scores and baseline-encoding codes over the candidate set (one pass)
        iv_path = self._sel_path(PC_STEP, self.args.pool_universe, self.args.third_cohort if dataset == THIRD else "", dataset, "iv_scores")
        scores = read_json(iv_path)["scores"] if iv_path.exists() else None
        need_iv = scores is None and any(pipeline == "IV pool" for _, pipeline in pending)
        need_codes = any(pipeline == "Full mRMR" or dataset != LENDINGCLUB for _, pipeline in pending)
        codes: dict[str, np.ndarray] = {}
        disc: dict[str, float] = {}
        y: pd.Series | None = None
        if need_iv or need_codes:
            codes, disc, y, new_scores = self._baseline_pass(dataset, union, need_iv=need_iv, need_codes=need_codes, settings=settings)
            if need_iv:
                scores = new_scores
                write_json(iv_path, {"dataset": dataset, "universe": self.args.pool_universe, "n_candidates": len(union), "interval": IV_INTERVAL, "settings": iv_settings(dataset), "scores": scores, "created_at_utc": _now()})

        # -- pools
        pools: dict[str, dict[str, list[str]]] = {}
        sources: dict[str, str] = {}
        for backbone in backbones:
            size = LLM_POOL_BUDGET[backbone]
            cand = candidates[backbone]
            members = set(cand)
            pools[backbone] = {}
            ranking, source = self.llm_pool_ranking(dataset, backbone)
            sources[backbone] = source
            if ranking is not None:
                ordered = list(dict.fromkeys(ranking))
                outside = [name for name in ordered[:size] if name not in members]
                pools[backbone]["LLM pool"] = [name for name in ordered if name in members][:size]
                if outside:
                    self.manifest.note(PC_STEP, f"{dataset}/{backbone}: {len(outside)} LLM-ranked names lie outside the candidate set and were skipped, e.g. {outside[:3]}")
                if len(pools[backbone]["LLM pool"]) < size:
                    self.manifest.note(PC_STEP, f"{dataset}/{backbone}: the LLM pool holds {len(pools[backbone]['LLM pool'])} names, not P={size}, because the ranking behind it has {len(ordered)} names ({source}); the IV and random pools keep P={size}")
            else:
                self.manifest.skip(PC_STEP, {"dataset": dataset, "backbone": backbone, "pipeline": "LLM pool"}, f"LLM ranking unavailable ({source})")
            if scores is not None:
                pools[backbone]["IV pool"] = iv_pool(scores, cand, size)
                if len(pools[backbone]["IV pool"]) < size:
                    self.manifest.note(PC_STEP, f"{dataset}/{backbone}: only {len(pools[backbone]['IV pool'])} candidates have {IV_INTERVAL[0]} <= IV <= {IV_INTERVAL[1]}")
            for seed in RANDOM_SEEDS:
                pools[backbone][f"Random pool s{seed}"] = random_pool(cand, size, seed)
            pools[backbone]["Full mRMR"] = list(cand)

        # -- timed mRMR stages, one at a time, in skeleton order
        for backbone, pipeline in pending:
            pool = pools[backbone].get(pipeline)
            key = {"dataset": dataset, "backbone": backbone, "pipeline": pipeline}
            if not pool:
                self.manifest.skip(PC_STEP, key, "pool unavailable")
                continue
            k = BUDGETS[backbone]
            one_hot = dataset == LENDINGCLUB and pipeline != "Full mRMR"
            encode_seconds = None
            if one_hot:
                from scripts.todo_fill.data import dense_preprocess

                X_all, y_dev = self.dev_all(dataset)
                started = time.perf_counter()
                dense = dense_preprocess(X_all.loc[:, pool], dataset)
                encode_seconds = time.perf_counter() - started
                pool_codes, pool_disc = discretize(dense, int(settings.get("n_bins", 10)))
                order = [str(name) for name in dense.columns]
                del dense
                with heartbeat(f"mRMR {dataset}/{backbone}/{pipeline} ({len(order)} one-hot columns)"):
                    selected, greedy = mrmr_from_codes(pool_codes, order, y_dev, k, settings)
                discretize_seconds = sum(pool_disc.values())
                del pool_codes
            else:
                members = set(pool)
                order = [name for name in union if name in members]
                with heartbeat(f"mRMR {dataset}/{backbone}/{pipeline} ({len(order)} columns)"):
                    selected, greedy = mrmr_from_codes(codes, order, y, k, settings)
                discretize_seconds = sum(disc[name] for name in order)
            record = {
                **key,
                "k": k,
                "pool_size_target": LLM_POOL_BUDGET[backbone] if pipeline != "Full mRMR" else None,
                "n_pool": len(pool),
                "pool": list(pool),
                "selection_input": "one-hot (Preprocessor, cat_min_frequency 50)" if one_hot else "frozen selection encoding (OriginalFeatureNumericEncoder)",
                "n_selection_columns": len(order),
                "features": selected,
                "select_seconds": discretize_seconds + greedy,
                "discretize_seconds": discretize_seconds,
                "greedy_seconds": greedy,
                "encode_seconds_excluded": encode_seconds,
                "mi_settings": settings,
                "pool_universe": self.args.pool_universe,
                "n_candidates": len(candidates[backbone]),
                "llm_ranking_source": sources.get(backbone) if pipeline == "LLM pool" else None,
                "third_cohort": self.args.third_cohort if dataset == THIRD else None,
                "threads": SELECT_THREADS,
                "created_at_utc": _now(),
            }
            if len(selected) < k:
                self.manifest.note(PC_STEP, f"{key}: mRMR returned {len(selected)} of K={k} features")
            write_json(self._pc_path(dataset, backbone, pipeline), record)
            self.pc_sel[(dataset, backbone, pipeline)] = record
            log(f"[{PC_STEP}] {dataset}/{backbone}/{pipeline}: {len(order)} columns -> {len(selected)} in {record['select_seconds']:.1f}s")
        del codes
        gc.collect()

    def _baseline_pass(self, dataset: str, columns: list[str], *, need_iv: bool, need_codes: bool, settings: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, float], pd.Series, dict[str, float] | None]:
        """Encode the candidate set once (in column batches on Stability) for IV scores and MI codes."""

        from scripts.todo_fill.data import encode_for_selection

        n_bins = int(settings.get("n_bins", 10))
        iv_config = iv_settings(dataset)
        codes: dict[str, np.ndarray] = {}
        disc: dict[str, float] = {}
        scores: dict[str, float] | None = {} if need_iv else None
        timings = {"load_seconds": 0.0, "encode_seconds": 0.0, "iv_seconds": 0.0}
        y_ref: pd.Series | None = None
        if dataset == THIRD:
            batch = max(1, int(self.args.third_batch_columns))
            ids_ref: np.ndarray | None = None
            with heartbeat(f"Stability 2024 candidate pass ({len(columns)} columns in batches of {batch})"):
                for start in range(0, len(columns), batch):
                    chunk = columns[start : start + batch]
                    t0 = time.perf_counter()
                    X, y_batch, ids = self.third_slice(chunk)
                    timings["load_seconds"] += time.perf_counter() - t0
                    if ids_ref is None:
                        ids_ref, y_ref = ids, y_batch
                    elif not (np.array_equal(ids_ref, ids) and np.array_equal(y_ref.to_numpy(), y_batch.to_numpy())):
                        raise FillError("Stability 2024 column batches returned different row orders")
                    self._encode_batch(X, y_batch, encode_for_selection, need_iv, need_codes, n_bins, iv_config, codes, disc, scores, timings)
                    del X
                    gc.collect()
                    log(f"[{PC_STEP}] Stability 2024 candidate pass: {min(start + batch, len(columns))}/{len(columns)} columns")
        else:
            X_all, y_ref = self.dev_all(dataset)
            frame = X_all if list(X_all.columns) == list(columns) else X_all.loc[:, columns]  # no copy of the whole universe
            self._encode_batch(frame, y_ref, encode_for_selection, need_iv, need_codes, n_bins, iv_config, codes, disc, scores, timings)
            del frame
        assert y_ref is not None
        self.dataset_info.setdefault(dataset, {}).update({f"candidate_pass_{key}": value for key, value in timings.items()})
        self.manifest.timing(f"{PC_STEP}:{dataset}:candidate_pass", sum(timings.values()) + sum(disc.values()), **timings, discretize_seconds=sum(disc.values()), n_columns=len(columns))
        return codes, disc, y_ref, scores

    @staticmethod
    def _encode_batch(X: pd.DataFrame, y: pd.Series, encoder: Callable[..., pd.DataFrame], need_iv: bool, need_codes: bool, n_bins: int, iv_config: dict[str, Any], codes: dict[str, np.ndarray], disc: dict[str, float], scores: dict[str, float] | None, timings: dict[str, float]) -> None:
        t0 = time.perf_counter()
        encoded = encoder(X)
        timings["encode_seconds"] += time.perf_counter() - t0
        if need_iv and scores is not None:
            t0 = time.perf_counter()
            scores.update(iv_scores(encoded, y, iv_config))
            timings["iv_seconds"] += time.perf_counter() - t0
        if need_codes:
            batch_codes, batch_disc = discretize(encoded, n_bins)
            codes.update(batch_codes)
            disc.update(batch_disc)
        del encoded

    # ============================================================================ fits
    def _pc_spec(self, dataset: str, backbone: str, pipeline: str) -> FitSpec | None:
        record = self._load_pc(dataset, backbone, pipeline)
        if not record or not record.get("features"):
            return None
        return FitSpec(dataset, backbone, FULL_DEV, tuple(record["features"]), label=f"pc:{pipeline}")

    def run_fits(self) -> None:
        from scripts.todo_fill.gate4.fits import raw_bases

        specs = [spec for item in self.pc_items if (spec := self._pc_spec(*item)) is not None]
        if not specs:
            return
        needed: dict[str, set[str]] = {}
        for spec in specs:
            if spec.dataset != THIRD:
                needed.setdefault(spec.dataset, set()).update(raw_bases(spec.features, self.records.universe(spec.dataset)))
        for dataset, features in needed.items():  # materialise the locked HO frame once per dataset
            self.data.holdout(dataset, [name for name in self.records.universe(dataset) if name in features])
        started = time.perf_counter()
        self.executor.run(specs, "fits")
        self.manifest.timing("fits", time.perf_counter() - started, n_specs=len(set(specs)))
        for spec in set(specs):
            payload = self.executor.lookup(spec)
            if payload and payload.get("missing_dense_columns"):
                self.manifest.note("fits", f"{spec.dataset}/{spec.backbone} {spec.label}: the refit could not rebuild {payload['missing_dense_columns']} and ran without them")

    def fit(self, spec: FitSpec | None) -> dict[str, Any] | None:
        return None if spec is None else self.executor.lookup(spec)

    # ======================================================================= inference
    def _identity(self, dataset: str, n_scores: int) -> pd.DataFrame:
        """HO identity of this run's fits; rebuilt if a concurrent worker write left it unreadable."""

        from scripts.todo_fill.gate4.fits import identity_path, load_holdout

        path = identity_path(self.work_dir, dataset)
        try:
            frame = pd.read_parquet(path)
            if len(frame) == n_scores:
                return frame
        except Exception:  # noqa: BLE001
            pass
        probe = self.records.third_candidates()[0] if dataset == THIRD else self.records.universe(dataset)[0]
        _, target, ids = load_holdout(self.work_dir, dataset, [probe])
        frame = pd.DataFrame({"stable_row_id": ids.to_numpy(), "target": target.to_numpy(dtype=int)})
        if len(frame) != n_scores:
            raise FillError(f"{dataset} HO identity has {len(frame)} rows; the score vectors have {n_scores}")
        frame.to_parquet(path, index=False)
        return frame

    def run_inference(self) -> None:
        from scripts.todo_fill.gate4.fits import ho_scores
        from scripts.todo_fill.gate4.inference import paired_auc_inference

        for dataset, backbone, other in self.delta_items:
            key = {"dataset": dataset, "backbone": backbone, "comparison": f"LLM pool vs {other}"}
            spec_a, spec_b = self._pc_spec(dataset, backbone, "LLM pool"), self._pc_spec(dataset, backbone, other)
            if self.fit(spec_a) is None or self.fit(spec_b) is None:
                self.manifest.skip(PC_STEP, key, "a full-DEV refit of the pair is unavailable")
                continue
            path = self.work_dir / "gate4" / "inference" / f"{spec_a.fit_id}__{spec_b.fit_id}.json"
            if path.exists():
                self.inference[(dataset, backbone, other)] = read_json(path)
                continue
            if spec_a.fit_id == spec_b.fit_id:
                auc = float(self.fit(spec_a)["ho_auc"])
                result = {"auc_A": auc, "auc_B": auc, "delta": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "p_value": None, "n_ho": int(self.fit(spec_a)["n_ho"]), "note": "identical subsets"}
                self.manifest.note(PC_STEP, f"{key}: both pipelines selected the identical subset")
            else:
                a, b = ho_scores(self.work_dir, spec_a), ho_scores(self.work_dir, spec_b)
                identity = self._identity(dataset, len(a))
                log(f"[{PC_STEP}] paired bootstrap {dataset}/{backbone} LLM pool vs {other} on {len(a)} HO rows")
                result = paired_auc_inference(identity["target"].to_numpy(dtype=int), a, b)
            result.update({"fit_A": spec_a.fit_id, "fit_B": spec_b.fit_id, "created_at_utc": _now()})
            write_json(path, result)
            self.inference[(dataset, backbone, other)] = result

    # ========================================================================= outputs
    def _set(self, name: str, row: dict[str, str], column: str, value: str) -> None:
        if column not in row or value == "":
            return
        if row[column] == "":
            row[column] = value
        elif row[column] != value:
            self.manifest.note("output", f"{name} {dict((c, row[c]) for c in SCHEMA[name][0][:4])} {column}: kept the pre-filled {row[column]!r}; this run computed {value!r}")

    def filled(self) -> dict[str, Skeleton]:
        out = {name: copy.deepcopy(skeleton) for name, skeleton in self.skeletons.items()}
        for row in out[CALLS_CSV].rows:
            dataset = DATASETS.get(row["dataset"])
            if dataset not in (HOMECREDIT, LENDINGCLUB) or not row["call_budget"].isdigit() or not row["repeat"].isdigit():
                continue
            name = call_name(dataset, None if dataset == HOMECREDIT else int(row["call_budget"]), int(row["repeat"]))
            accepted, failed = self.ranker.cached(RR_STEP, name), self.ranker.failed(RR_STEP, name)
            record = accepted or failed
            if record is None:
                continue
            self._set(CALLS_CSV, row, "validation_passed", "True" if accepted else "False")
            usage = record.get("usage_per_attempt") or []
            known = [item for item in usage if item.get("prompt_tokens") is not None]
            if known:  # every application attempt of the call, accepted or not
                prompt = sum(int(item["prompt_tokens"]) for item in known)
                completion = sum(int(item.get("completion_tokens") or 0) for item in known)
            elif accepted and accepted.get("prompt_tokens") is not None:  # no per-attempt usage: the accepted response
                prompt, completion = accepted.get("prompt_tokens"), accepted.get("completion_tokens")
            else:
                prompt = completion = None
            if prompt is not None:
                self._set(CALLS_CSV, row, "input_tokens", str(int(prompt)))
            if completion is not None:
                self._set(CALLS_CSV, row, "output_tokens", str(int(completion)))
        for row in out[POOLS_CSV].rows:
            dataset, backbone = DATASETS.get(row["dataset"]), BACKBONES.get(row["backbone"])
            if not dataset or not backbone:
                continue
            record = self._load_pc(dataset, backbone, row["pipeline"])
            if record is not None:
                self._set(POOLS_CSV, row, "select_seconds", fmt(record["select_seconds"], 2))
            payload = self.fit(self._pc_spec(dataset, backbone, row["pipeline"]))
            if payload is not None:
                self._set(POOLS_CSV, row, "ho_auc", fmt(payload["ho_auc"], 8))
        for row in out[DELTAS_CSV].rows:
            dataset, backbone = DATASETS.get(row["dataset"]), BACKBONES.get(row["backbone"])
            other = row["comparison"].partition(" vs ")[2]
            result = self.inference.get((dataset, backbone, other)) if dataset and backbone else None
            if result is not None:
                self._set(DELTAS_CSV, row, "delta_auc", fmt(result["delta"], 8))
                self._set(DELTAS_CSV, row, "ci_low", fmt(result["ci95_low"], 8))
                self._set(DELTAS_CSV, row, "ci_high", fmt(result["ci95_high"], 8))
        return out

    def write_outputs(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        owned = {"all": set(SCHEMA), "repeats": {CALLS_CSV}, "pools": {POOLS_CSV, DELTAS_CSV}}[self.args.part]
        for name, skeleton in self.filled().items():
            if name not in owned:  # a part never rewrites the other part's files
                continue
            write_like_source(skeleton, _resolve(self.args.todo_dir) / name, self.out_dir / name)
            filled, total = skeleton.fill_rate(VALUE_COLUMNS[name])
            self.manifest.csv_summary[name] = {"filled": filled, "total": total, "status": "complete" if filled == total and total else ("partial" if filled else "empty")}
        try:
            self._write_sidecars()
        except Exception as exc:  # noqa: BLE001
            self.manifest.note("output", f"sidecar writing failed: {type(exc).__name__}: {exc}\n{traceback.format_exc()}")
        self.manifest.write()

    def _write_sidecars(self) -> None:
        folder = self.sidecar_dir
        folder.mkdir(parents=True, exist_ok=True)
        if self.plan:
            plan = Path(self.plan["path"])
            if plan.exists():
                shutil.copy2(plan, folder / plan.name)
            lines = [f"{key}: {self.plan.get(key)}" for key in ("path", "repo", "branch", "commit", "committed_at", "sha256", "committed_by_this_run")]
            (folder / "rerun_plan_commit.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        calls = [self.ranker.cached(RR_STEP, call_name(*item)) for item in self.call_items]
        calls = [call for call in calls if call is not None]
        if calls:
            ranking_rows = [
                {"dataset": call["dataset"], "call_budget": call.get("call_budget") or call.get("budget") or HC_CALL_BUDGET, "repeat": call.get("repeat_id"), "rank": rank, "feature": feature, "response_id": call.get("response_id")}
                for call in calls
                for rank, feature in enumerate(call["ranking"], start=1)
            ]
            _write_csv(folder / "responserepeats_rankings.csv", ranking_rows)
            write_json(folder / "responserepeats_calls_log.json", [{key: value for key, value in call.items() if key not in {"attempts", "ranking", "ranking_ids"}} for call in calls])
        pool_rows = []
        for (d, b, p), rec in sorted(self.pc_sel.items()):
            payload = self.fit(self._pc_spec(d, b, p))
            pool_rows.append({"dataset": d, "backbone": b, "pipeline": p, "n_candidates": rec.get("n_candidates"), "n_pool": rec.get("n_pool"), "selection_input": rec.get("selection_input"), "n_selection_columns": rec.get("n_selection_columns"), "k": rec.get("k"), "n_selected": len(rec["features"]), "select_seconds": rec.get("select_seconds"), "discretize_seconds": rec.get("discretize_seconds"), "greedy_seconds": rec.get("greedy_seconds"), "encode_seconds_excluded": rec.get("encode_seconds_excluded"), "ho_auc": payload["ho_auc"] if payload else "", "n_ho": payload["n_ho"] if payload else "", "fit_id": payload["fit_id"] if payload else "", "selected": _pipe(rec["features"]), "pool": _pipe(rec["pool"]) if p != "Full mRMR" else f"(the {rec.get('n_pool')}-feature candidate set)"})
        if pool_rows:
            _write_csv(folder / "poolcontrols_selections.csv", pool_rows)
        issues = self.ranker.write_contract_issues(folder / "contract_issues.csv")
        if issues:
            self.manifest.note(RR_STEP, f"{issues} attempt rows in contract_issues.csv")
        self._write_decisions(folder)
        (folder / "rerun_settings.txt").write_text(self.settings_text(), encoding="utf-8")

    def _paper_auc(self) -> dict[tuple[str, str, str], float]:
        path = _resolve(self.args.paper_matrix)
        if not path.exists():
            return {}
        with path.open(newline="", encoding="utf-8-sig") as handle:
            return {(row["dataset"], row["backbone"], row["selector"]): float(row["ho_auc"]) for row in csv.DictReader(handle) if row.get("ho_auc")}

    def _write_decisions(self, folder: Path) -> None:
        """Pool-control decision rule of the plan (the response-repeat rule needs the supplied AUC file, which is not read)."""

        rows: list[dict[str, Any]] = []
        for dataset, backbone in sorted({(d, b) for d, b, _ in self.delta_items}):
            iv = self.inference.get((dataset, backbone, "IV pool"))
            randoms = [float(p["ho_auc"]) for seed in RANDOM_SEEDS if (p := self.fit(self._pc_spec(dataset, backbone, f"Random pool s{seed}"))) is not None]
            full = self.inference.get((dataset, backbone, "Full mRMR"))
            if iv is None:
                continue
            delta, low, high = iv["delta"], iv["ci95_low"], iv["ci95_high"]
            if delta >= MATERIALITY and low > 0:
                verdict = "supports the semantic reading (LLM - IV >= +0.010, interval excludes 0)"
            elif abs(delta) <= MATERIALITY:
                verdict = "gain attributed to pre-screening (IV pool within +/-0.010 of LLM pool)"
            else:
                verdict = "neither rule applies (outside +/-0.010 without meeting the semantic criterion)"
            rows.append({"part": "pool controls", "rule": "LLM pool - IV pool", "dataset": DATASET_LABELS[dataset], "backbone": backbone, "llm_pipeline": "LLM pool", "ho_auc_median": iv["auc_A"], "iv_pool_ho_auc": iv["auc_B"], "delta_llm_minus_iv": delta, "ci_low": low, "ci_high": high, "random_pool_ho_auc_min": min(randoms) if randoms else "", "random_pool_ho_auc_max": max(randoms) if randoms else "", "delta_llm_minus_full_mrmr": full["delta"] if full else "", "verdict": verdict})
        if rows:
            _write_csv(folder / "rerun_decisions.csv", rows)

    def settings_text(self) -> str:
        import scripts.b6_obfuscation as helpers

        lines = [
            "Reruns of todo/README_reruns.md, filled by scripts/fill_todo_reruns.py",
            f"generated: {_now()}",
            f"plan: {self.plan.get('path') if self.plan else '-'} at commit {self.plan.get('commit') if self.plan else '-'} ({self.plan.get('committed_at') if self.plan else '-'})",
            "",
            f"ranking calls: snapshot {helpers.MODEL_SNAPSHOT}, temperature {helpers.TEMPERATURE}, evidence mode {helpers.EVIDENCE_MODE}, Appendix A target-free template, JSON response of exactly {helpers.RANKING_BUDGET} names, strict validation with three application attempts and no fallback; candidate records as in the cached run (Home Credit: the cached full-DEV list; LendingClub: the cached full-DEV list of the same budget).",
            "call structure: Home Credit one 100-name call per repeat (budgets 20/40/60 are prefixes); LendingClub separate 20-, 40-, 60- and 100-name calls per repeat, the prompt being identical across budgets as in the cached run.",
            "input_tokens/output_tokens: summed over every application attempt of the call (usage reported by the API); validation_passed False = all three attempts failed the contract; calls that never reached validation (transport errors) are re-issued, up to "
            f"{TRANSPORT_RETRY_ROUNDS} extra rounds per run.",
            "responserepeats_auc.csv: supplied already filled; not opened, written or refitted by this script (size and modification time checked before and after the run).",
            "",
            f"pool controls: P = {LLM_POOL_BUDGET} (LR/CatBoost), K = {BUDGETS}; candidate set = {self.args.pool_universe} ({json.dumps({k: v.get('n_candidates') for k, v in self.dataset_info.items()})}); IV pool = top P by IV/WOE (n_bins 10, quantile, smoothing 0.5) within {IV_INTERVAL[0]} <= IV <= {IV_INTERVAL[1]}; random pools = numpy default_rng(seed).choice without replacement, seeds {list(RANDOM_SEEDS)}; Full mRMR = the whole candidate set.",
            "mRMR stage: mrmr_mutual_information (discrete plug-in MI, n_bins 10, objective mid), frozen selection encoding on Home Credit/Stability 2024 and for Full mRMR, one-hot matrix for the pooled LendingClub pipelines.",
            f"select_seconds: discretisation + MI relevance + greedy of the mRMR stage; data loading and the selection encoding are excluded; discretisation is column-wise, timed once per column and summed over the pipeline's columns; {SELECT_THREADS} BLAS/OpenMP threads (the MI estimator itself is single-threaded); every timed stage ran alone, before any backbone fit.",
            f"Stability 2024 cohort: {self.args.third_cohort} {json.dumps(THIRD_COHORTS[self.args.third_cohort]) if THIRD_COHORTS[self.args.third_cohort] else '(protocol lock boundaries)'}; LLM pool from the regenerated named ranking of the Gate-4 fills.",
            "backbones: frozen configs/models/{lr,catboost}.yaml, seed 42, single-threaded, refitted on full DEV and scored once on HO (the Gate-4 fit routine).",
            "poolcontrols_delta.csv: LLM pool minus the other pipeline on paired HO rows, 95% percentile interval of 2,000 target-stratified resamples, numpy default_rng(20260721).",
            "",
            f"dataset details: {json.dumps(self.dataset_info, default=str)}",
        ]
        return "\n".join(lines) + "\n"

    # ==================================================================== checks
    def reproduction_notes(self) -> None:
        """Compare the re-runs of existing pipelines with the paper's printed values."""

        from scripts.todo_fill.gate4.subsets import load_appendix_e

        paper = self._paper_auc()
        appendix = load_appendix_e(_resolve(self.args.appendix_e)) if self.args.appendix_e else {}
        for dataset, backbone in sorted({(d, b) for d, b, _ in self.pc_items}):
            for pipeline, label in (("LLM pool", "LLM then mRMR"), ("Full mRMR", "mRMR")):
                payload = self.fit(self._pc_spec(dataset, backbone, pipeline))
                reference = paper.get((dataset, backbone, label))
                if payload is not None and reference is not None:
                    self.manifest.note(PC_STEP, f"{dataset}/{backbone}: {pipeline} HO AUC {payload['ho_auc']:.6f} on {payload['n_ho']} rows vs the paper's {label} {reference:.6f} (difference {payload['ho_auc'] - reference:+.6f})")
                record = self._load_pc(dataset, backbone, pipeline)
                subset = appendix.get((dataset, backbone, label))
                if record is not None and subset is not None:
                    overlap = len(set(record["features"]) & set(subset))
                    self.manifest.note(PC_STEP, f"{dataset}/{backbone}: {pipeline} selection shares {overlap} of {len(subset)} features with the Appendix E {label} subset")


def write_like_source(skeleton: Skeleton, source: Path, destination: Path) -> None:
    """Write a filled skeleton with the source file's line terminator and BOM (the todo CSVs use CRLF)."""

    raw = source.read_bytes()
    terminator = "\r\n" if b"\r\n" in raw else "\n"
    encoding = "utf-8-sig" if raw.startswith(b"\xef\xbb\xbf") else "utf-8"
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".partial")
    with partial.open("w", encoding=encoding, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=skeleton.columns, lineterminator=terminator)
        writer.writeheader()
        for row in skeleton.rows:
            writer.writerow({column: row.get(column, "") for column in skeleton.columns})
    os.replace(partial, destination)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = list(dict.fromkeys(column for row in rows for column in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


# ---------------------------------------------------------------------------- estimates


def _microbench() -> dict[str, tuple[float, float]]:
    """Seconds per MI evaluation and per column discretisation at each dataset's DEV size."""

    from sklearn.metrics import mutual_info_score

    from credit_risk_fs.selectors.lightweight.mi_mrmr import _discretize_column

    rng = np.random.default_rng(0)
    out: dict[str, tuple[float, float]] = {}
    for dataset, rows in ((HOMECREDIT, 99_092), (LENDINGCLUB, 598_649), (THIRD, 1_157_512)):
        a = rng.integers(0, 10, rows).astype(np.int8)
        b = rng.integers(0, 10, rows).astype(np.int8)
        started = time.perf_counter()
        for _ in range(3):
            mutual_info_score(a, b)
        mi = (time.perf_counter() - started) / 3
        column = pd.Series(rng.normal(size=rows).astype(np.float32))
        started = time.perf_counter()
        _discretize_column(column, 10)
        out[dataset] = (mi, time.perf_counter() - started)
    return out


def estimate_minutes(jobs: int, third_jobs: int, bench: dict[str, tuple[float, float]], n_candidates: dict[str, int]) -> list[tuple[str, float, str]]:
    from scripts.todo_fill.gate4 import config
    from scripts.todo_fill.gate4.estimates import _parallel

    def greedy(dataset: str, d: int, k: int) -> float:
        return bench[dataset][0] * (d + k * d - k * k / 2)

    items: list[tuple[str, float, str]] = [("50 ranking calls", 50 * 25.0 / 60, "~25 s per call incl. retries (step-45 calls took 20-30 s)")]
    for dataset in (HOMECREDIT, LENDINGCLUB, THIRD):
        d = n_candidates[dataset]
        mi, disc = bench[dataset]
        seconds = d * disc * 3.0  # encoding + IV + discretisation of the candidate set
        if dataset == THIRD:
            seconds += (d / 120 + 1) * 40.0  # parquet batches
        for backbone in ("lr", "catboost"):
            k, p = BUDGETS[backbone], LLM_POOL_BUDGET[backbone]
            width = 2 * p if dataset == LENDINGCLUB else p
            seconds += greedy(dataset, d, k) + 7 * (greedy(dataset, width, k) + (width * disc if dataset == LENDINGCLUB else 0))
        items.append((f"pool-control mRMR {DATASET_LABELS[dataset]}", seconds / 60, f"MI {mi * 1000:.1f} ms/eval, discretise {disc * 1000:.0f} ms/column, d={d}"))
    regular: list[float] = []
    third: list[float] = []
    for dataset, backbone, count in ((HOMECREDIT, "lr", 8), (HOMECREDIT, "catboost", 8), (LENDINGCLUB, "lr", 8), (LENDINGCLUB, "catboost", 8), (THIRD, "lr", 8), (THIRD, "catboost", 8)):
        seconds = config.fit_seconds(dataset, backbone, FULL_DEV)
        (third if dataset == THIRD else regular).extend([seconds] * count)
    items.append(("backbone refits", (_parallel(regular, jobs) + _parallel(third, third_jobs)) / 60, f"{len(regular) + len(third)} full-DEV fits; jobs={jobs}, third-jobs={third_jobs}"))
    boot = sum(14 * config.bootstrap_seconds(rows) for rows in (120_053, 293_105, 369_147))
    items.append(("paired bootstraps", boot / 60, "42 comparisons x 2,000 draws"))
    return items


# ----------------------------------------------------------------------------- preflight


def preflight(args: argparse.Namespace, skeletons: dict[str, Skeleton], *, shared_dir: Path, out_dir: Path) -> int:
    """Light checks only: no API call, no selector or backbone fit, nothing written to the repo."""

    from scripts.todo_fill import llm_cache
    from scripts.todo_fill.data import ThirdDataset, encode_for_selection
    from scripts.todo_fill.gate4.llm import prompt_sha

    results: list[tuple[str, str, str]] = []
    protected_folders = (_resolve(args.todo_dir), out_dir, out_dir / "final")
    protected_before = protected_stamps(protected_folders)

    def check(label: str, function: Callable[[], tuple[str, str]]) -> None:
        started = time.perf_counter()
        try:
            status, detail = function()
        except Exception as exc:  # noqa: BLE001
            status, detail = "FAIL", f"{type(exc).__name__}: {exc}"
        results.append((status, label, detail))
        print(f"[{status}] {label}: {detail} ({time.perf_counter() - started:.1f}s)", flush=True)

    def skeleton_check() -> tuple[str, str]:
        parts = []
        status = "PASS"
        for name, skeleton in skeletons.items():
            filled, total = skeleton.fill_rate(VALUE_COLUMNS[name])
            parts.append(f"{name} {len(skeleton.rows)} rows, {total - filled}/{total} value cells blank")
            if len(skeleton.rows) != SCHEMA[name][1]:
                status = "WARN"
        return status, "; ".join(parts)

    def outputs_check() -> tuple[str, str]:
        existing = [name for name in SCHEMA if (out_dir / name).exists()]
        return ("WARN" if existing else "PASS"), (f"{existing} already in {out_dir} and will be rewritten from the skeletons" if existing else f"no name collision in {out_dir}")

    def plan_check() -> tuple[str, str]:
        state = plan_state(Path(args.plan).expanduser())
        if not state["exists"]:
            return "FAIL", f"{args.plan} not found"
        if state.get("repo") is None:
            return "FAIL", "the plan is not inside a git repository"
        if state["tracked"] and not state["dirty"]:
            return "PASS", f"committed at {state['commit'][:12]} on {state['branch']}"
        return "WARN", f"{'modified' if state['tracked'] else 'untracked'} in {state['repo']} (branch {state['branch']}); --commit-plan commits that file only before anything runs"

    def api_check() -> tuple[str, str]:
        from scripts.todo_fill.common import load_dotenv_if_present

        load_dotenv_if_present()
        import openai  # noqa: F401

        present = bool(os.getenv("OPENAI_API_KEY"))
        return ("PASS" if present else "FAIL"), f"OPENAI_API_KEY {'present' if present else 'missing (rerun 1 would stay blank)'}; openai {openai.__version__} importable; validity not tested (no API call)"

    def cache_check() -> tuple[str, str]:
        missing = []
        for dataset in (HOMECREDIT, LENDINGCLUB):
            for item in ("dev_X.parquet", "dev_meta.parquet", "universe.json", "dev_dtypes.json", "ho.parquet"):
                if not (shared_dir / "frames" / dataset / item).exists():
                    missing.append(f"frames/{dataset}/{item}")
        for item in ("hcms2024_matrix/_SUCCESS", "hcms2024_matrix/metadata.json", "gate4/third_candidates.json", "gate4/rankings/shared/third_named_full_dev.json"):
            if not (shared_dir / item).exists():
                missing.append(item)
        if missing:
            return "FAIL", f"missing under {shared_dir}: {missing}"
        third = read_json(shared_dir / "gate4" / "third_candidates.json")["retained"]
        named = read_json(shared_dir / "gate4" / "rankings" / "shared" / "third_named_full_dev.json")
        outside = [name for name in named["ranking"] if name not in set(third)]
        return ("PASS" if len(third) == 1068 and not outside and len(named["ranking"]) == 100 else "WARN"), f"frames, HO caches and Stability matrix present; {len(third)} Stability candidates; regenerated Stability ranking {len(named['ranking'])} names, {len(outside)} outside the candidates"

    def ranking_check() -> tuple[str, str]:
        hc = llm_cache.ranking(HOMECREDIT, FULL_DEV)
        parts = [f"HC full-DEV {len(hc.features)} names over {len(hc.candidate_features)} candidates"]
        status = "PASS"
        for budget in (20, 40, 60, 100):
            cached = llm_cache.ranking(LENDINGCLUB, FULL_DEV, budget=budget)
            parts.append(f"LC b{budget}: {len(cached.features)} names/{len(cached.candidate_features)} candidates")
            if budget in LLM_POOL_BUDGET.values() and len(cached.features) < budget:
                status = "WARN"
                parts.append(f"(the LendingClub {'CatBoost' if budget == 100 else 'LR'} LLM pool will hold {len(cached.features)} names, not P={budget})")
        return status, "; ".join(parts)

    def cohort_check() -> tuple[str, str]:
        third = ThirdDataset(shared_dir / "hcms2024_matrix")
        lock = third.boundaries
        paper = THIRD_COHORTS["paper"]
        if int(lock["dev"]["rows"]) + int(lock["oot"]["rows"]) != paper["dev"]["rows"] + paper["oot"]["rows"]:
            return "FAIL", "the paper cohort does not partition the same rows as the lock"
        use_third_cohort("paper")
        try:
            dev, ho = third.date_range(FULL_DEV), third.date_range("holdout")
        finally:
            use_third_cohort("lock")
        restored = third.date_range(FULL_DEV)
        ok = dev == ("2019-01-01", "2020-01-31", 1_157_512) and ho == ("2020-02-01", "2020-10-05", 369_147) and restored[1] == str(lock["dev"]["date_max"])
        return ("PASS" if ok else "FAIL"), f"lock DEV ..{lock['dev']['date_max']} ({lock['dev']['rows']}) / HO {lock['oot']['date_min']}.. ({lock['oot']['rows']}); patched DEV {dev}, HO {ho}; row counts are verified again when the frames are read"

    def mi_check() -> tuple[str, str]:
        from credit_risk_fs.selectors.lightweight.mi_mrmr import MutualInformationMRMRSelector

        rng = np.random.default_rng(7)
        n = 4000
        frame = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.integers(0, 5, n).astype(float), "d": pd.Series(rng.choice(["x", "y", "z", None], n), dtype="object"), "e": np.where(rng.random(n) < 0.3, np.nan, rng.normal(size=n)), "f": 1.0})
        frame["g"] = frame["a"] * 0.9 + rng.normal(scale=0.1, size=n)
        frame["h"] = pd.Series(rng.choice([f"lvl{i}" for i in range(40)], n), dtype="object")
        y = pd.Series(((frame["a"] + (frame["d"] == "x") + rng.normal(size=n)) > 0.7).astype(int))
        encoded = encode_for_selection(frame)
        settings = mi_settings(HOMECREDIT)
        reference = MutualInformationMRMRSelector(k=6, **settings).fit(encoded, y)
        codes, _ = discretize(encoded, int(settings["n_bins"]))
        mine, _ = mrmr_from_codes(codes, list(encoded.columns), y, 6, settings)
        subset = ["h", "a", "e"]
        independent = np.array_equal(encode_for_selection(frame.loc[:, subset]).to_numpy(), encoded.loc[:, subset].to_numpy())
        ok = list(reference.selected_features_) == mine and independent
        return ("PASS" if ok else "FAIL"), f"precomputed-code mRMR {mine} vs MutualInformationMRMRSelector.fit {list(reference.selected_features_)}; selection encoding column-independent: {independent}; settings {settings}"

    def iv_check() -> tuple[str, str]:
        rng = np.random.default_rng(11)
        n = 3000
        frame = pd.DataFrame({f"x{i}": rng.normal(size=n) + (i % 3) * 0.1 for i in range(8)})
        y = pd.Series((frame["x0"] + rng.normal(size=n) > 0.5).astype(int))
        settings = iv_settings(HOMECREDIT)
        whole = iv_scores(frame, y, settings)
        split = {**iv_scores(frame.iloc[:, :3], y, settings), **iv_scores(frame.iloc[:, 3:], y, settings)}
        pool = iv_pool(whole, list(frame.columns), 3)
        seeds_ok = random_pool([f"c{i}" for i in range(50)], 10, 101) == random_pool([f"c{i}" for i in range(50)], 10, 101)
        return ("PASS" if whole == split and seeds_ok else "FAIL"), f"column-batched IV equals one fit: {whole == split}; top-3 IV pool {pool}; seeded random pools reproducible: {seeds_ok}; settings {settings}"

    def protected_check() -> tuple[str, str]:
        present = {path: stamp for path, stamp in protected_before.items() if stamp is not None}
        return "PASS", f"{list(PROTECTED_FILES)} is outside the script's file set (read and written: {list(SCHEMA)}); copies present: {sorted(present) or 'none'}; their size/mtime are re-checked at the end of this preflight and of every run"

    def prompt_check() -> tuple[str, str]:
        from scripts.todo_fill.data import DataContext
        from scripts.todo_fill.gate4.records import RecordFactory

        manifest = Manifest(output_dir=Path(tempfile.gettempdir()) / "fill_todo_reruns_preflight")
        records = RecordFactory(DataContext(shared_dir), shared_dir, manifest, candidate_source="cache")
        parts = []
        status = "PASS"
        for dataset, budget in ((HOMECREDIT, None), (LENDINGCLUB, 20), (LENDINGCLUB, 100)):
            names = records.candidate_set(dataset, FULL_DEV, budget)
            digest = prompt_sha(records, records.named_records(dataset, names), names)
            reference = shared_dir / "gate4" / "rankings" / "45" / f"{call_name(dataset, budget, 1)}.json"
            same = str(read_json(reference).get("prompt_sha256")) == digest if reference.exists() else None
            if same is False:
                status = "WARN"
            parts.append(f"{dataset}{'/b' + str(budget) if budget else ''}: {len(names)} candidates, prompt {digest[:12]}{'' if same is None else (' = step-45 prompt' if same else ' != step-45 prompt')}")
        return status, "; ".join(parts)

    def candidates_check() -> tuple[str, str]:
        universe = {dataset: len(read_json(shared_dir / "frames" / dataset / "universe.json")) for dataset in (HOMECREDIT, LENDINGCLUB)}
        llm = {HOMECREDIT: len(llm_cache.ranking(HOMECREDIT, FULL_DEV).candidate_features), LENDINGCLUB: len(llm_cache.ranking(LENDINGCLUB, FULL_DEV, budget=60).candidate_features)}
        return "PASS", f"--pool-universe {args.pool_universe}: baseline HC {universe[HOMECREDIT]}, LC {universe[LENDINGCLUB]}, Stability 1068; llm HC {llm[HOMECREDIT]}, LC {llm[LENDINGCLUB]}, Stability 1068 (the paper states 373/675/1,068)"

    def writer_check() -> tuple[str, str]:
        with tempfile.TemporaryDirectory() as folder:
            work = Path(folder) / "work"
            prepare_work_dir(work, shared_dir)
            manifest = Manifest(output_dir=Path(folder) / "sidecars")
            run = Reruns(args, skeletons, manifest, work_dir=work, shared_dir=shared_dir, out_dir=Path(folder) / "out", sidecar_dir=Path(folder) / "sidecars", plan=None)
            counts = {"calls": len(run.call_items), "pool rows": len(run.pc_items), "deltas": len(run.delta_items)}
            run.write_outputs()
            same = all((Path(folder) / "out" / name).read_bytes() == (_resolve(args.todo_dir) / name).read_bytes() for name in SCHEMA)
            skips = len(manifest.skips)
            # every value cell reachable, nothing else touched: inject fake results into the fill path
            run.fit = lambda spec: {"status": "ok", "ho_auc": 0.75, "n_ho": 10, "fit_id": "f"} if spec is not None else None
            run._load_pc = lambda d, b, p: {"features": ["a"], "select_seconds": 1.234, "pool": ["a"], "n_pool": 1}
            run.ranker.cached = lambda step, name: {"ranking": ["a"], "usage_per_attempt": [{"prompt_tokens": 10, "completion_tokens": 2}, {"prompt_tokens": 11, "completion_tokens": 3}]}
            run.inference = {item: {"delta": 0.01, "ci95_low": -0.001, "ci95_high": 0.02} for item in run.delta_items}
            problems = []
            for name, skeleton in run.filled().items():
                original = skeletons[name]
                if skeleton.columns != original.columns or len(skeleton.rows) != len(original.rows):
                    problems.append(f"{name}: layout changed")
                for filled_row, source_row in zip(skeleton.rows, original.rows):
                    for column in original.columns:
                        if column in VALUE_COLUMNS[name] and filled_row[column] == "":
                            problems.append(f"{name}: {column} unreachable in {dict(list(source_row.items())[:4])}")
                        if column not in VALUE_COLUMNS[name] and filled_row[column] != source_row[column]:
                            problems.append(f"{name}: dimension {column} changed")
            sample = run.filled()[CALLS_CSV].rows[0]
            # a call that failed the contract in all three attempts still fills all three cells
            run.ranker.cached = lambda step, name: None
            run.ranker.failed = lambda step, name: {"usage_per_attempt": [{"prompt_tokens": 7, "completion_tokens": 1}] * 3}
            failed_ok = all(row["validation_passed"] == "False" and row["input_tokens"] == "21" and row["output_tokens"] == "3" for row in run.filled()[CALLS_CSV].rows)
        ok = same and not skips and not problems and sample["input_tokens"] == "21" and sample["validation_passed"] == "True" and failed_ok
        return ("PASS" if ok else "FAIL"), f"parsed {counts}, {skips} unrecognised rows; nothing computed -> CSVs byte-identical to the skeletons (CRLF kept): {same}; fake results reach every value cell without touching layout or dimension cells: {not problems}{' ' + str(problems[:3]) if problems else ''}; tokens summed over attempts: {sample['input_tokens']}; contract-failed calls fill all three cells: {failed_ok}"

    def ram_check() -> tuple[str, str]:
        free = available_ram_gb()
        need = frame_gib(598_649, 675, 8) * 1.3 + frame_gib(598_649, 675, 4) + 2
        return ("PASS" if free >= need else "WARN"), f"{free:.1f} GiB available; LendingClub candidate pass needs about {need:.1f} GiB, a Stability batch of {args.third_batch_columns} columns about {frame_gib(1_157_512, args.third_batch_columns + 5, 8) * 2 + 1.3:.1f} GiB"

    print(f"preflight of scripts/fill_todo_reruns.py (shared caches: {shared_dir})", flush=True)
    for label, function in (
        ("skeletons", skeleton_check),
        ("output names", outputs_check),
        ("rerun plan", plan_check),
        ("API key", api_check),
        ("cached inputs", cache_check),
        ("cached LLM rankings", ranking_check),
        ("Stability cohort", cohort_check),
        ("MI mRMR equivalence", mi_check),
        ("IV and random pools", iv_check),
        ("protected AUC file", protected_check),
        ("prompt rendering", prompt_check),
        ("candidate sets", candidates_check),
        ("CSV writer", writer_check),
        ("memory", ram_check),
    ):
        check(label, function)
    bench = _microbench()
    universe = {HOMECREDIT: 529, LENDINGCLUB: 675, THIRD: 1068} if args.pool_universe == "baseline" else {HOMECREDIT: 391, LENDINGCLUB: 161, THIRD: 1068}
    items = estimate_minutes(args.jobs, args.third_jobs, bench, universe)
    print("\nestimated wall-clock (rough; unit costs measured now or in the Gate-4 fills):", flush=True)
    for label, minutes, note in items:
        print(f"  {label:38s} {minutes:7.0f} min  {note}")
    print(f"  {'total':38s} {sum(minutes for _, minutes, _ in items):7.0f} min  (resumable; re-run the same command after an interruption)")
    unchanged = protected_stamps(protected_folders) == protected_before
    results.append(("PASS" if unchanged else "FAIL", "protected AUC file after preflight", ""))
    print(f"[{'PASS' if unchanged else 'FAIL'}] protected AUC file after preflight: size and mtime {'unchanged' if unchanged else 'CHANGED'} for {sorted(path for path, stamp in protected_before.items() if stamp is not None)}")
    failed = [label for status, label, _ in results if status == "FAIL"]
    print(f"\npreflight: {len(results) - len(failed)}/{len(results)} checks without failure" + (f"; FAILED: {failed}" if failed else ""))
    return 1 if failed else 0


# ---------------------------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--todo-dir", default="todo", help=f"folder with the skeletons {list(SCHEMA)}; {PROTECTED_FILES[0]} there is never opened (default: todo)")
    parser.add_argument("--output-dir", default="todo_done", help="folder that receives the filled CSVs under their own names (default: todo_done)")
    parser.add_argument("--sidecar-dir", default="todo_done/reruns_run", help="manifest, log, call rankings, pools, decisions, plan commit (default: todo_done/reruns_run)")
    parser.add_argument("--work-dir", default="outputs/todo_fill/reruns", help="checkpoints: calls, selections, fits, bootstraps (default: outputs/todo_fill/reruns)")
    parser.add_argument("--shared-work-dir", default="outputs/todo_fill", help="Gate-4 caches reused read-only: frames, Stability matrix, regenerated Stability ranking")
    parser.add_argument("--plan", default="~/projects/article/rerun_plan.md", help="the decision rules of Step 0 (default: ~/projects/article/rerun_plan.md)")
    parser.add_argument("--commit-plan", action="store_true", help="commit the plan (that file only) if it is untracked or modified, before anything runs")
    parser.add_argument("--part", default="all", choices=["all", "repeats", "pools"], help=f"all (default); repeats = the 50 calls, writing {CALLS_CSV} only; pools = the pool controls only")
    parser.add_argument("--pool-universe", default="baseline", choices=["baseline", "llm"], help="candidate set of the pool controls (see the module docstring)")
    parser.add_argument("--third-cohort", default="paper", choices=list(THIRD_COHORTS), help="Stability 2024 DEV/HO cutoff: the paper's 2020-02-01 (default) or the local lock's 2020-02-26")
    parser.add_argument("--third-batch-columns", type=int, default=120, help="columns per Stability 2024 read in the candidate pass (default: 120)")
    parser.add_argument("--jobs", type=int, default=6, help="parallel backbone fits for Home Credit / LendingClub; -1 = all cores (default: 6)")
    parser.add_argument("--third-jobs", type=int, default=3, help="parallel Stability 2024 fits (more RAM each; default: 3)")
    parser.add_argument("--no-api", action="store_true", help="never call the API (rerun 1 cells stay blank)")
    parser.add_argument("--reissue-failed-calls", action="store_true", help="re-issue calls that failed the contract in a previous run (a deviation from the frozen protocol; off by default)")
    parser.add_argument("--paper-matrix", default="~/projects/article/priority_1/full_matrix.csv", help="the paper's HO AUCs, used for the decision summary and reproduction notes")
    parser.add_argument("--appendix-e", default="~/projects/article/appendix_e_subsets.csv", help="the paper's headline subsets, used for reproduction notes")
    parser.add_argument("--preflight", action="store_true", help="light checks and time estimates only; no API call, no fit, nothing written to the repository")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cores = os.cpu_count() or 1
    args.jobs = cores if args.jobs < 1 else args.jobs
    args.third_jobs = cores if args.third_jobs < 1 else args.third_jobs
    todo_dir, out_dir, work_dir, shared_dir, sidecar_dir = (_resolve(value) for value in (args.todo_dir, args.output_dir, args.work_dir, args.shared_work_dir, args.sidecar_dir))
    skeletons = read_skeletons(todo_dir)
    if args.preflight:
        return preflight(args, skeletons, shared_dir=shared_dir, out_dir=out_dir)

    plan = ensure_plan_committed(Path(args.plan).expanduser(), args.commit_plan)
    prepare_work_dir(work_dir, shared_dir)
    marker = work_dir / "gate4" / "third_cohort.txt"  # fit ids do not carry the cohort, so one store serves one cohort
    if marker.exists() and marker.read_text(encoding="utf-8").strip() != args.third_cohort:
        raise SystemExit(f"{work_dir} holds fits of the Stability cohort '{marker.read_text(encoding='utf-8').strip()}'; use another --work-dir for '{args.third_cohort}'")
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(args.third_cohort + "\n", encoding="utf-8")
    configure_log(work_dir / "fill_todo_reruns.log")
    use_third_cohort(args.third_cohort)
    manifest = Manifest(output_dir=sidecar_dir)
    manifest.settings = {**vars(args), "repo_root": str(REPO_ROOT), "python": sys.version.split()[0], "plan": plan, "select_threads": SELECT_THREADS, "started_at_utc": _now()}
    log(f"fill_todo_reruns: todo={todo_dir} out={out_dir} work={work_dir} part={args.part} pool-universe={args.pool_universe} third-cohort={args.third_cohort}; plan {plan['relative']} at {plan['commit'][:12]}")
    protected_folders = (todo_dir, out_dir, out_dir / "final")
    protected_before = protected_stamps(protected_folders)
    manifest.settings["protected_files_before"] = protected_before
    run = Reruns(args, skeletons, manifest, work_dir=work_dir, shared_dir=shared_dir, out_dir=out_dir, sidecar_dir=sidecar_dir, plan=plan)
    if not run.ranker.enabled and args.part in ("all", "repeats"):
        manifest.note("setup", f"API calls disabled (no OPENAI_API_KEY or --no-api): {CALLS_CSV} stays blank")

    def check_protected() -> None:
        after = protected_stamps(protected_folders)
        if after != protected_before:
            manifest.note("setup", f"PROTECTED FILE CHANGED DURING THE RUN (not by this script, which never opens it): before {protected_before}, after {after}")
        else:
            manifest.note("setup", f"{list(PROTECTED_FILES)} untouched: size and mtime unchanged for {sorted(path for path, stamp in after.items() if stamp is not None)}")

    def phase(label: str, function: Callable[[], None]) -> None:
        started = time.perf_counter()
        log(f"==== {label}")
        try:
            function()
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001 - one failed phase never loses the others
            manifest.note(label, f"failed: {type(exc).__name__}: {exc}\n{traceback.format_exc()}")
        manifest.timing(f"phase:{label}", time.perf_counter() - started)
        run.write_outputs()

    try:
        if args.part in ("all", "repeats"):
            phase("rerun 1: ranking calls", run.make_calls)
        if args.part in ("all", "pools"):
            for dataset in (HOMECREDIT, LENDINGCLUB, THIRD):
                if any(item[0] == dataset for item in run.pc_items):
                    phase(f"rerun 2: timed mRMR stages ({DATASET_LABELS[dataset]})", lambda dataset=dataset: run.pool_selections(dataset))
                run.release(dataset)
            phase("rerun 2: backbone refits", run.run_fits)
            phase("rerun 2: paired bootstraps", run.run_inference)
            phase("reproduction notes", run.reproduction_notes)
    except KeyboardInterrupt:
        log("interrupted; writing what is available (re-run the same command to resume)")
        run.write_outputs()
        check_protected()
        manifest.write()
        return 130
    manifest.note("setup", f"ranking calls made in this run: {run.ranker.calls_made}; contract failures: {run.ranker.calls_failed}")
    check_protected()
    run.write_outputs()
    for name, summary in manifest.csv_summary.items():
        log(f"{name}: {summary['filled']}/{summary['total']} value cells ({summary['status']})")
    log(f"done; filled CSVs in {out_dir}, provenance in {sidecar_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
