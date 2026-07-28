"""Prompt 7 tiny-fixture integration pass for the lightweight selector controls.

Runs every registered Prompt 7 method end to end through the real registry, the
real feature-budget wiring, and the real atomic artifact writers, then reloads
each artifact and checks that identity, order, counts, scores, hashes, seed, and
selection mode survived the round trip.

Uses generated deterministic data only. No Home Credit file, no LendingClub file,
no OOT split, no saved prediction, and no model is touched. Artifacts are written
to a scratch directory and the script asserts that nothing landed in the active
research results tree or the frozen legacy bundle.

    python scripts/verify_lightweight_selectors.py [--scratch DIR] [--audit DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from credit_risk_fs.experiments.atomic_io import (  # noqa: E402
    write_csv_atomic,
    write_json_atomic,
)
from credit_risk_fs.experiments.config import (  # noqa: E402
    apply_feature_budget_to_selector_kwargs,
)
from credit_risk_fs.experiments.result_paths import (  # noqa: E402
    AUDITED_LEGACY_RESULTS_ROOT,
    reject_historical_write,
)
from credit_risk_fs.selectors.lightweight.contract import SelectionResult  # noqa: E402
from credit_risk_fs.selectors.lightweight.registry import (  # noqa: E402
    lightweight_method_ids,
    registry_snapshot,
    validate_method_selection_mode,
)
from credit_risk_fs.selectors.registry import get_selector  # noqa: E402

DEFAULT_AUDIT_ROOT = REPOSITORY_ROOT / "cleanup/audits/prompt_07_lightweight_selectors"
ACTIVE_RESULTS_ROOT = REPOSITORY_ROOT / "results"

FIXTURE_SEED = 20260728
FIXTURE_ROWS = 600
FEATURE_BUDGET = 4

#: Columns the fold pipeline never offers as candidates. Passed to every selector
#: so the exclusion guard is exercised on the real path rather than only in tests.
EXCLUDED_COLUMNS = ("SK_ID_CURR", "TARGET", "split_label", "time_index")


def build_fixture() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Deterministic synthetic fold: signal, redundancy, noise, degeneracy, gaps."""

    generator = np.random.default_rng(FIXTURE_SEED)
    n = FIXTURE_ROWS
    latent = generator.normal(size=n)
    logit = 1.8 * latent - 0.4
    target = pd.Series(
        (1.0 / (1.0 + np.exp(-logit)) > generator.random(n)).astype(int), name="TARGET"
    )

    candidates = pd.DataFrame(
        {
            # Ordered deliberately so the stable candidate list is not alphabetical;
            # a selector that leaks column order shows up immediately.
            "strong_signal": latent,
            "redundant_copy": latent + generator.normal(scale=0.02, size=n),
            "weak_signal": latent + generator.normal(scale=2.5, size=n),
            "pure_noise": generator.normal(size=n),
            "constant_column": np.ones(n),
            "sparse_with_gaps": np.where(
                generator.random(n) < 0.35, np.nan, latent + generator.normal(scale=1.0, size=n)
            ),
        }
    )

    metadata = pd.DataFrame(
        {
            "SK_ID_CURR": np.arange(100_000, 100_000 + n),
            "TARGET": target.to_numpy(),
            "split_label": ["dev_fold_train"] * n,
            "time_index": np.repeat(np.arange(n // 20), 20)[:n],
        }
    )
    return candidates, target, metadata


def _peak_rss_mb() -> float | None:
    try:
        import psutil
    except ImportError:  # pragma: no cover - psutil is an existing dependency
        return None
    return round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 1)


def _assert_scratch_is_isolated(scratch: Path) -> None:
    resolved = scratch.resolve()
    reject_historical_write(resolved)
    if resolved.is_relative_to(AUDITED_LEGACY_RESULTS_ROOT):
        raise SystemExit(f"refusing to write inside the frozen legacy bundle: {resolved}")
    if ACTIVE_RESULTS_ROOT.exists() and resolved.is_relative_to(ACTIVE_RESULTS_ROOT.resolve()):
        raise SystemExit(f"refusing to write inside the active results tree: {resolved}")


def run_one(
    method_id: str,
    candidates: pd.DataFrame,
    target: pd.Series,
    scratch: Path,
) -> dict[str, Any]:
    record: dict[str, Any] = {"method_id": method_id}
    started = time.perf_counter()

    selector_cls, kwargs = get_selector(method_id)
    kwargs = apply_feature_budget_to_selector_kwargs(method_id, kwargs, FEATURE_BUDGET)
    kwargs["excluded_columns"] = list(EXCLUDED_COLUMNS)

    selector = selector_cls(**kwargs)
    selector.fit(candidates, target if selector.supervised else None)
    result = selector.result
    validate_method_selection_mode(method_id, result.selection_mode)

    # Determinism: a second identical fit must reproduce the ordered result.
    repeat = selector_cls(**kwargs)
    repeat.fit(candidates, target if repeat.supervised else None)
    deterministic = (
        repeat.result.selected_features == result.selected_features
        and repeat.result.ranking == result.ranking
        and repeat.result.candidate_universe_sha256 == result.candidate_universe_sha256
    )

    # Real atomic serializers, then reload and compare.
    json_path = scratch / f"{method_id}_selection_result.json"
    csv_path = scratch / f"{method_id}_selection_ranking.csv"
    json_meta = write_json_atomic(json_path, result.to_dict())
    long_frame = result.to_long_frame()
    csv_meta = write_csv_atomic(csv_path, long_frame)

    restored = SelectionResult.from_json(json_path.read_text(encoding="utf-8"))
    reloaded_frame = pd.read_csv(csv_path)

    scores_match = True
    if result.raw_scores is not None and restored.raw_scores is not None:
        scores_match = all(
            restored.raw_scores[name] == value for name, value in result.raw_scores.items()
        )

    serialization_exact = (
        restored.method_id == result.method_id
        and restored.implementation_id == result.implementation_id
        and restored.selection_mode == result.selection_mode
        and restored.selected_features == result.selected_features
        and restored.ranking == result.ranking
        and restored.candidate_universe == result.candidate_universe
        and restored.candidate_universe_sha256 == result.candidate_universe_sha256
        and restored.requested_budget == result.requested_budget
        and restored.budget_status == result.budget_status
        and restored.natural_selected == result.natural_selected
        and restored.seed == result.seed
        and restored.training_identity_sha256 == result.training_identity_sha256
        and scores_match
        and len(reloaded_frame) == len(long_frame)
        and list(reloaded_frame["feature"]) == list(long_frame["feature"])
        and list(reloaded_frame["rank"]) == list(long_frame["rank"])
    )

    inside_universe = set(result.selected_features).issubset(set(candidates.columns))
    excluded_clean = not (set(result.selected_features) & set(EXCLUDED_COLUMNS))

    record.update(
        {
            "status": "PASS"
            if deterministic and serialization_exact and inside_universe and excluded_clean
            else "FAIL",
            "display_label": result.display_label,
            "implementation_id": result.implementation_id,
            "selection_mode": result.selection_mode,
            "supervised": result.supervised,
            "fit_scope": result.fit_scope,
            "requested_budget": result.requested_budget,
            "actual_selected_count": result.actual_selected_count,
            "natural_selected_count": result.natural_selected_count,
            "budget_status": result.budget_status,
            "selected_features": list(result.selected_features),
            "ranking_length": len(result.ranking or ()),
            "candidate_universe_count": result.candidate_universe_count,
            "candidate_universe_sha256": result.candidate_universe_sha256,
            "training_identity_sha256": result.training_identity_sha256,
            "training_row_count": result.training_row_count,
            "seed": result.seed,
            "tie_rule": result.tie_rule,
            "score_orientation": result.score_orientation,
            "deterministic_on_refit": deterministic,
            "serialization_exact": serialization_exact,
            "selection_inside_candidate_universe": inside_universe,
            "excluded_columns_absent_from_selection": excluded_clean,
            "artifacts": {
                "json": {"path": json_path.name, "sha256": json_meta.sha256},
                "csv": {"path": csv_path.name, "sha256": csv_meta.sha256},
            },
            "warnings": list(result.warnings),
            "failure_reason": result.failure_reason,
            "fit_seconds": round(float(result.fit_seconds), 6),
            "elapsed_seconds": round(time.perf_counter() - started, 6),
        }
    )
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scratch",
        type=Path,
        default=Path(os.environ.get("TEMP", "/tmp")) / "prompt_07_selector_fixture",
        help="scratch directory for artifact round-trip checks",
    )
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT_ROOT)
    arguments = parser.parse_args(argv)

    scratch = arguments.scratch
    scratch.mkdir(parents=True, exist_ok=True)
    _assert_scratch_is_isolated(scratch)

    candidates, target, metadata = build_fixture()
    started = time.perf_counter()

    records = [
        run_one(method_id, candidates, target, scratch)
        for method_id in lightweight_method_ids()
        if method_id != "legacy_rf_relevance_corr"  # exercised by its own suite
    ]

    failures = [record for record in records if record["status"] != "PASS"]
    payload = {
        "schema_version": "prompt_07_tiny_fixture_results_v1",
        "status": "PASS" if not failures else "FAIL",
        "generated_at_utc": pd.Timestamp.now("UTC").isoformat(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "fixture": {
            "kind": "deterministic_synthetic_only",
            "seed": FIXTURE_SEED,
            "rows": int(len(candidates)),
            "candidate_order": list(candidates.columns),
            "candidate_count": int(candidates.shape[1]),
            "excluded_columns": list(EXCLUDED_COLUMNS),
            "metadata_columns_supplied_separately": list(metadata.columns),
            "target_positive_rate": round(float(target.mean()), 6),
            "feature_budget": FEATURE_BUDGET,
            "contains_strong_signal": True,
            "contains_redundant_copy": True,
            "contains_noise": True,
            "contains_constant": True,
            "contains_missing_values": True,
            "real_dataset_loaded": False,
            "oot_data_loaded": False,
        },
        "isolation": {
            "scratch_directory": str(scratch),
            "inside_active_results_root": False,
            "inside_legacy_results_root": False,
        },
        "selectors": records,
        "failure_count": len(failures),
        "total_elapsed_seconds": round(time.perf_counter() - started, 6),
        "peak_process_rss_mb": _peak_rss_mb(),
    }

    arguments.audit.mkdir(parents=True, exist_ok=True)
    write_json_atomic(arguments.audit / "tiny_fixture_results.json", payload)
    write_json_atomic(arguments.audit / "selector_registry_snapshot.json", registry_snapshot())

    for record in records:
        print(
            f"{record['status']:4s} {record['method_id']:24s} "
            f"mode={record['selection_mode']:19s} "
            f"selected={record['actual_selected_count']}/"
            f"{record['requested_budget']} "
            f"status={record['budget_status']}"
        )
    print()
    print(f"tiny fixture: {payload['status']} ({len(records)} selectors, {len(failures)} failures)")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
