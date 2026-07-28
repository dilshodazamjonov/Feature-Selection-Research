"""Prompt 8 synthetic integration fixture for the heavy selectors.

Runs the three heavy methods end to end through the real registry, the real
feature-budget wiring, and the real atomic artifact writers, then reloads each
artifact and checks that identity, order, support states, budget semantics, seed,
explanation-sample identity, and estimator hashes survived the round trip.

Deterministic synthetic data only. No Home Credit file, no LendingClub file, no
OOT split, no saved prediction, and no real fold. Estimator profiles are
deliberately tiny **synthetic-test profiles** and are NOT the frozen research
configuration -- Prompt 9 measures real single-fold cost and freezes production.

    python scripts/verify_heavy_selectors.py [--scratch DIR] [--audit DIR]
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
    get_method_descriptor,
    validate_method_selection_mode,
)
from credit_risk_fs.selectors.registry import get_selector  # noqa: E402

DEFAULT_AUDIT_ROOT = REPOSITORY_ROOT / "cleanup/audits/prompt_08_heavy_selectors"
ACTIVE_RESULTS_ROOT = REPOSITORY_ROOT / "results"

FIXTURE_SEED = 20260729
FIXTURE_ROWS = 500
FEATURE_BUDGET = 3

EXCLUDED_COLUMNS = ("SK_ID_CURR", "TARGET", "split_label", "time_index")

#: Synthetic-test estimator profiles. NOT the frozen research configuration.
TEST_CATBOOST_PROFILE = {"iterations": 30, "depth": 3}
TEST_FOREST_PROFILE = {"n_estimators": 40, "max_depth": 4}
TEST_BORUTA_PROFILE = {"max_iter": 10}


def build_fixture() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Deterministic synthetic fold with every shape the heavy methods must face."""

    generator = np.random.default_rng(FIXTURE_SEED)
    n = FIXTURE_ROWS
    linear = generator.normal(size=n)
    nonlinear = generator.normal(size=n)
    # Nonlinear term CatBoost can exploit but a linear model cannot.
    logit = 2.0 * linear + 1.7 * (nonlinear**2 - 1.0) - 0.3
    target = pd.Series(
        (1.0 / (1.0 + np.exp(-logit)) > generator.random(n)).astype(int), name="TARGET"
    )

    candidates = pd.DataFrame(
        {
            # Non-alphabetical on purpose: a column-order leak would show up.
            "zulu_linear_signal": linear,
            "delta_nonlinear_signal": nonlinear,
            "mike_redundant_copy": linear + generator.normal(scale=0.02, size=n),
            "alpha_noise": generator.normal(size=n),
            "bravo_noise": generator.normal(size=n),
            "charlie_constant": np.ones(n),
            "echo_sparse_with_gaps": np.where(
                generator.random(n) < 0.30,
                np.nan,
                linear + generator.normal(scale=0.8, size=n),
            ),
        }
    )
    metadata = pd.DataFrame(
        {
            "SK_ID_CURR": np.arange(500_000, 500_000 + n),
            "TARGET": target.to_numpy(),
            "split_label": ["dev_fold_train"] * n,
            "time_index": np.repeat(np.arange(n // 20), 20)[:n],
        }
    )
    return candidates, target, metadata


def impute_for_finite_engines(frame: pd.DataFrame) -> pd.DataFrame:
    """Median-impute for engines that reject non-finite input.

    BorutaPy delegates to a scikit-learn forest, which refuses NaN; the legacy
    ``BorutaSelector`` shares that constraint. The imputation is done here, in the
    harness, and recorded -- the selector itself refuses non-finite input rather
    than imputing silently, which would be a hidden preprocessing change.
    """

    return frame.fillna(frame.median(numeric_only=True))


def _peak_rss_mb() -> float | None:
    try:
        import psutil
    except ImportError:  # pragma: no cover
        return None
    return round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 1)


def _assert_scratch_is_isolated(scratch: Path) -> None:
    resolved = scratch.resolve()
    reject_historical_write(resolved)
    if resolved.is_relative_to(AUDITED_LEGACY_RESULTS_ROOT):
        raise SystemExit(f"refusing to write inside the frozen legacy bundle: {resolved}")
    if ACTIVE_RESULTS_ROOT.exists() and resolved.is_relative_to(ACTIVE_RESULTS_ROOT.resolve()):
        raise SystemExit(f"refusing to write inside the active results tree: {resolved}")


def _case_kwargs(method_id: str, selection_mode: str | None, budget: int | None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if method_id == "boruta_random_forest":
        kwargs["forest_params"] = dict(TEST_FOREST_PROFILE)
        kwargs["boruta_params"] = dict(TEST_BORUTA_PROFILE)
        kwargs["selection_mode"] = selection_mode or "natural_confirmed"
    else:
        kwargs["catboost_params"] = dict(TEST_CATBOOST_PROFILE)
    if method_id == "catboost_shap":
        kwargs["explanation_sample_size"] = 200
    kwargs["excluded_columns"] = list(EXCLUDED_COLUMNS)
    return kwargs


def run_case(
    *,
    case_name: str,
    method_id: str,
    selection_mode: str | None,
    budget: int | None,
    candidates: pd.DataFrame,
    target: pd.Series,
    scratch: Path,
) -> dict[str, Any]:
    record: dict[str, Any] = {"case": case_name, "method_id": method_id}
    started = time.perf_counter()
    scratch.mkdir(parents=True, exist_ok=True)

    selector_cls, defaults = get_selector(method_id)
    defaults = dict(defaults)
    if budget is not None:
        defaults = apply_feature_budget_to_selector_kwargs(method_id, defaults, budget)
    else:
        defaults.pop("k", None)
    kwargs = _case_kwargs(method_id, selection_mode, budget)
    for key in kwargs:
        defaults.pop(key, None)
    if budget is None:
        defaults.pop("k", None)

    frame = candidates
    imputed = False
    if method_id == "boruta_random_forest":
        frame = impute_for_finite_engines(candidates)
        imputed = True

    selector = selector_cls(**defaults, **kwargs)
    selector.fit(frame, target)
    result = selector.result
    validate_method_selection_mode(method_id, result.selection_mode)

    repeat = selector_cls(**defaults, **kwargs)
    repeat.fit(frame, target)
    deterministic = (
        repeat.result.selected_features == result.selected_features
        and repeat.result.ranking == result.ranking
        and repeat.result.natural_selected == result.natural_selected
        and repeat.result.estimator_config_sha256 == result.estimator_config_sha256
    )

    json_path = scratch / f"{case_name}_selection_result.json"
    csv_path = scratch / f"{case_name}_selection_ranking.csv"
    json_meta = write_json_atomic(json_path, result.to_dict())
    long_frame = result.to_long_frame()
    csv_meta = write_csv_atomic(csv_path, long_frame)

    restored = SelectionResult.from_json(json_path.read_text(encoding="utf-8"))
    reloaded = pd.read_csv(csv_path)

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
        and restored.natural_selected == result.natural_selected
        and restored.budget_status == result.budget_status
        and restored.requested_budget == result.requested_budget
        and restored.candidate_universe_sha256 == result.candidate_universe_sha256
        and restored.training_identity_sha256 == result.training_identity_sha256
        and restored.estimator_config_sha256 == result.estimator_config_sha256
        and restored.heavy_metadata == result.heavy_metadata
        and scores_match
        and list(reloaded["feature"]) == list(long_frame["feature"])
        and list(reloaded["rank"]) == list(long_frame["rank"])
    )

    excluded_clean = not (set(result.selected_features) & set(EXCLUDED_COLUMNS))
    inside_universe = set(result.selected_features).issubset(set(frame.columns))
    descriptor = get_method_descriptor(method_id)

    heavy = result.heavy_metadata or {}
    record.update(
        {
            "status": "PASS"
            if deterministic and serialization_exact and excluded_clean and inside_universe
            else "FAIL",
            "display_label": result.display_label,
            "implementation_id": result.implementation_id,
            "cost_class": descriptor.cost_class,
            "estimator_family": descriptor.estimator_family,
            "selection_mode": result.selection_mode,
            "requested_budget": result.requested_budget,
            "actual_selected_count": result.actual_selected_count,
            "natural_selected_count": result.natural_selected_count,
            "budget_status": result.budget_status,
            "selected_features": list(result.selected_features),
            "ranking_length": len(result.ranking or ()),
            "candidate_universe_count": result.candidate_universe_count,
            "candidate_universe_sha256": result.candidate_universe_sha256,
            "training_identity_sha256": result.training_identity_sha256,
            "estimator_config_sha256": result.estimator_config_sha256,
            "seed": result.seed,
            "tie_rule": result.tie_rule,
            "deterministic_on_refit": deterministic,
            "serialization_exact": serialization_exact,
            "excluded_columns_absent_from_selection": excluded_clean,
            "selection_inside_candidate_universe": inside_universe,
            "harness_median_imputation_applied": imputed,
            "estimator_fit_count": heavy.get("estimator_fit_count"),
            "elimination_history_rows": (
                None if heavy.get("elimination_history") is None
                else len(heavy["elimination_history"])
            ),
            "support_state_counts": (
                None
                if heavy.get("confirmed") is None
                else {
                    "confirmed": heavy["confirmed_count"],
                    "tentative": heavy["tentative_count"],
                    "rejected": heavy["rejected_count"],
                }
            ),
            "explanation_sample": heavy.get("explanation_sample") or None,
            "shap_calc_type": heavy.get("shap_calc_type"),
            "peak_process_rss_bytes": heavy.get("peak_process_rss_bytes"),
            "artifacts": {
                "json": {"path": json_path.name, "sha256": json_meta.sha256},
                "csv": {"path": csv_path.name, "sha256": csv_meta.sha256},
            },
            "warnings": list(result.warnings),
            "fit_seconds": round(float(result.fit_seconds), 6),
            "elapsed_seconds": round(time.perf_counter() - started, 6),
        }
    )
    return record


CASES = (
    # One feasible RFE budget.
    ("rfe_catboost_feasible_budget", "rfe_catboost", None, FEATURE_BUDGET),
    # Boruta's own all-relevant answer.
    ("boruta_natural_confirmed", "boruta_random_forest", "natural_confirmed", None),
    # Boruta asked for more than it can confirm.
    ("boruta_insufficient_confirmed", "boruta_random_forest", "confirmed_top_k", 7),
    # One feasible CatBoost-SHAP budget.
    ("catboost_shap_feasible_budget", "catboost_shap", None, FEATURE_BUDGET),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scratch",
        type=Path,
        default=Path(os.environ.get("TEMP", "/tmp")) / "prompt_08_heavy_fixture",
    )
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT_ROOT)
    arguments = parser.parse_args(argv)

    scratch = arguments.scratch
    scratch.mkdir(parents=True, exist_ok=True)
    _assert_scratch_is_isolated(scratch)

    candidates, target, metadata = build_fixture()
    started = time.perf_counter()

    records = [
        run_case(
            case_name=case_name,
            method_id=method_id,
            selection_mode=selection_mode,
            budget=budget,
            candidates=candidates,
            target=target,
            scratch=scratch / case_name,
        )
        for case_name, method_id, selection_mode, budget in CASES
    ]
    failures = [record for record in records if record["status"] != "PASS"]

    payload = {
        "schema_version": "prompt_08_synthetic_fixture_results_v1",
        "status": "PASS" if not failures else "FAIL",
        "generated_at_utc": pd.Timestamp.now("UTC").isoformat(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "estimator_profiles": {
            "classification": "synthetic_test_profiles_not_frozen_research_settings",
            "catboost": TEST_CATBOOST_PROFILE,
            "random_forest": TEST_FOREST_PROFILE,
            "boruta": TEST_BORUTA_PROFILE,
            "frozen_by": "Prompt 9 after real single-fold runtime measurement",
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
            "contains_linear_signal": True,
            "contains_nonlinear_signal": True,
            "contains_redundant_copy": True,
            "contains_noise": True,
            "contains_constant": True,
            "contains_missing_values": True,
            "real_dataset_loaded": False,
            "oot_data_loaded": False,
            "real_fold_executed": False,
        },
        "isolation": {
            "scratch_directory": str(scratch),
            "inside_active_results_root": False,
            "inside_legacy_results_root": False,
            "catboost_info_written": (Path.cwd() / "catboost_info").exists(),
        },
        "cases": records,
        "failure_count": len(failures),
        "total_elapsed_seconds": round(time.perf_counter() - started, 6),
        "peak_process_rss_mb": _peak_rss_mb(),
    }

    arguments.audit.mkdir(parents=True, exist_ok=True)
    write_json_atomic(arguments.audit / "synthetic_fixture_results.json", payload)

    for record in records:
        print(
            f"{record['status']:4s} {record['case']:32s} "
            f"mode={record['selection_mode']:24s} "
            f"selected={record['actual_selected_count']}/{record['requested_budget']} "
            f"status={record['budget_status']}"
        )
    print()
    print(
        f"synthetic heavy fixture: {payload['status']} "
        f"({len(records)} cases, {len(failures)} failures, "
        f"{payload['total_elapsed_seconds']:.1f}s)"
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
