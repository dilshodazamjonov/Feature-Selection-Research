"""Artifact-only audit of the frozen Prompt 10 full-baseline matrix.

This module deliberately has no dataset loader dependency.  Every calculation is
derived from completed manifests, saved predictions, saved fold metrics, saved
selected-feature tables, and saved resource traces.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from credit_risk_fs.evaluation.paired_inference import (
    holm_adjust,
    ks_statistic,
    lift_at_fraction,
    paired_delong_test,
)
from credit_risk_fs.evaluation.stability import (
    kuncheva_stability,
    mean_pairwise_jaccard,
    nogueira_stability,
)
from credit_risk_fs.experiments.atomic_io import sha256_file
from credit_risk_fs.experiments.full_baseline import (
    FullBaselineCell,
    FullBaselinePlan,
    inspect_cell,
    load_full_baseline_plan,
)


AUDIT_SCHEMA_VERSION = "prompt_11_baseline_results_audit_v1"
METRIC_ABSOLUTE_TOLERANCE = 1e-12
LIFT_ABSOLUTE_TOLERANCE = 1e-12
BOOTSTRAP_REPETITIONS = 2_000
BOOTSTRAP_MINIMUM_VALID = 1_900
BOOTSTRAP_SEED = 20_260_721
COMPARISON_METHODS = (
    "iv_woe",
    "mrmr_mutual_information",
    "lasso_l1_logistic",
    "legacy_rf_relevance_corr",
    "catboost_shap",
    "boruta_random_forest",
    "rfe_catboost",
)
COMPARISON_REFERENCES = ("full_features", "random_k")
NATURAL_SUPPORT_METHODS = frozenset({"lasso_l1_logistic", "boruta_random_forest"})


class BaselineAuditError(ValueError):
    """Raised when persisted baseline evidence violates an audit invariant."""


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _ordered_string_sha256(values: Iterable[Any]) -> str:
    """Match the completion manifest's newline-delimited ordered-ID digest."""

    digest = hashlib.sha256()
    for value in values:
        encoded = json.dumps(None if pd.isna(value) else str(value), ensure_ascii=False)
        digest.update(encoded.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _identity_target_sha256(frame: pd.DataFrame) -> str:
    return _canonical_json_sha256(
        [
            [str(row_id), int(target)]
            for row_id, target in zip(
                frame["stable_row_id"], frame["target"], strict=True
            )
        ]
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - exercised by corruption tests
        raise BaselineAuditError(f"unreadable JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise BaselineAuditError(f"JSON artifact is not an object: {path}")
    return value


def validate_prediction_frame(
    frame: pd.DataFrame,
    *,
    expected_dataset: str | None = None,
    expected_model: str | None = None,
    expected_method: str | None = None,
    expected_split: str | None = None,
) -> pd.DataFrame:
    """Validate saved prediction identity, target, and probability integrity."""

    required = {
        "stable_row_id",
        "target",
        "prediction_probability",
        "dataset",
        "model",
        "method",
        "split",
        "run_id",
    }
    missing = required - set(frame.columns)
    if missing:
        raise BaselineAuditError(f"prediction columns missing: {sorted(missing)}")
    output = frame.copy()
    if output.empty:
        raise BaselineAuditError("prediction artifact is empty")
    if output["stable_row_id"].isna().any():
        raise BaselineAuditError("prediction identities contain missing values")
    normalized_ids = output["stable_row_id"].astype(str)
    if normalized_ids.duplicated().any():
        raise BaselineAuditError("prediction identities are duplicated")
    target = pd.to_numeric(output["target"], errors="raise")
    if target.isna().any() or not target.isin([0, 1]).all():
        raise BaselineAuditError("prediction targets must be finite binary 0/1")
    probability = pd.to_numeric(output["prediction_probability"], errors="raise")
    values = probability.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise BaselineAuditError("prediction probabilities must be finite")
    if np.any((values < 0.0) | (values > 1.0)):
        raise BaselineAuditError("prediction probabilities must lie in [0, 1]")
    checks = {
        "dataset": expected_dataset,
        "model": expected_model,
        "method": expected_method,
        "split": expected_split,
    }
    for column, expected in checks.items():
        observed = set(output[column].dropna().astype(str))
        if expected is not None and observed != {str(expected)}:
            raise BaselineAuditError(
                f"prediction {column} mismatch: expected={expected!r}, observed={sorted(observed)}"
            )
    output["stable_row_id"] = normalized_ids
    output["target"] = target.astype("int8")
    output["prediction_probability"] = values
    return output


def recompute_prediction_metrics(frame: pd.DataFrame) -> dict[str, float]:
    """Recompute metrics supported by one authenticated prediction artifact."""

    target = frame["target"].to_numpy(dtype=int)
    score = frame["prediction_probability"].to_numpy(dtype=float)
    ids = frame["stable_row_id"].astype(str).to_numpy()
    if len(np.unique(target)) != 2:
        raise BaselineAuditError("headline metrics require both target classes")
    auc = float(roc_auc_score(target, score))
    ks, _ = ks_statistic(target, score)
    return {
        "auc": auc,
        "gini": float(2.0 * auc - 1.0),
        "ks": float(ks),
        "lift_at_10": float(lift_at_fraction(target, score, ids, fraction=0.10)),
        "log_loss": float(log_loss(target, score, labels=[0, 1])),
        "brier": float(brier_score_loss(target, score)),
    }


def aggregate_dev_folds(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize five ordered fold rows without pooling their predictions."""

    required = {"run_id", "dataset", "model", "method_id", "fold", "auc", "gini", "ks", "lift_at_10"}
    missing = required - set(frame.columns)
    if missing:
        raise BaselineAuditError(f"DEV fold table missing: {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    group_columns = ["run_id", "dataset", "model", "method_id"]
    for keys, group in frame.groupby(group_columns, sort=False):
        folds = pd.to_numeric(group["fold"], errors="coerce")
        if folds.isna().any() or sorted(folds.astype(int).tolist()) != [1, 2, 3, 4, 5]:
            raise BaselineAuditError(f"invalid expanding-window fold identity for {keys[0]}")
        row = dict(zip(group_columns, keys, strict=True))
        row["fold_order"] = "1|2|3|4|5"
        row["valid_fold_count"] = 5
        row["pooling_performed"] = False
        for metric in ("auc", "gini", "ks", "lift_at_10", "selected_feature_count"):
            column = "selected_features" if metric == "selected_feature_count" else metric
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
            row[f"{metric}_min"] = float(values.min()) if len(values) else np.nan
            row[f"{metric}_max"] = float(values.max()) if len(values) else np.nan
            row[f"{metric}_range"] = float(values.max() - values.min()) if len(values) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_oot_availability(frame: pd.DataFrame | None) -> dict[str, Any]:
    """Report OOT availability without inferring evidence from DEV."""

    count = 0 if frame is None else int(len(frame))
    return {
        "oot_available": count > 0,
        "oot_evaluation_units": count,
        "interpretation": (
            "authenticated_saved_oot_evidence_available"
            if count > 0
            else "oot_conclusions_unavailable_do_not_infer_from_dev"
        ),
    }


def aggregate_random_k_replicates(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize every frozen random-k seed; never select a favorable seed."""

    required = {"dataset", "model", "method_id", "seed", "auc"}
    missing = required - set(frame.columns)
    if missing:
        raise BaselineAuditError(f"random-k table missing: {sorted(missing)}")
    random_rows = frame.loc[frame["method_id"].eq("random_k")].copy()
    if random_rows.empty:
        return pd.DataFrame(
            columns=[
                "dataset",
                "model",
                "replicate_count",
                "seeds",
                "auc_mean",
                "auc_std",
                "favorable_seed_selected",
            ]
        )
    rows = []
    for (dataset, model), group in random_rows.groupby(["dataset", "model"], sort=False):
        if group["seed"].duplicated().any():
            raise BaselineAuditError(f"duplicate random-k seed for {dataset}/{model}")
        values = pd.to_numeric(group["auc"], errors="raise")
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "replicate_count": len(group),
                "seeds": "|".join(map(str, sorted(group["seed"].astype(int)))),
                "auc_mean": float(values.mean()),
                "auc_std": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                "favorable_seed_selected": False,
            }
        )
    return pd.DataFrame(rows)


def _validate_manifested_file(cell_dir: Path, entry: Mapping[str, Any]) -> None:
    relative = str(entry.get("path", ""))
    path = cell_dir / relative
    if not path.is_file():
        raise BaselineAuditError(f"manifested artifact is missing: {path}")
    if path.stat().st_size != int(entry.get("size_bytes", -1)):
        raise BaselineAuditError(f"manifested artifact size mismatch: {path}")
    if sha256_file(path) != entry.get("sha256"):
        raise BaselineAuditError(f"manifested artifact hash mismatch: {path}")


def authenticate_cell_artifacts(plan: FullBaselinePlan, cell: FullBaselineCell) -> dict[str, Any]:
    """Authenticate one completed cell and its finalized manifest bindings."""

    inspection = inspect_cell(plan, cell)
    if not inspection.valid_completed:
        raise BaselineAuditError(f"cell is not authenticated complete: {cell.cell_id}: {inspection.reason}")
    cell_dir = plan.results_root / "runs" / cell.dataset / cell.cell_id
    checkpoint = _read_json(cell_dir / "checkpoint.json")
    manifest = _read_json(cell_dir / "manifest.json")
    if checkpoint.get("status") != "completed" or manifest.get("status") != "completed":
        raise BaselineAuditError(f"terminal state mismatch for {cell.cell_id}")
    if checkpoint.get("run_id") != cell.cell_id or manifest.get("run_id") != cell.cell_id:
        raise BaselineAuditError(f"run identity mismatch for {cell.cell_id}")
    finalized = checkpoint.get("finalized_artifacts")
    if not isinstance(finalized, dict) or not finalized:
        raise BaselineAuditError(f"finalized artifact manifest missing for {cell.cell_id}")
    for entry in finalized.values():
        if not isinstance(entry, Mapping):
            raise BaselineAuditError(f"invalid finalized artifact entry for {cell.cell_id}")
        _validate_manifested_file(cell_dir, entry)
    return {
        "cell_dir": cell_dir,
        "checkpoint": checkpoint,
        "manifest": manifest,
        "finalized": finalized,
    }


def _read_prediction(
    cell: FullBaselineCell,
    cell_dir: Path,
    finalized: Mapping[str, Any],
    split: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    relative = f"results/{split}_predictions.csv"
    entry = finalized.get(relative)
    if not isinstance(entry, Mapping):
        raise BaselineAuditError(f"prediction is not finalized: {cell.cell_id} {split}")
    path = cell_dir / relative
    frame = pd.read_csv(path)
    frame = validate_prediction_frame(
        frame,
        expected_dataset=cell.dataset,
        expected_model=cell.model,
        expected_method=cell.method_id,
        expected_split=split,
    )
    if len(frame) != int(entry.get("row_count", -1)):
        raise BaselineAuditError(f"prediction row count mismatch: {cell.cell_id} {split}")
    ordered_hash = _ordered_string_sha256(frame["stable_row_id"])
    if ordered_hash != entry.get("ordered_row_identity_sha256"):
        raise BaselineAuditError(f"prediction ordered-row hash mismatch: {cell.cell_id} {split}")
    return frame, {
        "path": relative,
        "sha256": str(entry["sha256"]),
        "row_count": len(frame),
        "ordered_row_identity_sha256": ordered_hash,
        "identity_target_sha256": _identity_target_sha256(frame),
    }


def _feature_sets(frame: pd.DataFrame) -> list[set[str]]:
    if not {"fold_id", "feature_name"}.issubset(frame.columns):
        raise BaselineAuditError("fold selection artifact has an invalid schema")
    fold_id = pd.to_numeric(frame["fold_id"], errors="coerce")
    output: list[set[str]] = []
    for fold in range(1, 6):
        values = frame.loc[fold_id.eq(fold), "feature_name"]
        if values.isna().any() or values.astype(str).duplicated().any():
            raise BaselineAuditError(f"invalid selected features in fold {fold}")
        output.append(set(values.astype(str)))
    return output


def recompute_selection_stability(
    selections: pd.DataFrame,
    *,
    candidate_count: int,
) -> dict[str, Any]:
    sets = _feature_sets(selections)
    counts = [len(value) for value in sets]
    fixed_size = len(set(counts)) == 1
    union = set().union(*sets)
    intersection = set.intersection(*sets) if sets else set()
    frequencies = {
        feature: sum(feature in selected for selected in sets) / len(sets)
        for feature in sorted(union)
    }
    pairwise = [
        len(left & right) / len(left | right) if left | right else 1.0
        for left, right in itertools.combinations(sets, 2)
    ]
    return {
        "fold_selected_counts": "|".join(map(str, counts)),
        "minimum_selected_count": min(counts),
        "maximum_selected_count": max(counts),
        "natural_support_varies": not fixed_size,
        "union_size": len(union),
        "intersection_size": len(intersection),
        "core_frequency_80_count": sum(value >= 0.8 for value in frequencies.values()),
        "mean_pairwise_jaccard": float(mean_pairwise_jaccard(sets)),
        "minimum_pairwise_jaccard": float(min(pairwise)) if pairwise else np.nan,
        "maximum_pairwise_jaccard": float(max(pairwise)) if pairwise else np.nan,
        "kuncheva_applicability": "applicable" if fixed_size else "not_applicable_varying_subset_size",
        "kuncheva_stability": float(kuncheva_stability(sets, candidate_count)) if fixed_size else np.nan,
        "nogueira_applicability": "applicable_fixed_universe_selection_matrix",
        "nogueira_stability": float(nogueira_stability(sets, candidate_count)),
    }


def build_structural_feasibility(
    selection_stability: pd.DataFrame,
    runtime_resources: pd.DataFrame,
) -> pd.DataFrame:
    """Build outcome-blind fold-level feasibility from saved support/resource metadata."""

    rows: list[dict[str, Any]] = []
    boruta = selection_stability.loc[
        selection_stability["method_id"].eq("boruta_random_forest")
    ]
    runtime_lookup = {
        (row.dataset, row.model, row.method_id): row
        for row in runtime_resources.itertuples(index=False)
    }
    for item in boruta.itertuples(index=False):
        counts = [int(value) for value in str(item.fold_selected_counts).split("|")]
        for fold, observed in enumerate(counts, start=1):
            for combination in (
                "boruta_then_rfe_catboost",
                "boruta_then_mrmr_mutual_information",
            ):
                budget = 20 if item.model == "lr" else 40
                if observed < budget:
                    state = "infeasible_natural_support"
                elif observed == budget:
                    # A fixed-budget Boruta artifact proves only at least k when it
                    # reaches k.  Exact natural support is therefore unavailable.
                    state = "requires_new_natural_support_fit"
                else:  # pragma: no cover - fixed-budget artifacts cannot exceed k
                    state = "refinement_feasible"
                resource = runtime_lookup[(item.dataset, item.model, "boruta_random_forest")]
                rows.append(
                    {
                        "combination_id": combination,
                        "dataset": item.dataset,
                        "evaluation_model": item.model,
                        "fold": fold,
                        "final_budget": budget,
                        "saved_boruta_selected_count": observed,
                        "feasibility_state": state,
                        "authentic_natural_support_count_available": observed < budget,
                        "reusable_component_artifact": False,
                        "reuse_reason": "baseline Boruta was budget-specific and lacks complete natural-support provenance",
                        "required_new_stage_fits": "boruta_natural_support+stage_2_if_feasible",
                        "forecast_peak_rss_gib_lower_bound": resource.peak_process_rss_gib,
                        "forecast_active_seconds_lower_bound": resource.active_computation_seconds,
                        "predictive_outcome_used": False,
                    }
                )
    for dataset in ("homecredit", "lendingclub_v2"):
        for fold in range(1, 6):
            for pool in (100, 200, 300):
                rows.append(
                    {
                        "combination_id": "iv_then_boruta",
                        "dataset": dataset,
                        "evaluation_model": "model_independent_selection",
                        "fold": fold,
                        "final_budget": "natural",
                        "saved_boruta_selected_count": np.nan,
                        "feasibility_state": "requires_new_iv_and_boruta_fit",
                        "authentic_natural_support_count_available": False,
                        "reusable_component_artifact": False,
                        "reuse_reason": f"no saved IV-top-{pool} then Boruta artifact",
                        "required_new_stage_fits": f"iv_top_{pool}+boruta_confirmed_only",
                        "forecast_peak_rss_gib_lower_bound": np.nan,
                        "forecast_active_seconds_lower_bound": np.nan,
                        "predictive_outcome_used": False,
                    }
                )
            rows.append(
                {
                    "combination_id": "statistical_normalized_average_rank",
                    "dataset": dataset,
                    "evaluation_model": "model_specific_top_k",
                    "fold": fold,
                    "final_budget": "20|40",
                    "saved_boruta_selected_count": np.nan,
                    "feasibility_state": "requires_new_complete_rank_component_fits",
                    "authentic_natural_support_count_available": False,
                    "reusable_component_artifact": False,
                    "reuse_reason": "Prompt 10 retained selected subsets but not an authenticated common-universe five-voter rank bundle",
                    "required_new_stage_fits": "iv+lasso+rfe+boruta+catboost_shap_once_per_training_identity",
                    "forecast_peak_rss_gib_lower_bound": np.nan,
                    "forecast_active_seconds_lower_bound": np.nan,
                    "predictive_outcome_used": False,
                }
            )
    return pd.DataFrame(rows)


def _auc_from_weighted_sample(
    positive_scores: np.ndarray,
    negative_scores: np.ndarray,
    positive_counts: np.ndarray,
    negative_counts: np.ndarray,
) -> np.ndarray:
    """Exact midrank AUC for multiple bootstrap count vectors."""

    order = np.argsort(negative_scores, kind="mergesort")
    sorted_negative = negative_scores[order]
    cumulative = np.cumsum(negative_counts[:, order], axis=1, dtype=np.int64)
    padded = np.concatenate(
        [np.zeros((len(positive_counts), 1), dtype=np.int64), cumulative], axis=1
    )
    left = np.searchsorted(sorted_negative, positive_scores, side="left")
    right = np.searchsorted(sorted_negative, positive_scores, side="right")
    lower = padded[:, left]
    through = padded[:, right]
    wins = lower + 0.5 * (through - lower)
    numerator = np.sum(wins * positive_counts, axis=1, dtype=np.float64)
    denominator = positive_scores.size * negative_scores.size
    return numerator / float(denominator)


def paired_stratified_auc_bootstrap_many(
    target: Sequence[int],
    scores: Mapping[str, Sequence[float]],
    *,
    repetitions: int = BOOTSTRAP_REPETITIONS,
    seed: int = BOOTSTRAP_SEED,
    chunk_size: int = 20,
) -> dict[str, np.ndarray]:
    """Exact paired target-stratified AUC bootstrap shared across methods.

    Sampling counts are generated once and applied to every method, preserving
    pairing while avoiding duplicate resampling work for each comparison.
    """

    y = np.asarray(target, dtype=np.int8)
    if y.ndim != 1 or not set(np.unique(y)).issubset({0, 1}) or len(np.unique(y)) != 2:
        raise BaselineAuditError("stratified bootstrap requires a binary two-class target")
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    arrays = {name: np.asarray(value, dtype=float) for name, value in scores.items()}
    if not arrays or any(value.shape != y.shape for value in arrays.values()):
        raise BaselineAuditError("bootstrap score arrays must match the target")
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise BaselineAuditError("bootstrap scores must be finite")
    output = {name: np.empty(repetitions, dtype=float) for name in arrays}
    generator = np.random.default_rng(seed)
    for start in range(0, repetitions, chunk_size):
        width = min(chunk_size, repetitions - start)
        positive_counts = np.empty((width, len(pos)), dtype=np.int32)
        negative_counts = np.empty((width, len(neg)), dtype=np.int32)
        for offset in range(width):
            sampled_positive = generator.choice(len(pos), size=len(pos), replace=True)
            sampled_negative = generator.choice(len(neg), size=len(neg), replace=True)
            positive_counts[offset] = np.bincount(
                sampled_positive, minlength=len(pos)
            ).astype(np.int32, copy=False)
            negative_counts[offset] = np.bincount(
                sampled_negative, minlength=len(neg)
            ).astype(np.int32, copy=False)
        for name, values in arrays.items():
            output[name][start : start + width] = _auc_from_weighted_sample(
                values[pos], values[neg], positive_counts, negative_counts
            )
    return output


def _aligned_group_predictions(
    paths: Mapping[str, Path],
    *,
    dataset: str,
    model: str,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    base: pd.DataFrame | None = None
    scores: dict[str, np.ndarray] = {}
    for method, path in paths.items():
        frame = pd.read_csv(
            path,
            usecols=[
                "stable_row_id",
                "target",
                "prediction_probability",
                "dataset",
                "model",
                "method",
                "split",
                "run_id",
            ],
        )
        frame = validate_prediction_frame(
            frame,
            expected_dataset=dataset,
            expected_model=model,
            expected_method=method,
            expected_split="oot",
        )
        identity = frame[["stable_row_id", "target"]].reset_index(drop=True)
        if base is None:
            base = identity
        elif not identity.equals(base):
            raise BaselineAuditError(
                f"OOT rows are not identically ordered for {dataset}/{model}/{method}"
            )
        scores[method] = frame["prediction_probability"].to_numpy(dtype=float)
    if base is None:
        raise BaselineAuditError(f"no OOT predictions for {dataset}/{model}")
    return base, scores


def run_predeclared_comparisons(
    prediction_paths: Mapping[tuple[str, str, str], Path],
    dev_fold_rows: pd.DataFrame,
    selected_counts: Mapping[tuple[str, str, str], int],
    *,
    repetitions: int = BOOTSTRAP_REPETITIONS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run only the preregistered baseline comparison families."""

    comparison_rows: list[dict[str, Any]] = []
    for dataset in ("homecredit", "lendingclub_v2"):
        for model in ("lr", "catboost"):
            methods = (*COMPARISON_REFERENCES, *COMPARISON_METHODS)
            paths = {method: prediction_paths[(dataset, model, method)] for method in methods}
            identity, scores = _aligned_group_predictions(paths, dataset=dataset, model=model)
            bootstrap = paired_stratified_auc_bootstrap_many(
                identity["target"].to_numpy(dtype=int),
                scores,
                repetitions=repetitions,
                seed=seed,
            )
            target = identity["target"].to_numpy(dtype=int)
            row_hash = _identity_target_sha256(identity)
            for reference in COMPARISON_REFERENCES:
                family = f"baseline_vs_{reference}__{dataset}__{model}"
                raw_family: list[dict[str, Any]] = []
                for method in COMPARISON_METHODS:
                    result = paired_delong_test(target, scores[method], scores[reference])
                    differences = bootstrap[method] - bootstrap[reference]
                    lower, upper = np.percentile(differences, [2.5, 97.5])
                    fold_method = dev_fold_rows.loc[
                        dev_fold_rows["dataset"].eq(dataset)
                        & dev_fold_rows["model"].eq(model)
                        & dev_fold_rows["method_id"].eq(method)
                    ].sort_values("fold")
                    fold_reference = dev_fold_rows.loc[
                        dev_fold_rows["dataset"].eq(dataset)
                        & dev_fold_rows["model"].eq(model)
                        & dev_fold_rows["method_id"].eq(reference)
                    ].sort_values("fold")
                    if list(fold_method["fold"]) != list(fold_reference["fold"]):
                        raise BaselineAuditError(f"DEV fold alignment failed for {family}/{method}")
                    fold_deltas = (
                        fold_method["auc"].to_numpy(dtype=float)
                        - fold_reference["auc"].to_numpy(dtype=float)
                    )
                    direction = np.sign(float(result["auc_difference_a_minus_b"]))
                    same_direction = int(np.sum(np.sign(fold_deltas) == direction)) if direction else int(np.sum(fold_deltas == 0))
                    count_a = selected_counts[(dataset, model, method)]
                    count_b = selected_counts[(dataset, model, reference)]
                    raw_family.append(
                        {
                            "family_id": family,
                            "dataset": dataset,
                            "model": model,
                            "split": "oot",
                            "method_a": method,
                            "method_b": reference,
                            "direction_convention": "method_a_minus_method_b",
                            "ordered_identity_target_sha256": row_hash,
                            "aligned_row_count": len(identity),
                            "positive_count": int(target.sum()),
                            "negative_count": int(len(target) - target.sum()),
                            "auc_a": float(result["auc_a"]),
                            "auc_b": float(result["auc_b"]),
                            "delta_auc": float(result["auc_difference_a_minus_b"]),
                            "delta_gini": float(2.0 * result["auc_difference_a_minus_b"]),
                            "delong_variance": float(result["variance"]),
                            "delong_z": float(result["z_score"]),
                            "delong_raw_p": float(result["two_sided_p_value"]),
                            "delong_valid": True,
                            "bootstrap_auc_point": float(result["auc_difference_a_minus_b"]),
                            "bootstrap_ci95_lower": float(lower),
                            "bootstrap_ci95_upper": float(upper),
                            "bootstrap_seed": seed,
                            "bootstrap_attempted": repetitions,
                            "bootstrap_valid": repetitions,
                            "bootstrap_interval_valid": repetitions >= BOOTSTRAP_MINIMUM_VALID,
                            "dev_fold_auc_deltas": "|".join(f"{value:.12g}" for value in fold_deltas),
                            "dev_folds_same_oot_direction": same_direction,
                            "feature_count_a": count_a,
                            "feature_count_b": count_b,
                            "feature_count_comparability": "matched" if count_a == count_b else "different_counts",
                            "natural_support_caveat": method in NATURAL_SUPPORT_METHODS,
                        }
                    )
                adjusted = holm_adjust(row["delong_raw_p"] for row in raw_family)
                for row, value in zip(raw_family, adjusted, strict=True):
                    row["holm_family_size"] = len(raw_family)
                    row["holm_adjusted_p"] = float(value)
                    row["holm_reject_0_05"] = bool(value <= 0.05)
                    ci_excludes = bool(
                        row["bootstrap_ci95_lower"] > 0 or row["bootstrap_ci95_upper"] < 0
                    )
                    row["bootstrap_ci_excludes_zero"] = ci_excludes
                    same_direction = int(row["dev_folds_same_oot_direction"])
                    if row["holm_reject_0_05"] and ci_excludes and same_direction >= 4:
                        strength = "strong"
                    elif (row["holm_reject_0_05"] or ci_excludes) and same_direction >= 3:
                        strength = "moderate"
                    elif abs(float(row["delta_auc"])) > 0:
                        strength = "weak"
                    else:
                        strength = "not_supported"
                    if not row["holm_reject_0_05"] and not ci_excludes:
                        strength = "not_supported"
                    row["evidence_strength"] = strength
                    row["interpretation_limit"] = (
                        "natural-support/count mismatch; non-significance is not equivalence"
                        if row["natural_support_caveat"] or row["feature_count_comparability"] != "matched"
                        else "scoped to this dataset/model/OOT split; non-significance is not equivalence"
                    )
                    comparison_rows.append(row)
    comparisons = pd.DataFrame(comparison_rows)
    families = (
        comparisons.groupby("family_id", as_index=False)
        .agg(
            dataset=("dataset", "first"),
            model=("model", "first"),
            reference=("method_b", "first"),
            family_size=("holm_family_size", "first"),
            rejected_count=("holm_reject_0_05", "sum"),
            minimum_raw_p=("delong_raw_p", "min"),
            minimum_adjusted_p=("holm_adjusted_p", "min"),
        )
    )
    families["holm_scope"] = "within_named_dataset_model_reference_family"
    return comparisons, families


def _runtime_row(cell: FullBaselineCell, cell_dir: Path) -> dict[str, Any]:
    runtime = pd.read_csv(cell_dir / "results/runtime_summary.csv").iloc[0].to_dict()
    resource = _read_json(cell_dir / "resource_usage.json")
    checkpoint = _read_json(cell_dir / "checkpoint.json")
    attempts = cell_dir / "incomplete" / "attempt_history"
    attempt_count = len([path for path in attempts.glob("attempt_*") if path.is_dir()]) if attempts.is_dir() else 0
    return {
        "run_id": cell.cell_id,
        "dataset": cell.dataset,
        "model": cell.model,
        "method_id": cell.method_id,
        "selection_seconds": float(runtime.get("feature_selection_time_sec", np.nan)),
        "final_model_fit_seconds": float(runtime.get("training_time_sec", np.nan)),
        "evaluation_seconds": float(runtime.get("evaluation_time_sec", np.nan)),
        "reported_total_runtime_seconds": float(runtime.get("total_runtime_seconds", np.nan)),
        "active_computation_seconds": float(resource.get("active_computation_seconds", np.nan)),
        "ram_wait_seconds": float(resource.get("total_ram_wait_seconds", 0.0)),
        "peak_process_rss_gib": float(resource.get("peak_process_tree_rss_bytes", 0)) / 2**30,
        "minimum_available_ram_gib": float(resource.get("minimum_system_available_ram_bytes", 0)) / 2**30,
        "resource_sample_count": len(resource.get("samples", [])),
        "ram_wait_count": int(resource.get("ram_control", {}).get("wait_count", 0)),
        "historical_attempt_count": attempt_count,
        "completed_attempts": 1,
        "terminal_state": checkpoint.get("status"),
        "warnings": " | ".join(map(str, resource.get("warnings", []))),
    }


def audit_completed_baselines(
    repository_root: str | Path,
    *,
    bootstrap_repetitions: int = BOOTSTRAP_REPETITIONS,
) -> dict[str, pd.DataFrame | dict[str, Any]]:
    """Authenticate and audit all 36 baselines without resolving a dataset path."""

    root = Path(repository_root).resolve()
    plan = load_full_baseline_plan(root)
    long_rows: list[dict[str, Any]] = []
    reconciliation_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    oot_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []
    score_psi_rows: list[dict[str, Any]] = []
    feature_psi_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    prediction_paths: dict[tuple[str, str, str], Path] = {}
    selected_counts: dict[tuple[str, str, str], int] = {}
    candidate_counts = {"homecredit": 529, "lendingclub_v2": 675}
    identity_hashes: dict[tuple[str, str], set[str]] = defaultdict(set)

    for cell in plan.cells:
        authenticated = authenticate_cell_artifacts(plan, cell)
        cell_dir = authenticated["cell_dir"]
        finalized = authenticated["finalized"]
        config = _read_json(cell_dir / "config.json")
        checkpoint = authenticated["checkpoint"]
        selections = pd.read_csv(cell_dir / "fold_selections.csv")
        final_selection = pd.read_csv(cell_dir / "selected_features.csv")
        selected_count = len(final_selection)
        selected_counts[(cell.dataset, cell.model, cell.method_id)] = selected_count
        stability = recompute_selection_stability(
            selections, candidate_count=candidate_counts[cell.dataset]
        )
        stability_rows.append(
            {
                "run_id": cell.cell_id,
                "dataset": cell.dataset,
                "model": cell.model,
                "method_id": cell.method_id,
                "selection_mode": (
                    "natural_support_or_capped" if cell.method_id in NATURAL_SUPPORT_METHODS else "exact_or_full_control"
                ),
                **stability,
            }
        )
        runtime_rows.append(_runtime_row(cell, cell_dir))
        psi = pd.read_csv(cell_dir / "results/model_score_psi.csv").iloc[0]
        score_psi = float(psi["model_score_psi"])
        score_drift_level = (
            "stable"
            if score_psi < 0.10
            else ("moderate" if score_psi < 0.25 else "unstable")
        )
        score_psi_rows.append(
            {
                "run_id": cell.cell_id,
                "dataset": cell.dataset,
                "model": cell.model,
                "method_id": cell.method_id,
                "score_psi": score_psi,
                "drift_level": score_drift_level,
                "descriptive_thresholds": "stable<0.10; moderate[0.10,0.25); unstable>=0.25",
                "reference_split": "saved_dev_predictions",
                "comparison_split": "saved_oot_predictions",
                "binning": "DEV_OOF_quantile_as_recorded_by_baseline",
                "recomputed": False,
                "audit_status": "saved_authenticated_convention_audited",
            }
        )
        feature_psi = pd.read_csv(cell_dir / "results/selected_feature_psi.csv")
        for item in feature_psi.itertuples(index=False):
            feature_psi_rows.append(
                {
                    "run_id": cell.cell_id,
                    "dataset": cell.dataset,
                    "model": cell.model,
                    "method_id": cell.method_id,
                    "feature_name": str(item.feature_name),
                    "psi": float(item.psi),
                    "drift_level": str(item.drift_level),
                    "source": "authenticated_saved_feature_psi",
                    "raw_feature_matrix_opened": False,
                }
            )

        metrics = pd.read_csv(cell_dir / "results/prediction_metrics.csv")
        oot_test = pd.read_csv(cell_dir / "results/oot_test_results.csv").iloc[0]
        prediction_meta: dict[str, dict[str, Any]] = {}
        for split in ("dev", "oot"):
            prediction, metadata = _read_prediction(cell, cell_dir, finalized, split)
            prediction_meta[split] = metadata
            identity_hashes[(cell.dataset, split)].add(metadata["identity_target_sha256"])
            computed = recompute_prediction_metrics(prediction)
            stored = metrics.loc[metrics["split"].eq(split)].iloc[0]
            for metric in ("auc", "gini", "ks", "log_loss", "brier"):
                delta = float(computed[metric] - float(stored[metric]))
                reconciliation_rows.append(
                    {
                        "run_id": cell.cell_id,
                        "dataset": cell.dataset,
                        "model": cell.model,
                        "method_id": cell.method_id,
                        "split": split,
                        "metric": metric,
                        "stored_value": float(stored[metric]),
                        "recomputed_value": float(computed[metric]),
                        "absolute_difference": abs(delta),
                        "tolerance": METRIC_ABSOLUTE_TOLERANCE,
                        "status": "pass" if abs(delta) <= METRIC_ABSOLUTE_TOLERANCE else "fail",
                    }
                )
            if split == "oot":
                delta = float(computed["lift_at_10"] - float(oot_test["lift_at_10"]))
                reconciliation_rows.append(
                    {
                        "run_id": cell.cell_id,
                        "dataset": cell.dataset,
                        "model": cell.model,
                        "method_id": cell.method_id,
                        "split": split,
                        "metric": "lift_at_10",
                        "stored_value": float(oot_test["lift_at_10"]),
                        "recomputed_value": float(computed["lift_at_10"]),
                        "absolute_difference": abs(delta),
                        "tolerance": LIFT_ABSOLUTE_TOLERANCE,
                        "status": "pass" if abs(delta) <= LIFT_ABSOLUTE_TOLERANCE else "fail",
                    }
                )
            long_rows.append(
                {
                    "run_id": cell.cell_id,
                    "cell_id": cell.cell_id,
                    "cell_index": cell.cell_index,
                    "phase": "full_dev_refit_diagnostic" if split == "dev" else "oot",
                    "dataset": cell.dataset,
                    "fold_or_split": "full_dev_in_sample" if split == "dev" else "locked_oot",
                    "method_id": cell.method_id,
                    "implementation_id": cell.implementation_id,
                    "method_role": "control" if cell.method_id in COMPARISON_REFERENCES else "selector",
                    "evaluation_model": cell.model,
                    "candidate_universe_count": candidate_counts[cell.dataset],
                    "candidate_universe_hash": "unsupported_not_persisted_by_prompt_10",
                    "selected_feature_count": selected_count,
                    "budget_semantics": "natural_support_or_cap" if cell.method_id in NATURAL_SUPPORT_METHODS else ("full_universe" if cell.method_id == "full_features" else "exact_budget"),
                    "requested_budget": cell.feature_budget,
                    "natural_support": cell.method_id in NATURAL_SUPPORT_METHODS,
                    "feasibility_state": "infeasible_natural_support" if cell.feature_budget is not None and selected_count < cell.feature_budget else "completed",
                    "seed": cell.seed,
                    "prediction_sha256": metadata["sha256"],
                    "prediction_row_identity_sha256": metadata["ordered_row_identity_sha256"],
                    "identity_target_sha256": metadata["identity_target_sha256"],
                    "metric_artifact_sha256": finalized["results/prediction_metrics.csv"]["sha256"],
                    "configuration_lock_sha256": plan.configuration_sha256,
                    "auc": computed["auc"],
                    "gini": computed["gini"],
                    "ks": computed["ks"],
                    "lift_at_10": computed["lift_at_10"],
                    "runtime_seconds": runtime_rows[-1]["active_computation_seconds"],
                    "peak_process_rss_gib": runtime_rows[-1]["peak_process_rss_gib"],
                    "minimum_available_ram_gib": runtime_rows[-1]["minimum_available_ram_gib"],
                    "ram_wait_seconds": runtime_rows[-1]["ram_wait_seconds"],
                    "warnings": runtime_rows[-1]["warnings"],
                    "terminal_state": "completed",
                }
            )
            if split == "oot":
                prediction_paths[(cell.dataset, cell.model, cell.method_id)] = cell_dir / metadata["path"]
                oot_rows.append(long_rows[-1].copy())

        cv = pd.read_csv(cell_dir / "results/cv_results.csv")
        cv["fold_numeric"] = pd.to_numeric(cv["fold"], errors="coerce")
        cv = cv.loc[cv["fold_numeric"].notna()].copy()
        if sorted(cv["fold_numeric"].astype(int).tolist()) != [1, 2, 3, 4, 5]:
            raise BaselineAuditError(f"invalid CV fold rows: {cell.cell_id}")
        for item in cv.itertuples(index=False):
            fold = int(item.fold_numeric)
            row = {
                "run_id": cell.cell_id,
                "cell_id": cell.cell_id,
                "cell_index": cell.cell_index,
                "phase": "dev_expanding_window_fold",
                "dataset": cell.dataset,
                "fold_or_split": fold,
                "fold": fold,
                "method_id": cell.method_id,
                "implementation_id": cell.implementation_id,
                "method_role": "control" if cell.method_id in COMPARISON_REFERENCES else "selector",
                "evaluation_model": cell.model,
                "model": cell.model,
                "candidate_universe_count": candidate_counts[cell.dataset],
                "candidate_universe_hash": "unsupported_not_persisted_by_prompt_10",
                "selected_feature_count": int(item.selected_features),
                "selected_features": int(item.selected_features),
                "budget_semantics": "natural_support_or_cap" if cell.method_id in NATURAL_SUPPORT_METHODS else ("full_universe" if cell.method_id == "full_features" else "exact_budget"),
                "requested_budget": cell.feature_budget,
                "natural_support": cell.method_id in NATURAL_SUPPORT_METHODS,
                "feasibility_state": "infeasible_natural_support" if cell.feature_budget is not None and int(item.selected_features) < cell.feature_budget else "completed",
                "seed": cell.seed,
                "prediction_sha256": None,
                "prediction_row_identity_sha256": None,
                "identity_target_sha256": None,
                "metric_artifact_sha256": finalized["results/prediction_metrics.csv"]["sha256"],
                "configuration_lock_sha256": plan.configuration_sha256,
                "auc": float(item.auc),
                "gini": float(item.gini),
                "ks": float(item.ks),
                "lift_at_10": float(item.lift_at_10),
                "runtime_seconds": float(item.fold_time_sec),
                "selection_seconds": float(item.feature_selection_time_sec),
                "model_fit_seconds": float(item.training_time_sec),
                "peak_process_rss_gib": runtime_rows[-1]["peak_process_rss_gib"],
                "minimum_available_ram_gib": runtime_rows[-1]["minimum_available_ram_gib"],
                "ram_wait_seconds": runtime_rows[-1]["ram_wait_seconds"],
                "warnings": "fold predictions not persisted; stored metrics cannot be prediction-recomputed",
                "terminal_state": "completed",
            }
            long_rows.append(row)
            fold_rows.append(row)

    for (dataset, split), hashes in identity_hashes.items():
        if len(hashes) != 1:
            raise BaselineAuditError(
                f"prediction identity/target mismatch across methods: {dataset}/{split}"
            )

    long_frame = pd.DataFrame(long_rows).sort_values(
        ["cell_index", "phase", "fold_or_split"], kind="mergesort"
    )
    fold_frame = pd.DataFrame(fold_rows)
    dev_summary = aggregate_dev_folds(fold_frame)
    runtime_frame = pd.DataFrame(runtime_rows)
    stability_frame = pd.DataFrame(stability_rows)
    comparisons, families = run_predeclared_comparisons(
        prediction_paths,
        fold_frame,
        selected_counts,
        repetitions=bootstrap_repetitions,
        seed=BOOTSTRAP_SEED,
    )
    feasibility = build_structural_feasibility(stability_frame, runtime_frame)
    reconciliation = pd.DataFrame(reconciliation_rows)
    final_cell = plan.cells[-1]
    final_success = (
        plan.results_root
        / "runs"
        / final_cell.dataset
        / final_cell.cell_id
        / "_SUCCESS"
    )
    if not final_success.is_file():
        raise BaselineAuditError("Final Cell 036 _SUCCESS marker is missing")
    authentication = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "matrix_id": "full_baseline_v1",
        "configuration_sha256": plan.configuration_sha256,
        "runtime_policy_sha256": plan.runtime_policy.source_sha256,
        "expected_cells": 36,
        "authenticated_cells": len(plan.cells),
        "cell_036": plan.cells[-1].cell_id,
        "cell_036_completed": True,
        "success_marker_present": True,
        "success_marker_path": final_success.relative_to(root).as_posix(),
        "success_marker_sha256": sha256_file(final_success),
        "phase_composition": {
            "dev_expanding_window_fold_evaluations": int(len(fold_frame)),
            "full_dev_refit_diagnostics": 36,
            "oot_evaluations": len(oot_rows),
            "total_evaluation_units": len(long_frame),
        },
        "prediction_identity_families": {
            f"{dataset}__{split}": next(iter(hashes))
            for (dataset, split), hashes in sorted(identity_hashes.items())
        },
        "metric_reconciliation_failures": int(reconciliation["status"].eq("fail").sum()),
        "raw_dataset_paths_resolved": False,
        "baseline_refit_performed": False,
    }
    return {
        "authentication": authentication,
        "baseline_results_long": long_frame,
        "baseline_metric_reconciliation": reconciliation,
        "baseline_dev_fold_summary": dev_summary,
        "baseline_oot_summary": pd.DataFrame(oot_rows),
        "baseline_pairwise_comparisons": comparisons,
        "baseline_holm_families": families,
        "baseline_selection_stability": stability_frame,
        "baseline_score_psi": pd.DataFrame(score_psi_rows),
        "baseline_feature_psi_audit": pd.DataFrame(feature_psi_rows),
        "baseline_runtime_resources": runtime_frame,
        "combination_structural_feasibility": feasibility,
    }


__all__ = [
    "AUDIT_SCHEMA_VERSION",
    "BOOTSTRAP_MINIMUM_VALID",
    "BOOTSTRAP_REPETITIONS",
    "BOOTSTRAP_SEED",
    "BaselineAuditError",
    "aggregate_dev_folds",
    "aggregate_random_k_replicates",
    "audit_completed_baselines",
    "authenticate_cell_artifacts",
    "build_structural_feasibility",
    "paired_stratified_auc_bootstrap_many",
    "recompute_prediction_metrics",
    "recompute_selection_stability",
    "run_predeclared_comparisons",
    "summarize_oot_availability",
    "validate_prediction_frame",
]
