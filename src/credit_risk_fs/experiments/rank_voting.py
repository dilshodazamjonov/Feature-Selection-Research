"""The explicit prospective cross-dataset rank-voting v1 aggregation rule."""

from __future__ import annotations

import math
import hashlib
import json
import logging
import time
import unicodedata
import gc
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


PROTOCOL_NAME = "cross_dataset_rank_voting_v1"
ELIGIBLE_VOTERS = ("rf_corr_mrmr", "boruta")
VOTER_ALIASES = {
    "rf_corr_mrmr": "rf_corr_mrmr",
    "randomforestrelevancemrmrselector": "rf_corr_mrmr",
    "boruta": "boruta",
    "borutaselector": "boruta",
}
REQUIRED_FIT_SCOPE = "dev_fold_training_only"
DEFAULT_FORBIDDEN_FEATURES = {
    "target",
    "sk_id_curr",
    "sk_id_bureau",
    "sk_id_prev",
    "loan_id",
    "id",
    "member_id",
    "recent_decision",
    "issue_d",
    "loan_status",
}


def _canonical_name(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("feature and voter names must be strings")
    name = unicodedata.normalize("NFC", value.strip())
    if not name:
        raise ValueError("feature and voter names must not be empty")
    return name.casefold()


def _canonical_voter(value: str) -> str:
    compact = _canonical_name(value).replace("_", "").replace("-", "")
    aliases = {
        alias.replace("_", "").replace("-", ""): canonical
        for alias, canonical in VOTER_ALIASES.items()
    }
    if compact not in aliases:
        raise ValueError(f"unknown voter for {PROTOCOL_NAME}: {value!r}")
    return aliases[compact]


def aggregate_cross_dataset_rank_voting(
    *,
    eligible_features: Iterable[str],
    rankings: Mapping[str, Iterable[str]],
    fit_scopes: Mapping[str, str],
    candidate_cap: int | None = None,
    forbidden_features: Iterable[str] = DEFAULT_FORBIDDEN_FEATURES,
) -> pd.DataFrame:
    """Aggregate the two registered fold-local rankings under the frozen rule."""

    universe = list(eligible_features)
    canonical_features = [_canonical_name(feature) for feature in universe]
    if len(set(canonical_features)) != len(canonical_features):
        raise ValueError("eligible feature universe contains duplicates or canonical-name collisions")
    forbidden = {_canonical_name(feature) for feature in forbidden_features}
    contamination = sorted(set(canonical_features) & forbidden)
    if contamination:
        raise ValueError(f"eligible feature universe contains leakage/identity fields: {contamination}")
    if not universe:
        raise ValueError("eligible feature universe must not be empty")
    feature_by_canonical = dict(zip(canonical_features, universe, strict=True))

    canonical_rankings: dict[str, list[str]] = {}
    for supplied_voter, supplied_features in rankings.items():
        voter = _canonical_voter(supplied_voter)
        if voter in canonical_rankings:
            raise ValueError(f"voter alias duplicates an existing vote: {voter}")
        ranking = list(supplied_features)
        normalized = [_canonical_name(feature) for feature in ranking]
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"{voter} ranking contains duplicate features")
        unknown = sorted(set(normalized) - set(canonical_features))
        if unknown:
            raise ValueError(f"{voter} ranking contains unknown features: {unknown}")
        if set(normalized) & forbidden:
            raise ValueError(f"{voter} ranking contains leakage/identity fields")
        canonical_rankings[voter] = normalized
    if set(canonical_rankings) != set(ELIGIBLE_VOTERS):
        raise ValueError(
            f"{PROTOCOL_NAME} requires exactly {list(ELIGIBLE_VOTERS)}; "
            f"received {sorted(canonical_rankings)}"
        )

    canonical_scopes: dict[str, str] = {}
    for supplied_voter, scope in fit_scopes.items():
        voter = _canonical_voter(supplied_voter)
        if voter in canonical_scopes:
            raise ValueError(f"voter alias duplicates a fitting boundary: {voter}")
        canonical_scopes[voter] = str(scope)
    allowed_fit_scopes = {REQUIRED_FIT_SCOPE, "full_dev_only"}
    if set(canonical_scopes) != set(ELIGIBLE_VOTERS) or any(
        scope not in allowed_fit_scopes for scope in canonical_scopes.values()
    ):
        raise ValueError(
            "every voter must use a registered fitting boundary: "
            f"{sorted(allowed_fit_scopes)!r}"
        )

    universe_size = len(universe)
    if candidate_cap is not None and (
        isinstance(candidate_cap, bool)
        or not isinstance(candidate_cap, int)
        or candidate_cap < 1
        or candidate_cap > universe_size
    ):
        raise ValueError("candidate_cap must be an integer within the eligible universe")

    rank_lookup = {
        voter: {feature: rank for rank, feature in enumerate(ranking, start=1)}
        for voter, ranking in canonical_rankings.items()
    }
    rows = []
    denominator = max(universe_size - 1, 1)
    for canonical_feature in canonical_features:
        row: dict[str, object] = {
            "feature": feature_by_canonical[canonical_feature],
            "normalized_feature_name": canonical_feature,
        }
        scores: list[float] = []
        ranks: list[int] = []
        presence_count = 0
        for voter in ELIGIBLE_VOTERS:
            rank = rank_lookup[voter].get(canonical_feature)
            present = rank is not None
            score = 1.0 - (rank - 1) / denominator if present else 0.0
            if not math.isfinite(score):
                raise ValueError(f"{voter} produced a non-finite normalized score")
            row[f"{voter}_raw_rank"] = rank
            row[f"{voter}_normalized_score"] = score
            row[f"{voter}_present"] = present
            scores.append(score)
            if present:
                presence_count += 1
                ranks.append(int(rank))
        row["aggregate_score"] = sum(scores) / len(ELIGIBLE_VOTERS)
        row["voter_presence_count"] = presence_count
        row["best_individual_rank"] = min(ranks) if ranks else universe_size + 1
        rows.append(row)

    result = pd.DataFrame(rows).sort_values(
        [
            "aggregate_score",
            "voter_presence_count",
            "best_individual_rank",
            "normalized_feature_name",
        ],
        ascending=[False, False, True, True],
        kind="mergesort",
    )
    result = result.reset_index(drop=True)
    result.insert(0, "aggregate_rank", range(1, len(result) + 1))
    return result.head(candidate_cap).copy() if candidate_cap is not None else result


def _ordered_name_hash(values: Iterable[str]) -> str:
    return hashlib.sha256(
        json.dumps(list(values), ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def build_long_voter_ranking_frame(
    *,
    dataset: str,
    fold_id: int,
    eligible_features: Iterable[str],
    rankings: Mapping[str, Iterable[str]],
    raw_scores: Mapping[str, Mapping[str, float | None]],
    selector_configurations: Mapping[str, Mapping[str, Any]],
    seed: int,
    training_row_identity_sha256: str,
    training_identity_target_sha256: str,
    input_artifact_hash: str,
    protocol_sha256: str,
    fit_scope: str = REQUIRED_FIT_SCOPE,
) -> pd.DataFrame:
    """Adapt fitted voter output to one canonical long row per feature/voter."""

    if fit_scope not in {REQUIRED_FIT_SCOPE, "full_dev_only"}:
        raise ValueError("unsupported voter ranking fit scope")
    universe = [str(value) for value in eligible_features]
    canonical = [_canonical_name(value) for value in universe]
    if len(set(canonical)) != len(canonical):
        raise ValueError("long voter adapter received a normalized-name collision")
    by_canonical = dict(zip(canonical, universe, strict=True))
    universe_size = len(universe)
    rows: list[dict[str, Any]] = []
    canonical_rankings: dict[str, list[str]] = {}
    for supplied_voter, supplied_ranking in rankings.items():
        voter = _canonical_voter(supplied_voter)
        if voter in canonical_rankings:
            raise ValueError(f"voter alias would cast a duplicate vote: {voter}")
        ranking = [_canonical_name(value) for value in supplied_ranking]
        if len(set(ranking)) != len(ranking):
            raise ValueError(f"{voter} ranking contains duplicate canonical votes")
        if set(ranking) - set(canonical):
            raise ValueError(f"{voter} ranking contains an unknown candidate")
        canonical_rankings[voter] = ranking
    if set(canonical_rankings) != set(ELIGIBLE_VOTERS):
        raise ValueError("long voter adapter requires exactly the two frozen voters")

    for voter in ELIGIBLE_VOTERS:
        rank_lookup = {
            feature: rank
            for rank, feature in enumerate(canonical_rankings[voter], start=1)
        }
        score_lookup = {
            _canonical_name(name): value
            for name, value in raw_scores.get(voter, {}).items()
        }
        configuration = json.dumps(
            dict(selector_configurations[voter]),
            sort_keys=True,
            separators=(",", ":"),
        )
        for normalized in canonical:
            rank = rank_lookup.get(normalized)
            present = rank is not None
            rows.append(
                {
                    "dataset": dataset,
                    "protocol_version": PROTOCOL_NAME,
                    "fold_id": int(fold_id),
                    "voter_id": voter,
                    "original_feature_name": by_canonical[normalized],
                    "normalized_feature_name": normalized,
                    "original_rank": rank,
                    "original_score": score_lookup.get(normalized),
                    "score_direction": "rank_1_is_best",
                    "candidate_universe_count": universe_size,
                    "present": bool(present),
                    "missing_rank_contribution": 0.0 if not present else None,
                    "normalized_score": (
                        1.0 - (int(rank) - 1) / max(universe_size - 1, 1)
                        if present
                        else 0.0
                    ),
                    "selector_configuration_json": configuration,
                    "seed": int(seed),
                    "fit_scope": fit_scope,
                    "training_row_identity_sha256": training_row_identity_sha256,
                    "training_identity_target_sha256": training_identity_target_sha256,
                    "candidate_universe_sha256": _ordered_name_hash(universe),
                    "input_artifact_hash": input_artifact_hash,
                    "protocol_sha256": protocol_sha256,
                }
            )
    frame = pd.DataFrame(rows)
    duplicates = frame.duplicated(["voter_id", "normalized_feature_name"])
    if len(frame) != 2 * universe_size or duplicates.any():
        raise ValueError("long voter ranking failed exact one-vote-per-candidate validation")
    return frame


def _canonical_first_fold(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    stable_row_ids: pd.Series,
    time_values: pd.Series,
) -> dict[str, Any]:
    if len(X) != len(y):
        raise ValueError("fold inputs have inconsistent row counts")
    projection = canonical_fold_projection(
        y=y,
        stable_row_ids=stable_row_ids,
        time_values=time_values,
        fold_id=1,
    )
    positions = projection["source_positions"]
    return {
        "X": X.iloc[positions].reset_index(drop=True),
        "y": projection["y"],
        "ids": projection["ids"],
        "times": projection["times"],
        "training_indices": projection["training_indices"],
        "validation_indices": projection["validation_indices"],
    }


def canonical_fold_projection(
    *,
    y: pd.Series,
    stable_row_ids: pd.Series,
    time_values: pd.Series,
    fold_id: int,
) -> dict[str, Any]:
    """Return canonical fold positions without materializing a feature-frame copy."""

    from credit_risk_fs.experiments.lendingclub_identity import stable_chronological_order
    from credit_risk_fs.models._cv_utils import GroupedTimeSeriesSplit

    if not (len(y) == len(stable_row_ids) == len(time_values)):
        raise ValueError("fold inputs have inconsistent row counts")
    if isinstance(fold_id, bool) or not isinstance(fold_id, int) or fold_id not in range(1, 6):
        raise ValueError("canonical fold_id must be an integer within [1, 5]")
    order_frame = pd.DataFrame(
        {
            "stable_row_id": stable_row_ids.astype(str).to_numpy(),
            "time_value": time_values.to_numpy(),
            "__source_position__": np.arange(len(y), dtype=np.int64),
        }
    )
    ordered = stable_chronological_order(
        order_frame,
        time_column="time_value",
        identity_column="stable_row_id",
    )
    positions = ordered["__source_position__"].to_numpy(dtype=int)
    y_ordered = y.iloc[positions].reset_index(drop=True)
    ids_ordered = ordered["stable_row_id"].reset_index(drop=True)
    times_ordered = ordered["time_value"].reset_index(drop=True)
    splitter = GroupedTimeSeriesSplit(n_splits=5, gap=1)
    splits = list(splitter.split(times_ordered.to_numpy()))
    if len(splits) != 5:
        raise ValueError("canonical grouped time splitter did not produce five folds")
    training_indices, validation_indices = splits[fold_id - 1]
    training_ids = ids_ordered.iloc[training_indices]
    validation_ids = ids_ordered.iloc[validation_indices]
    if set(training_ids) & set(validation_ids):
        raise ValueError("canonical first-fold training/validation identities overlap")
    return {
        "y": y_ordered,
        "ids": ids_ordered,
        "times": times_ordered,
        "source_positions": positions,
        "training_indices": training_indices,
        "validation_indices": validation_indices,
        "fold_id": fold_id,
    }


def _selector_configurations(seed: int, estimator_threads: int) -> dict[str, dict[str, Any]]:
    return {
        "rf_corr_mrmr": {
            "implementation": "RandomForestRelevanceMRMRSelector",
            "k": 300,
            "method": "mrmr",
            "scientific_identity": "rf_impurity_relevance_with_correlation_redundancy",
            "random_state": seed,
            "n_jobs": estimator_threads,
        },
        "boruta": {
            "implementation": "BorutaSelector.feature_ranking_",
            "max_iter": 15,
            "random_forest_estimators": 500,
            "random_forest_depth": 6,
            "random_state": seed,
            "n_jobs": estimator_threads,
        },
    }


def fit_voters_sequentially_memory_safe(
    *,
    X_numeric: pd.DataFrame,
    y: pd.Series,
    seed: int,
    estimator_threads: int,
    selector_factories: Mapping[str, Callable[[], Any]] | None = None,
    stage_callback: Callable[[str, int | None], None] | None = None,
    lifetime_observer: Callable[[str, Any | None], None] | None = None,
    fold_id: int | None = None,
) -> dict[str, Any]:
    """Fit frozen voters in order while retaining only compact voter output."""

    from credit_risk_fs.selectors.boruta import BorutaSelector
    from credit_risk_fs.selectors.mrmr import RandomForestRelevanceMRMRSelector

    if X_numeric.dtypes.astype(str).nunique() != 1 or str(X_numeric.dtypes.iloc[0]) != "float32":
        raise ValueError("refined voter input must preserve the frozen float32 dtype")
    factories = dict(selector_factories or {})
    if stage_callback:
        stage_callback("voter_rf_corr_mrmr", fold_id)
    mrmr = factories.get(
        "rf_corr_mrmr",
        lambda: RandomForestRelevanceMRMRSelector(
            k=300,
            method="mrmr",
            random_state=seed,
            n_jobs=estimator_threads,
        ),
    )()
    if lifetime_observer:
        lifetime_observer("rf_corr_mrmr_constructed", mrmr)
    mrmr.fit(X_numeric, y)
    mrmr_ranking = list(mrmr.selected_features_ or [])
    raw_mrmr = {
        str(feature): float(value)
        for feature, value in getattr(
            mrmr, "rf_importances_", pd.Series(dtype=float)
        ).items()
    }
    del mrmr
    gc.collect()
    if lifetime_observer:
        lifetime_observer("rf_corr_mrmr_released", None)

    if stage_callback:
        stage_callback("voter_boruta", fold_id)
    boruta = factories.get(
        "boruta",
        lambda: BorutaSelector(
            max_iter=15,
            random_state=seed,
            n_features=None,
            n_jobs=estimator_threads,
        ),
    )()
    if lifetime_observer:
        lifetime_observer("boruta_constructed", boruta)
    boruta.fit(X_numeric, y)
    boruta_ranking = list(boruta.feature_ranking_ or [])
    del boruta
    gc.collect()
    if lifetime_observer:
        lifetime_observer("boruta_released", None)

    if len(mrmr_ranking) != 300 or len(boruta_ranking) != X_numeric.shape[1]:
        raise ValueError(
            "voter finite-ranking contract failed: "
            f"mrmr={len(mrmr_ranking)}, boruta={len(boruta_ranking)}"
        )
    return {
        "rankings": {
            "rf_corr_mrmr": mrmr_ranking,
            "boruta": boruta_ranking,
        },
        "raw_scores": {"rf_corr_mrmr": raw_mrmr, "boruta": {}},
        "selector_configurations": _selector_configurations(seed, estimator_threads),
    }


def fit_fold_local_voting_adapter(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    stable_row_ids: pd.Series,
    time_values: pd.Series,
    dataset: str,
    model_name: str,
    candidate_pool_budget: int = 200,
    seed: int = 42,
    estimator_threads: int = 1,
    protocol_sha256: str,
    input_artifact_hash: str,
    stage_callback: Callable[[str, int | None], None] | None = None,
    selector_factories: Mapping[str, Callable[[], Any]] | None = None,
) -> dict[str, Any]:
    """Fit both voters, aggregate, and run exact-budget RFE on first-fold training only."""

    from credit_risk_fs.experiments.row_alignment import (
        ordered_row_id_sha256,
        ordered_row_id_target_sha256,
    )
    from credit_risk_fs.preprocessing.encoding import OriginalFeatureNumericEncoder
    from credit_risk_fs.selectors.boruta import BorutaSelector
    from credit_risk_fs.selectors.mrmr import RandomForestRelevanceMRMRSelector
    from credit_risk_fs.selectors.rfe import RFESelector

    if model_name not in {"lr", "catboost"}:
        raise ValueError(f"unsupported final model: {model_name}")
    if candidate_pool_budget != 200 or seed != 42:
        raise ValueError("bounded pilot requires K=200 and seed=42")
    if estimator_threads < 1 or estimator_threads > 4:
        raise ValueError("voting pilot estimator threads must be within [1, 4]")
    candidates = [str(column) for column in X.columns]
    if candidate_pool_budget > len(candidates):
        raise ValueError("candidate pool budget exceeds frozen universe")
    fold = _canonical_first_fold(
        X=X,
        y=y,
        stable_row_ids=stable_row_ids,
        time_values=time_values,
    )
    tr_idx = fold["training_indices"]
    va_idx = fold["validation_indices"]
    X_train_raw = fold["X"].iloc[tr_idx].loc[:, candidates]
    X_validation_raw = fold["X"].iloc[va_idx].loc[:, candidates]
    y_train = fold["y"].iloc[tr_idx]
    y_validation = fold["y"].iloc[va_idx]
    training_ids = fold["ids"].iloc[tr_idx]
    validation_ids = fold["ids"].iloc[va_idx]
    training_hash = ordered_row_id_sha256(training_ids.tolist())
    training_target_hash = ordered_row_id_target_sha256(
        training_ids.tolist(), y_train.tolist()
    )

    selection_encoder = OriginalFeatureNumericEncoder()
    X_train_numeric = selection_encoder.fit_transform(X_train_raw)
    factories = dict(selector_factories or {})
    mrmr = factories.get("rf_corr_mrmr", lambda: RandomForestRelevanceMRMRSelector(
        k=300,
        method="mrmr",
        random_state=seed,
        n_jobs=estimator_threads,
    ))()
    boruta = factories.get("boruta", lambda: BorutaSelector(
        max_iter=15,
        random_state=seed,
        n_features=None,
        n_jobs=estimator_threads,
    ))()
    if stage_callback:
        stage_callback("voter_rf_corr_mrmr", 1)
    mrmr.fit(X_train_numeric, y_train)
    if stage_callback:
        stage_callback("voter_boruta", 1)
    boruta.fit(X_train_numeric, y_train)
    mrmr_ranking = list(mrmr.selected_features_ or [])
    boruta_ranking = list(boruta.feature_ranking_ or [])
    if len(mrmr_ranking) != 300 or len(boruta_ranking) != len(candidates):
        raise ValueError(
            "voter finite-ranking contract failed: "
            f"mrmr={len(mrmr_ranking)}, boruta={len(boruta_ranking)}"
        )
    raw_mrmr = {
        str(feature): float(value)
        for feature, value in getattr(mrmr, "rf_importances_", pd.Series(dtype=float)).items()
    }
    selector_configs = {
        "rf_corr_mrmr": {
            "implementation": "RandomForestRelevanceMRMRSelector",
            "k": 300,
            "method": "mrmr",
            "scientific_identity": "rf_impurity_relevance_with_correlation_redundancy",
            "random_state": seed,
            "n_jobs": estimator_threads,
        },
        "boruta": {
            "implementation": "BorutaSelector.feature_ranking_",
            "max_iter": 15,
            "random_forest_estimators": 500,
            "random_forest_depth": 6,
            "random_state": seed,
            "n_jobs": estimator_threads,
        },
    }
    rankings = {"rf_corr_mrmr": mrmr_ranking, "boruta": boruta_ranking}
    long_frame = build_long_voter_ranking_frame(
        dataset=dataset,
        fold_id=1,
        eligible_features=candidates,
        rankings=rankings,
        raw_scores={"rf_corr_mrmr": raw_mrmr, "boruta": {}},
        selector_configurations=selector_configs,
        seed=seed,
        training_row_identity_sha256=training_hash,
        training_identity_target_sha256=training_target_hash,
        input_artifact_hash=input_artifact_hash,
        protocol_sha256=protocol_sha256,
    )
    if stage_callback:
        stage_callback("rank_aggregation", 1)
    aggregate = aggregate_cross_dataset_rank_voting(
        eligible_features=candidates,
        rankings=rankings,
        fit_scopes={voter: REQUIRED_FIT_SCOPE for voter in ELIGIBLE_VOTERS},
    )
    aggregate.insert(0, "fold_id", 1)
    aggregate.insert(0, "dataset", dataset)
    aggregate["protocol_version"] = PROTOCOL_NAME
    aggregate["candidate_pool_membership"] = aggregate["aggregate_rank"].le(
        candidate_pool_budget
    )
    top_candidates = aggregate.head(candidate_pool_budget)["feature"].astype(str).tolist()
    if len(top_candidates) != 200 or len(set(top_candidates)) != 200:
        raise ValueError("aggregate top-200 candidate list is not exact and unique")

    if stage_callback:
        stage_callback("rfe", 1)
    rfe_encoder = OriginalFeatureNumericEncoder()
    X_rfe_train = rfe_encoder.fit_transform(X_train_raw.loc[:, top_candidates])
    final_budget = 20 if model_name == "lr" else 40
    rfe = factories.get("rfe", lambda: RFESelector(
        n_features=final_budget,
        step=10,
        random_state=seed,
        thread_count=estimator_threads,
    ))()
    rfe.fit(X_rfe_train, y_train)
    supported = set(rfe.selected_features_ or [])
    selected_features = [feature for feature in top_candidates if feature in supported]
    if len(selected_features) != final_budget or len(set(selected_features)) != final_budget:
        raise ValueError(
            "downstream RFE exact-budget validation failed: "
            f"expected={final_budget}, observed={len(selected_features)}"
        )
    trace = rfe.selection_trace_.copy()
    trace.insert(0, "fold_id", 1)
    trace.insert(0, "dataset", dataset)
    trace["aggregate_rank"] = trace["feature"].map(
        dict(zip(aggregate["feature"], aggregate["aggregate_rank"], strict=True))
    )
    trace = trace.sort_values("aggregate_rank", kind="mergesort").reset_index(drop=True)
    return {
        "X_train_raw": X_train_raw,
        "X_validation_raw": X_validation_raw,
        "y_train": y_train,
        "y_validation": y_validation,
        "training_ids": training_ids.reset_index(drop=True),
        "validation_ids": validation_ids.reset_index(drop=True),
        "training_times": fold["times"].iloc[tr_idx].reset_index(drop=True),
        "validation_times": fold["times"].iloc[va_idx].reset_index(drop=True),
        "training_row_identity_sha256": training_hash,
        "training_identity_target_sha256": training_target_hash,
        "voter_rankings": long_frame,
        "aggregate_ranking": aggregate,
        "candidate_features": top_candidates,
        "selected_features": selected_features,
        "rfe_trace": trace,
        "rfe_effective_config": dict(rfe.effective_estimator_config_ or {}),
        "selector_configurations": selector_configs,
        "fold_id": 1,
    }


def _fit_final_model(
    *,
    repository_root: Path,
    dataset: str,
    model_name: str,
    selected_features: list[str],
    X_train_raw: pd.DataFrame,
    y_train: pd.Series,
    X_validation_raw: pd.DataFrame,
    seed: int,
    estimator_threads: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    import yaml

    from credit_risk_fs.models.registry import get_model_bundle
    from credit_risk_fs.preprocessing.encoding import Preprocessor

    base = yaml.safe_load((repository_root / "configs/base.yaml").read_text(encoding="utf-8"))
    model_kwargs = dict(base["model_params"][model_name])
    model_kwargs["random_state"] = seed
    if model_name == "catboost":
        model_kwargs["thread_count"] = estimator_threads
    dataset_config = yaml.safe_load(
        (repository_root / f"configs/experiments/{dataset}_matrix.yaml").read_text(
            encoding="utf-8"
        )
    )
    preprocessor_kwargs = dict(dataset_config.get("preprocessor_kwargs", {}))
    preprocessor = Preprocessor(**preprocessor_kwargs)
    selected_train = X_train_raw.loc[:, selected_features]
    selected_validation = X_validation_raw.loc[:, selected_features]
    if list(selected_train.columns) != selected_features:
        raise ValueError("final-model projected column order mismatch")
    train_encoded = preprocessor.fit_transform(selected_train)
    validation_encoded = preprocessor.transform(selected_validation)
    get_model, _, predict_proba, _ = get_model_bundle(model_name, model_kwargs)
    model = get_model()
    # Held-out validation targets are deliberately not supplied as an eval set.
    model.fit(train_encoded, y_train, eval_set=None)
    probabilities = np.asarray(predict_proba(model, validation_encoded), dtype=float)
    if probabilities.ndim != 1 or not np.isfinite(probabilities).all():
        raise ValueError("final model produced invalid validation probabilities")
    actual_classes = [int(value) for value in model.model.classes_]
    if actual_classes != [0, 1]:
        raise ValueError(f"positive-class probability orientation is invalid: {actual_classes}")
    actual_params = model.model.get_params()
    if model_name == "lr":
        for key in ("solver", "max_iter", "class_weight", "random_state"):
            if actual_params.get(key) != model_kwargs[key]:
                raise ValueError(f"effective LR configuration mismatch for {key}")
        effective_penalty = _resolve_effective_lr_penalty(actual_params)
        task_type = "CPU"
        thread_count = 1
    else:
        for key, expected in model_kwargs.items():
            if actual_params.get(key) != expected:
                raise ValueError(
                    f"effective CatBoost configuration mismatch for {key}: "
                    f"expected={expected!r}, observed={actual_params.get(key)!r}"
                )
        fitted_params = model.model.get_all_params()
        task_type = str(fitted_params.get("task_type", "CPU"))
        thread_count = int(actual_params.get("thread_count", -1))
        if task_type.upper() != "CPU" or thread_count < 1 or thread_count > 4:
            raise ValueError("effective CatBoost task/thread configuration is unsafe")
    effective = {
        "model": model_name,
        "implementation": f"credit_risk_fs.models.{model_name}",
        "requested_model_configuration": model_kwargs,
        "actual_estimator_configuration": actual_params,
        "effective_penalty": effective_penalty if model_name == "lr" else None,
        "preprocessing": {
            "implementation": "credit_risk_fs.preprocessing.encoding.Preprocessor",
            "configuration": preprocessor_kwargs,
            "fit_boundary": "dev_fold_training_only",
            "input_original_feature_count": len(selected_features),
            "encoded_feature_count": train_encoded.shape[1],
        },
        "validation_target_used_for_fit": False,
        "probability_classes": actual_classes,
        "positive_probability_column": 1,
        "probability_orientation": "class_1_higher_default_risk",
        "task_type": task_type,
        "estimator_thread_count": thread_count,
        "final_feature_budget": len(selected_features),
    }
    return probabilities, effective


def _fit_full_dev_capacity_model(
    *,
    repository_root: Path,
    dataset: str,
    model_name: str,
    selected_features: list[str],
    X_train_raw: pd.DataFrame,
    y_train: pd.Series,
    seed: int,
    estimator_threads: int,
) -> dict[str, Any]:
    """Exercise the frozen full-DEV model fit without opening or scoring OOT."""

    import yaml

    from credit_risk_fs.models.registry import get_model_bundle
    from credit_risk_fs.preprocessing.encoding import Preprocessor

    base = yaml.safe_load((repository_root / "configs/base.yaml").read_text(encoding="utf-8"))
    model_kwargs = dict(base["model_params"][model_name])
    model_kwargs["random_state"] = seed
    if model_name == "catboost":
        model_kwargs["thread_count"] = estimator_threads
    dataset_config = yaml.safe_load(
        (repository_root / f"configs/experiments/{dataset}_matrix.yaml").read_text(
            encoding="utf-8"
        )
    )
    preprocessor_kwargs = dict(dataset_config.get("preprocessor_kwargs", {}))
    preprocessor = Preprocessor(**preprocessor_kwargs)
    selected_train = X_train_raw.loc[:, selected_features]
    train_encoded = preprocessor.fit_transform(selected_train)
    get_model, _, _, _ = get_model_bundle(model_name, model_kwargs)
    model = get_model()
    model.fit(train_encoded, y_train, eval_set=None)
    actual_classes = [int(value) for value in model.model.classes_]
    if actual_classes != [0, 1]:
        raise ValueError(f"positive-class probability orientation is invalid: {actual_classes}")
    actual_params = model.model.get_params()
    if model_name == "lr":
        for key in ("solver", "max_iter", "class_weight", "random_state"):
            if actual_params.get(key) != model_kwargs[key]:
                raise ValueError(f"effective LR configuration mismatch for {key}")
        effective_penalty = _resolve_effective_lr_penalty(actual_params)
        task_type = "CPU"
        thread_count = 1
    else:
        for key, expected in model_kwargs.items():
            if actual_params.get(key) != expected:
                raise ValueError(
                    f"effective CatBoost configuration mismatch for {key}: "
                    f"expected={expected!r}, observed={actual_params.get(key)!r}"
                )
        fitted_params = model.model.get_all_params()
        task_type = str(fitted_params.get("task_type", "CPU"))
        thread_count = int(actual_params.get("thread_count", -1))
        if task_type.upper() != "CPU" or thread_count not in range(1, 5):
            raise ValueError("effective CatBoost task/thread configuration is unsafe")
        effective_penalty = None
    effective = {
        "model": model_name,
        "implementation": f"credit_risk_fs.models.{model_name}",
        "requested_model_configuration": model_kwargs,
        "actual_estimator_configuration": actual_params,
        "effective_penalty": effective_penalty,
        "preprocessing": {
            "implementation": "credit_risk_fs.preprocessing.encoding.Preprocessor",
            "configuration": preprocessor_kwargs,
            "fit_boundary": "full_dev_only",
            "input_original_feature_count": len(selected_features),
            "encoded_feature_count": train_encoded.shape[1],
        },
        "validation_target_used_for_fit": False,
        "probability_classes": actual_classes,
        "positive_probability_column": 1,
        "probability_orientation": "class_1_higher_default_risk",
        "task_type": task_type,
        "estimator_thread_count": thread_count,
        "final_feature_budget": len(selected_features),
        "capacity_fit_only": True,
        "oot_opened": False,
        "oot_scored": False,
    }
    del model, train_encoded, selected_train, preprocessor
    gc.collect()
    return effective


def fit_rfe_memory_safe(
    *,
    X_numeric: pd.DataFrame,
    y: pd.Series,
    top_candidates: list[str],
    model_name: str,
    seed: int,
    estimator_threads: int,
    selector_factory: Callable[[], Any] | None = None,
    lifetime_observer: Callable[[str, Any | None], None] | None = None,
) -> dict[str, Any]:
    """Run one exact frozen RFE branch and release its fitted selector."""

    from credit_risk_fs.selectors.rfe import RFESelector

    if list(X_numeric.columns) != top_candidates:
        raise ValueError("RFE top-candidate order mismatch")
    if X_numeric.dtypes.astype(str).nunique() != 1 or str(X_numeric.dtypes.iloc[0]) != "float32":
        raise ValueError("refined RFE input must preserve the frozen float32 dtype")
    final_budget = 20 if model_name == "lr" else 40
    rfe = (
        selector_factory()
        if selector_factory is not None
        else RFESelector(
            n_features=final_budget,
            step=10,
            random_state=seed,
            thread_count=estimator_threads,
        )
    )
    if lifetime_observer:
        lifetime_observer(f"{model_name}_rfe_constructed", rfe)
    rfe.fit(X_numeric, y)
    supported = set(rfe.selected_features_ or [])
    selected = [feature for feature in top_candidates if feature in supported]
    if len(selected) != final_budget or len(set(selected)) != final_budget:
        raise ValueError(
            "downstream RFE exact-budget validation failed: "
            f"expected={final_budget}, observed={len(selected)}"
        )
    trace = rfe.selection_trace_.copy()
    effective = dict(rfe.effective_estimator_config_ or {})
    del rfe
    gc.collect()
    if lifetime_observer:
        lifetime_observer(f"{model_name}_rfe_released", None)
    return {
        "selected_features": selected,
        "rfe_trace": trace,
        "rfe_effective_config": effective,
    }


def _logical_bytes(value: Any) -> int:
    if isinstance(value, pd.DataFrame):
        return int(value.memory_usage(index=True, deep=True).sum())
    if isinstance(value, pd.Series):
        return int(value.memory_usage(index=True, deep=True))
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    return 0


def _append_ownership_snapshot(
    rows: list[dict[str, Any]],
    *,
    stage: str,
    branch: str | None,
    owned: Mapping[str, Any],
) -> None:
    import os
    import psutil

    rss = int(psutil.Process(os.getpid()).memory_info().rss)
    if not owned:
        rows.append(
            {
                "stage": stage,
                "branch": branch or "",
                "object_name": "none",
                "rows": 0,
                "columns": 0,
                "dtype_summary": "",
                "logical_bytes": 0,
                "worker_rss_bytes": rss,
                "live_representation_count": 0,
            }
        )
        return
    live_count = sum(_logical_bytes(value) > 0 for value in owned.values())
    for name, value in owned.items():
        shape = getattr(value, "shape", ())
        if isinstance(value, pd.DataFrame):
            dtype_summary = ";".join(sorted(set(value.dtypes.astype(str))))
        else:
            dtype_summary = str(getattr(value, "dtype", ""))
        rows.append(
            {
                "stage": stage,
                "branch": branch or "",
                "object_name": name,
                "rows": int(shape[0]) if len(shape) > 0 else len(value) if hasattr(value, "__len__") else 0,
                "columns": int(shape[1]) if len(shape) > 1 else 1 if len(shape) == 1 else 0,
                "dtype_summary": dtype_summary,
                "logical_bytes": _logical_bytes(value),
                "worker_rss_bytes": rss,
                "live_representation_count": live_count,
            }
        )


def _resolve_effective_lr_penalty(actual_params: Mapping[str, Any]) -> str:
    """Resolve sklearn's penalty deprecation bridge without changing configuration."""

    reported = actual_params.get("penalty")
    if reported == "l2":
        return "l2"
    if reported == "deprecated" and float(actual_params.get("l1_ratio", -1)) == 0.0:
        return "l2_via_sklearn_deprecation_bridge_l1_ratio_0"
    raise ValueError(
        "effective LR penalty cannot be resolved as L2 from the fitted estimator: "
        f"penalty={reported!r}, l1_ratio={actual_params.get('l1_ratio')!r}"
    )


def _resume_voting_pilot_after_selection(
    *,
    stop_event: Any,
    stage_queue: Any,
    checkpoint_identity: Mapping[str, Any],
    run_dir: Path,
    root: Path,
    dataset: str,
    model_name: str,
    candidate_pool_budget: int,
    seed: int,
    estimator_threads: int,
    protocol_sha256: str,
) -> dict[str, Any]:
    """Reuse validated selection artifacts and resume at final-model fitting."""

    from credit_risk_fs.experiments.atomic_io import inspect_artifact, write_csv_atomic, write_json_atomic
    from credit_risk_fs.experiments.checkpointing import CheckpointManager
    from credit_risk_fs.experiments.prediction_contract import (
        PILOT_COVERAGE,
        PROBABILITY_ORIENTATION,
        publish_prediction_artifact,
    )
    from credit_risk_fs.experiments.row_alignment import ordered_row_id_sha256
    from credit_risk_fs.pipelines.common import prepare_voting_pilot_dev_data
    from credit_risk_fs.utils.logging import run_log_context

    checkpoint = CheckpointManager(run_dir)
    logger = logging.getLogger("voting_pilot_resume")
    timings: list[dict[str, Any]] = []
    stage_started = time.perf_counter()

    def report(stage: str, fold_id: int | None = 1) -> None:
        nonlocal stage_started
        now = time.perf_counter()
        if timings:
            timings[-1]["elapsed_seconds"] = now - stage_started
        stage_started = now
        timings.append({"stage": stage, "fold_id": fold_id, "elapsed_seconds": None})
        stage_queue.put({"stage": stage, "fold_id": fold_id})
        if stop_event.is_set():
            raise RuntimeError(f"cooperative stop requested before stage {stage}")

    with run_log_context(run_dir / "run.log"):
        report("resume_reload_validated_dev", None)
        logger.info("Reusing validated selection checkpoint; reloading DEV projection only")
        prepared = prepare_voting_pilot_dev_data(root, dataset=dataset, csv_chunk_rows=25_000)
        input_artifact_hash = hashlib.sha256(
            json.dumps(
                prepared.source_artifact_hashes,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        prior_access = json.loads((run_dir / "data_access_log.json").read_text(encoding="utf-8"))
        if prior_access.get("combined_input_artifact_hash") != input_artifact_hash:
            raise ValueError("resume input-artifact hash differs from validated data checkpoint")
        fold = _canonical_first_fold(
            X=prepared.X,
            y=prepared.y,
            stable_row_ids=prepared.stable_row_ids,
            time_values=prepared.time_values,
        )
        tr_idx = fold["training_indices"]
        va_idx = fold["validation_indices"]
        training_ids = fold["ids"].iloc[tr_idx].reset_index(drop=True)
        validation_ids = fold["ids"].iloc[va_idx].reset_index(drop=True)
        prior_fold = json.loads((run_dir / "fold_identity_manifest.json").read_text(encoding="utf-8"))
        if prior_fold.get("training_identity_sha256") != ordered_row_id_sha256(
            training_ids.tolist()
        ) or prior_fold.get("validation_identity_sha256") != ordered_row_id_sha256(
            validation_ids.tolist()
        ):
            raise ValueError("resume fold identities differ from validated checkpoint")

        candidate_frame = pd.read_csv(run_dir / "candidate_features.csv")
        selected_frame = pd.read_csv(run_dir / "selected_features.csv").sort_values(
            "selection_order", kind="mergesort"
        )
        candidate_features = candidate_frame["feature"].astype(str).tolist()
        selected_features = selected_frame["feature"].astype(str).tolist()
        expected_final = 20 if model_name == "lr" else 40
        if (
            len(candidate_features) != candidate_pool_budget
            or len(set(candidate_features)) != candidate_pool_budget
            or len(selected_features) != expected_final
            or len(set(selected_features)) != expected_final
            or not set(selected_features).issubset(candidate_features)
        ):
            raise ValueError("resume selection artifact exact-budget contract failed")
        X_ordered = fold["X"]
        y_ordered = fold["y"]
        report("resume_final_model_fit", 1)
        probabilities, effective_model = _fit_final_model(
            repository_root=root,
            dataset=dataset,
            model_name=model_name,
            selected_features=selected_features,
            X_train_raw=X_ordered.iloc[tr_idx],
            y_train=y_ordered.iloc[tr_idx],
            X_validation_raw=X_ordered.iloc[va_idx],
            seed=seed,
            estimator_threads=estimator_threads,
        )
        effective_payload = {
            **effective_model,
            "rfe_effective_estimator_configuration": {
                "implementation": "catboost.CatBoostClassifier",
                "iterations": 500,
                "depth": 6,
                "learning_rate": 0.05,
                "verbose": False,
                "random_state": seed,
                "allow_writing_files": False,
                "thread_count": estimator_threads,
                "task_type": "CPU",
                "rfe_step": 10,
                "n_features_to_select": expected_final,
            },
            "candidate_pool_budget": candidate_pool_budget,
            "selector_seed": seed,
            "protocol_sha256": protocol_sha256,
            "checkpoint_resolved_config_hash": checkpoint_identity["resolved_config_hash"],
            "selection_checkpoint_reused": True,
        }
        effective_meta = write_json_atomic(
            run_dir / "effective_model_config.json", effective_payload, overwrite=False
        )
        checkpoint.transition("model_fit_completed", artifacts=(effective_meta,))

        report("resume_dev_prediction_publication", 1)
        validation_targets = y_ordered.iloc[va_idx].astype("int8").reset_index(drop=True)
        prediction_frame = pd.DataFrame(
            {
                "stable_row_id": validation_ids.astype(str),
                "target": validation_targets,
                "prediction_probability": probabilities,
                "predicted_class": (probabilities >= 0.5).astype("int8"),
                "fold_id": 1,
                "split": "DEV",
                "row_position_or_order_key": range(1, len(validation_ids) + 1),
                "dataset": dataset,
                "run_id": run_dir.name,
                "method": "rank_voting_v1",
                "model": model_name,
                "seed": seed,
                "coverage_type": PILOT_COVERAGE,
                "research_eligible": False,
                "comparison_eligible": False,
                "probability_orientation": PROBABILITY_ORIENTATION,
            }
        )
        prediction_meta, prediction_sidecar_meta, prediction_payload = publish_prediction_artifact(
            path=run_dir / "predictions_dev.csv",
            metadata_path=run_dir / "prediction_metadata.json",
            frame=prediction_frame,
            expected_identities=validation_ids,
            expected_targets=validation_targets,
            coverage_type=PILOT_COVERAGE,
            expected_split="DEV",
            research_eligible=False,
            comparison_eligible=False,
            context={
                "dataset": dataset,
                "run_id": run_dir.name,
                "method": "rank_voting_v1",
                "model": model_name,
                "seed": seed,
                "protocol_hash": protocol_sha256,
                "configuration_hash": checkpoint_identity["resolved_config_hash"],
                "fold_definition": "grouped_time_series_cv_5_splits_gap_1_expanding_fold_1",
                "split": "DEV",
            },
        )
        validation_payload = {
            "status": "passed",
            "purpose": "integration_resource_pilot",
            "research_eligible": False,
            "comparison_eligible": False,
            "training_validation_identity_overlap_count": 0,
            "training_rows": len(tr_idx),
            "validation_rows": len(va_idx),
            "candidate_universe_count": len(prepared.candidate_features),
            "voter_row_count": len(pd.read_csv(run_dir / "voter_rankings.csv")),
            "voter_count": 2,
            "top_k": len(candidate_features),
            "final_feature_count": len(selected_features),
            "prediction_row_count": len(prediction_frame),
            "prediction_contract": prediction_payload,
            "opened_oot_paths": [],
            "retained_oot_rows": 0,
            "api_calls": 0,
            "clip_embedding_shap_workloads": 0,
            "gpu_training": False,
            "selection_checkpoint_reused": True,
        }
        validation_meta = write_json_atomic(
            run_dir / "pilot_validation.json", validation_payload, overwrite=False
        )
        checkpoint.transition(
            "dev_prediction_completed",
            artifacts=(prediction_meta, prediction_sidecar_meta, validation_meta),
        )
        report("resume_worker_complete", 1)
        timings[-1]["elapsed_seconds"] = time.perf_counter() - stage_started
        timing_meta = write_csv_atomic(
            run_dir / "stage_timings.csv", pd.DataFrame(timings), overwrite=False
        )
        checkpoint.transition("dev_prediction_completed", artifacts=(timing_meta,))

    additional = {
        "voter_rankings": "voter_rankings.csv",
        "aggregate_ranking": "aggregate_ranking.csv",
        "candidate_features": "candidate_features.csv",
        "rfe_selection_trace": "rfe_selection_trace.csv",
        "candidate_projection_manifest": "candidate_projection_manifest.json",
        "selection_artifact_manifest": "selection_artifact_manifest.json",
        "fold_identity_manifest": "fold_identity_manifest.json",
        "data_access_log": "data_access_log.json",
        "effective_model_config": "effective_model_config.json",
        "prediction_metadata": "prediction_metadata.json",
        "pilot_validation": "pilot_validation.json",
        "stage_timings": "stage_timings.csv",
    }
    for relative in additional.values():
        inspect_artifact(run_dir / relative)
    return {
        "summary": {
            "purpose": "integration_resource_pilot",
            "research_eligible": False,
            "comparison_eligible": False,
            "dataset": dataset,
            "model": model_name,
            "fold_id": 1,
            "training_rows": len(tr_idx),
            "validation_rows": len(va_idx),
            "candidate_universe_count": len(prepared.candidate_features),
            "candidate_pool_count": len(candidate_features),
            "final_feature_count": len(selected_features),
            "prediction_rows": len(prediction_frame),
            "selection_checkpoint_reused": True,
        },
        "additional_artifacts": additional,
    }


def voting_pilot_worker(
    *,
    stop_event: Any,
    stage_queue: Any,
    experiment_config: Any,
    checkpoint_identity: Mapping[str, Any],
    run_directory: str,
    repository_root: str,
    dataset: str,
    model_name: str,
    candidate_pool_budget: int,
    seed: int,
    estimator_threads: int,
    protocol_sha256: str,
) -> dict[str, Any]:
    """Spawn-safe bounded pilot worker using the canonical checkpoint/artifact layer."""

    from credit_risk_fs.experiments.atomic_io import (
        inspect_artifact,
        write_csv_atomic,
        write_json_atomic,
    )
    from credit_risk_fs.experiments.checkpointing import CheckpointManager
    from credit_risk_fs.experiments.prediction_contract import (
        PILOT_COVERAGE,
        PROBABILITY_ORIENTATION,
        publish_prediction_artifact,
    )
    from credit_risk_fs.experiments.row_alignment import ordered_row_id_sha256
    from credit_risk_fs.pipelines.common import prepare_voting_pilot_dev_data
    from credit_risk_fs.utils.logging import run_log_context

    root = Path(repository_root).resolve()
    run_dir = Path(run_directory).resolve()
    checkpoint = CheckpointManager(run_dir)
    if "selection_completed" in checkpoint.load().get("completed_stages", []):
        return _resume_voting_pilot_after_selection(
            stop_event=stop_event,
            stage_queue=stage_queue,
            checkpoint_identity=checkpoint_identity,
            run_dir=run_dir,
            root=root,
            dataset=dataset,
            model_name=model_name,
            candidate_pool_budget=candidate_pool_budget,
            seed=seed,
            estimator_threads=estimator_threads,
            protocol_sha256=protocol_sha256,
        )
    logger = logging.getLogger("voting_pilot")
    stage_started = time.perf_counter()
    timings: list[dict[str, Any]] = []

    def report(stage: str, fold_id: int | None = 1) -> None:
        nonlocal stage_started
        now = time.perf_counter()
        if timings:
            timings[-1]["elapsed_seconds"] = now - stage_started
        stage_started = now
        timings.append({"stage": stage, "fold_id": fold_id, "elapsed_seconds": None})
        stage_queue.put({"stage": stage, "fold_id": fold_id})
        if stop_event.is_set():
            raise RuntimeError(f"cooperative stop requested before stage {stage}")

    with run_log_context(run_dir / "run.log"):
        report("dev_projected_loading", None)
        logger.info("Loading full frozen %s DEV projection; OOT retention is disabled", dataset)
        prepared = prepare_voting_pilot_dev_data(root, dataset=dataset, csv_chunk_rows=25_000)
        input_artifact_hash = hashlib.sha256(
            json.dumps(
                prepared.source_artifact_hashes,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        data_access_payload = {
            "schema_version": "dev_only_data_access_log_v1",
            "dataset": dataset,
            "opened_paths": prepared.data_access_log,
            "opened_oot_paths": [],
            "retained_oot_rows": 0,
            "load_oot": False,
            "implicit_all_column_requests": 0,
            "source_artifact_hashes": prepared.source_artifact_hashes,
            "combined_input_artifact_hash": input_artifact_hash,
            "load_report": prepared.data_load_report,
        }
        data_access_meta = write_json_atomic(
            run_dir / "data_access_log.json", data_access_payload, overwrite=False
        )
        report("canonical_first_fold", 1)
        fold_preview = _canonical_first_fold(
            X=prepared.X,
            y=prepared.y,
            stable_row_ids=prepared.stable_row_ids,
            time_values=prepared.time_values,
        )
        tr_idx = fold_preview["training_indices"]
        va_idx = fold_preview["validation_indices"]
        training_ids = fold_preview["ids"].iloc[tr_idx]
        validation_ids = fold_preview["ids"].iloc[va_idx]
        fold_manifest_payload = {
            "schema_version": "single_dev_fold_pilot_identity_v1",
            "dataset": dataset,
            "canonical_fold_id": 1,
            "pilot_id_suffix": "f0",
            "fold_protocol": "grouped_time_series_cv_5_splits_gap_1_expanding",
            "training_row_count": len(tr_idx),
            "validation_row_count": len(va_idx),
            "training_identity_sha256": ordered_row_id_sha256(training_ids.tolist()),
            "validation_identity_sha256": ordered_row_id_sha256(validation_ids.tolist()),
            "identity_overlap_count": len(set(training_ids) & set(validation_ids)),
            "training_time_min": int(fold_preview["times"].iloc[tr_idx].min()),
            "training_time_max": int(fold_preview["times"].iloc[tr_idx].max()),
            "validation_time_min": int(fold_preview["times"].iloc[va_idx].min()),
            "validation_time_max": int(fold_preview["times"].iloc[va_idx].max()),
            "fit_scope": REQUIRED_FIT_SCOPE,
            "full_dev_split_evidence": prepared.split_evidence,
        }
        if fold_manifest_payload["identity_overlap_count"]:
            raise ValueError("pilot fold identities overlap")
        fold_manifest_meta = write_json_atomic(
            run_dir / "fold_identity_manifest.json",
            fold_manifest_payload,
            overwrite=False,
        )
        checkpoint.transition(
            "data_validated", artifacts=(data_access_meta, fold_manifest_meta)
        )

        report("fold_local_selection", 1)
        result = fit_fold_local_voting_adapter(
            X=prepared.X,
            y=prepared.y,
            stable_row_ids=prepared.stable_row_ids,
            time_values=prepared.time_values,
            dataset=dataset,
            model_name=model_name,
            candidate_pool_budget=candidate_pool_budget,
            seed=seed,
            estimator_threads=estimator_threads,
            protocol_sha256=protocol_sha256,
            input_artifact_hash=input_artifact_hash,
            stage_callback=report,
        )
        voter_meta = write_csv_atomic(
            run_dir / "voter_rankings.csv",
            result["voter_rankings"],
            overwrite=False,
        )
        aggregate_meta = write_csv_atomic(
            run_dir / "aggregate_ranking.csv",
            result["aggregate_ranking"],
            overwrite=False,
        )
        aggregate_lookup = result["aggregate_ranking"].set_index("feature")
        candidate_frame = pd.DataFrame(
            {
                "dataset": dataset,
                "fold_id": 1,
                "feature": result["candidate_features"],
                "aggregate_rank": range(1, candidate_pool_budget + 1),
                "candidate_pool_budget": candidate_pool_budget,
            }
        )
        candidate_meta = write_csv_atomic(
            run_dir / "candidate_features.csv", candidate_frame, overwrite=False
        )
        trace = result["rfe_trace"].copy()
        trace_meta = write_csv_atomic(
            run_dir / "rfe_selection_trace.csv", trace, overwrite=False
        )
        fold_selection = candidate_frame.copy()
        rfe_rank_lookup = trace.set_index("feature")["rfe_rank"].to_dict()
        selected_set = set(result["selected_features"])
        fold_selection["rfe_rank"] = fold_selection["feature"].map(rfe_rank_lookup)
        fold_selection["selected"] = fold_selection["feature"].isin(selected_set)
        fold_selection["final_feature_budget"] = len(selected_set)
        fold_selection_meta = write_csv_atomic(
            run_dir / "fold_selections.csv", fold_selection, overwrite=False
        )
        selected_frame = pd.DataFrame(
            {
                "dataset": dataset,
                "fold_id": 1,
                "model": model_name,
                "feature": result["selected_features"],
                "selection_order": range(1, len(result["selected_features"]) + 1),
                "aggregate_rank": [
                    int(aggregate_lookup.loc[feature, "aggregate_rank"])
                    for feature in result["selected_features"]
                ],
                "final_feature_budget": len(result["selected_features"]),
            }
        )
        selected_meta = write_csv_atomic(
            run_dir / "selected_features.csv", selected_frame, overwrite=False
        )
        normalized_mapping = [
            {
                "original_feature_name": feature,
                "normalized_feature_name": _canonical_name(feature),
            }
            for feature in prepared.candidate_features
        ]
        projection_payload = {
            "schema_version": "candidate_projection_manifest_v1",
            "dataset": dataset,
            "source_projections": prepared.source_projections,
            "source_projection_counts": {
                table: len(columns) for table, columns in prepared.source_projections.items()
            },
            "row_validation_columns": ["stable_row_id", "target", "time_value"],
            "voter_columns": list(prepared.candidate_features),
            "voter_column_count": len(prepared.candidate_features),
            "candidate_universe_sha256": _ordered_name_hash(prepared.candidate_features),
            "aggregation_inputs": ["voter_rankings.csv"],
            "rfe_columns": result["candidate_features"],
            "rfe_column_count": len(result["candidate_features"]),
            "final_model_columns": result["selected_features"],
            "final_model_column_count": len(result["selected_features"]),
            "evaluation_columns": [
                "stable_row_id",
                "target",
                "prediction_probability",
                "predicted_class",
                "fold_id",
                "split",
                "row_position_or_order_key",
            ],
            "original_normalized_mapping": normalized_mapping,
            "missing_columns": [],
            "extra_columns": [],
            "reordered_columns": False,
            "implicit_all_column_requests": 0,
            "source_artifact_hashes": prepared.source_artifact_hashes,
        }
        projection_meta = write_json_atomic(
            run_dir / "candidate_projection_manifest.json",
            projection_payload,
            overwrite=False,
        )
        selection_integrity = {
            "voter_rankings": voter_meta.to_dict(),
            "aggregate_ranking": aggregate_meta.to_dict(),
            "candidate_features": candidate_meta.to_dict(),
            "rfe_selection_trace": trace_meta.to_dict(),
            "fold_selections": fold_selection_meta.to_dict(),
            "selected_features": selected_meta.to_dict(),
            "candidate_projection_manifest": projection_meta.to_dict(),
        }
        selection_integrity_meta = write_json_atomic(
            run_dir / "selection_artifact_manifest.json",
            selection_integrity,
            overwrite=False,
        )
        checkpoint.transition(
            "selection_completed",
            artifacts=(
                voter_meta,
                aggregate_meta,
                candidate_meta,
                trace_meta,
                fold_selection_meta,
                selected_meta,
                projection_meta,
                selection_integrity_meta,
            ),
            completed_fold_id=1,
        )

        report("final_model_fit", 1)
        probabilities, effective_model = _fit_final_model(
            repository_root=root,
            dataset=dataset,
            model_name=model_name,
            selected_features=result["selected_features"],
            X_train_raw=result["X_train_raw"],
            y_train=result["y_train"],
            X_validation_raw=result["X_validation_raw"],
            seed=seed,
            estimator_threads=estimator_threads,
        )
        effective_payload = {
            **effective_model,
            "rfe_effective_estimator_configuration": result["rfe_effective_config"],
            "candidate_pool_budget": candidate_pool_budget,
            "selector_seed": seed,
            "protocol_sha256": protocol_sha256,
            "checkpoint_resolved_config_hash": checkpoint_identity["resolved_config_hash"],
        }
        effective_meta = write_json_atomic(
            run_dir / "effective_model_config.json",
            effective_payload,
            overwrite=False,
        )
        checkpoint.transition("model_fit_completed", artifacts=(effective_meta,))

        report("dev_prediction_publication", 1)
        validation_ids = result["validation_ids"].astype(str).reset_index(drop=True)
        validation_targets = result["y_validation"].astype("int8").reset_index(drop=True)
        prediction_frame = pd.DataFrame(
            {
                "stable_row_id": validation_ids,
                "target": validation_targets,
                "prediction_probability": probabilities,
                "predicted_class": (probabilities >= 0.5).astype("int8"),
                "fold_id": 1,
                "split": "DEV",
                "row_position_or_order_key": range(1, len(validation_ids) + 1),
                "dataset": dataset,
                "run_id": run_dir.name,
                "method": "rank_voting_v1",
                "model": model_name,
                "seed": seed,
                "coverage_type": PILOT_COVERAGE,
                "research_eligible": False,
                "comparison_eligible": False,
                "probability_orientation": PROBABILITY_ORIENTATION,
            }
        )
        prediction_meta, prediction_sidecar_meta, prediction_payload = publish_prediction_artifact(
            path=run_dir / "predictions_dev.csv",
            metadata_path=run_dir / "prediction_metadata.json",
            frame=prediction_frame,
            expected_identities=validation_ids,
            expected_targets=validation_targets,
            coverage_type=PILOT_COVERAGE,
            expected_split="DEV",
            research_eligible=False,
            comparison_eligible=False,
            context={
                "dataset": dataset,
                "run_id": run_dir.name,
                "method": "rank_voting_v1",
                "model": model_name,
                "seed": seed,
                "protocol_hash": protocol_sha256,
                "configuration_hash": checkpoint_identity["resolved_config_hash"],
                "fold_definition": "grouped_time_series_cv_5_splits_gap_1_expanding_fold_1",
                "split": "DEV",
            },
        )
        validation_payload = {
            "status": "passed",
            "purpose": "integration_resource_pilot",
            "research_eligible": False,
            "comparison_eligible": False,
            "training_validation_identity_overlap_count": 0,
            "training_rows": len(result["training_ids"]),
            "validation_rows": len(result["validation_ids"]),
            "candidate_universe_count": len(prepared.candidate_features),
            "voter_row_count": len(result["voter_rankings"]),
            "voter_count": 2,
            "top_k": len(result["candidate_features"]),
            "final_feature_count": len(result["selected_features"]),
            "prediction_row_count": len(prediction_frame),
            "prediction_contract": prediction_payload,
            "opened_oot_paths": [],
            "retained_oot_rows": 0,
            "api_calls": 0,
            "clip_embedding_shap_workloads": 0,
            "gpu_training": False,
        }
        validation_meta = write_json_atomic(
            run_dir / "pilot_validation.json", validation_payload, overwrite=False
        )
        checkpoint.transition(
            "dev_prediction_completed",
            artifacts=(prediction_meta, prediction_sidecar_meta, validation_meta),
        )
        report("worker_complete", 1)
        timings[-1]["elapsed_seconds"] = time.perf_counter() - stage_started
        timing_frame = pd.DataFrame(timings)
        timing_meta = write_csv_atomic(
            run_dir / "stage_timings.csv", timing_frame, overwrite=False
        )
        checkpoint.transition(
            "dev_prediction_completed", artifacts=(timing_meta,)
        )

    additional = {
        "voter_rankings": "voter_rankings.csv",
        "aggregate_ranking": "aggregate_ranking.csv",
        "candidate_features": "candidate_features.csv",
        "rfe_selection_trace": "rfe_selection_trace.csv",
        "candidate_projection_manifest": "candidate_projection_manifest.json",
        "selection_artifact_manifest": "selection_artifact_manifest.json",
        "fold_identity_manifest": "fold_identity_manifest.json",
        "data_access_log": "data_access_log.json",
        "effective_model_config": "effective_model_config.json",
        "prediction_metadata": "prediction_metadata.json",
        "pilot_validation": "pilot_validation.json",
        "stage_timings": "stage_timings.csv",
    }
    # Fail closed before returning control to the lifecycle finalizer.
    for relative in additional.values():
        inspect_artifact(run_dir / relative)
    return {
        "summary": {
            "purpose": "integration_resource_pilot",
            "research_eligible": False,
            "comparison_eligible": False,
            "dataset": dataset,
            "model": model_name,
            "fold_id": 1,
            "training_rows": len(result["training_ids"]),
            "validation_rows": len(result["validation_ids"]),
            "candidate_universe_count": len(prepared.candidate_features),
            "candidate_pool_count": len(result["candidate_features"]),
            "final_feature_count": len(result["selected_features"]),
            "prediction_rows": len(prediction_frame),
        },
        "additional_artifacts": additional,
    }


def lendingclub_memory_capacity_worker(
    *,
    stop_event: Any,
    stage_queue: Any,
    experiment_config: Any,
    checkpoint_identity: Mapping[str, Any],
    run_directory: str,
    repository_root: str,
    scenario: Mapping[str, Any],
    estimator_threads: int,
    protocol_sha256: str,
) -> dict[str, Any]:
    """Run one Prompt-5 capacity shape through the registered lifecycle."""

    from credit_risk_fs.experiments.atomic_io import (
        inspect_artifact,
        write_csv_atomic,
        write_json_atomic,
    )
    from credit_risk_fs.experiments.checkpointing import CheckpointManager
    from credit_risk_fs.experiments.prediction_contract import (
        CAPACITY_SINGLE_FOLD_COVERAGE,
        PROBABILITY_ORIENTATION,
        publish_prediction_artifact,
    )
    from credit_risk_fs.experiments.row_alignment import (
        ordered_row_id_sha256,
        ordered_row_id_target_sha256,
    )
    from credit_risk_fs.pipelines.common import prepare_voting_pilot_dev_data
    from credit_risk_fs.preprocessing.encoding import OriginalFeatureNumericEncoder
    from credit_risk_fs.utils.logging import run_log_context

    root = Path(repository_root).resolve()
    run_dir = Path(run_directory).resolve()
    checkpoint = CheckpointManager(run_dir)
    scenario_values = dict(scenario)
    scenario_id = str(scenario_values.get("scenario_id", ""))
    mode = str(scenario_values.get("mode", ""))
    fold_id = scenario_values.get("fold_id")
    candidate_pool_budget = int(scenario_values.get("candidate_pool", -1))
    seed = int(scenario_values.get("seed", -1))
    branches = list(scenario_values.get("branches", []))
    expected = {
        "dataset": "lendingclub_v2",
        "candidate_pool": candidate_pool_budget,
        "seed": 42,
        "branches": ["lr", "catboost"],
        "load_oot": False,
        "research_eligible": False,
        "comparison_eligible": False,
    }
    for key, value in expected.items():
        if scenario_values.get(key) != value:
            raise ValueError(f"capacity scenario invariant mismatch for {key}")
    if candidate_pool_budget not in {200, 300} or mode not in {"fold", "full_dev"}:
        raise ValueError("unsupported capacity scenario shape")
    if mode == "fold" and fold_id not in range(1, 6):
        raise ValueError("fold capacity scenario requires canonical fold within [1, 5]")
    if mode == "full_dev" and fold_id is not None:
        raise ValueError("full-DEV capacity scenario cannot name a fold")
    if estimator_threads not in range(1, 5):
        raise ValueError("capacity estimator threads must remain within [1, 4]")

    logger = logging.getLogger("lendingclub_memory_capacity")
    timings: list[dict[str, Any]] = []
    ownership_rows: list[dict[str, Any]] = []
    stage_started = time.perf_counter()

    def report(stage: str, reported_fold: int | None = None) -> None:
        nonlocal stage_started
        now = time.perf_counter()
        if timings:
            timings[-1]["elapsed_seconds"] = now - stage_started
        stage_started = now
        timings.append(
            {
                "stage": stage,
                "fold_id": reported_fold if reported_fold is not None else "",
                "elapsed_seconds": None,
            }
        )
        stage_queue.put({"stage": stage, "fold_id": reported_fold})
        if stop_event.is_set():
            raise RuntimeError(f"cooperative stop requested before stage {stage}")

    additional: dict[str, str] = {}
    with run_log_context(run_dir / "run.log"):
        completed_stages = set(checkpoint.load().get("completed_stages", []))
        selection_reused = "selection_completed" in completed_stages
        input_artifact_hash = ""
        full_dtypes: dict[str, str] = {}
        source_artifact_hashes: dict[str, str] = {}
        source_projections: dict[str, list[str]] = {}
        fold: dict[str, Any]

        if not selection_reused:
            report("dev_projected_loading", fold_id if mode == "fold" else None)
            prepared = prepare_voting_pilot_dev_data(
                root,
                dataset="lendingclub_v2",
                csv_chunk_rows=25_000,
            )
            universe = tuple(prepared.candidate_universe or prepared.candidate_features)
            if len(universe) != 675 or tuple(prepared.candidate_features) != universe:
                raise ValueError("capacity voter load did not retain the exact 675-feature universe")
            input_artifact_hash = hashlib.sha256(
                json.dumps(
                    prepared.source_artifact_hashes,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            source_artifact_hashes = dict(prepared.source_artifact_hashes)
            source_projections = dict(prepared.source_projections)
            full_dtypes = {str(name): str(dtype) for name, dtype in prepared.X.dtypes.items()}
            _append_ownership_snapshot(
                ownership_rows,
                stage="dev_projected_loading_complete",
                branch=None,
                owned={"candidate_source_frame": prepared.X},
            )

            fold = canonical_fold_projection(
                y=prepared.y,
                stable_row_ids=prepared.stable_row_ids,
                time_values=prepared.time_values,
                fold_id=int(fold_id or 1),
            )
            if mode == "full_dev":
                training_indices = np.arange(len(fold["y"]), dtype=np.int64)
                validation_indices = np.asarray([], dtype=np.int64)
            else:
                training_indices = fold["training_indices"]
                validation_indices = fold["validation_indices"]
            training_ids = fold["ids"].iloc[training_indices].reset_index(drop=True)
            validation_ids = fold["ids"].iloc[validation_indices].reset_index(drop=True)
            y_train = fold["y"].iloc[training_indices].reset_index(drop=True)
            y_validation = fold["y"].iloc[validation_indices].reset_index(drop=True)
            if set(training_ids) & set(validation_ids):
                raise ValueError("capacity training and validation identities overlap")

            data_access_payload = {
                "schema_version": "memory_capacity_data_access_v1",
                "scenario_id": scenario_id,
                "purpose": "memory_capacity_validation",
                "opened_paths": prepared.data_access_log,
                "opened_oot_paths": [],
                "retained_oot_rows": 0,
                "load_oot": False,
                "oot_scored": False,
                "implicit_all_column_requests": 0,
                "source_artifact_hashes": source_artifact_hashes,
                "combined_input_artifact_hash": input_artifact_hash,
                "load_report": prepared.data_load_report,
            }
            data_access_meta = write_json_atomic(
                run_dir / "data_access_log.json", data_access_payload, overwrite=False
            )
            fold_manifest_payload = {
                "schema_version": "memory_capacity_identity_v1",
                "scenario_id": scenario_id,
                "mode": mode,
                "canonical_fold_id": fold_id,
                "training_row_count": len(training_indices),
                "validation_row_count": len(validation_indices),
                "training_identity_sha256": ordered_row_id_sha256(training_ids.tolist()),
                "training_identity_target_sha256": ordered_row_id_target_sha256(
                    training_ids.tolist(), y_train.tolist()
                ),
                "validation_identity_sha256": (
                    ordered_row_id_sha256(validation_ids.tolist())
                    if len(validation_ids)
                    else None
                ),
                "validation_identity_target_sha256": (
                    ordered_row_id_target_sha256(
                        validation_ids.tolist(), y_validation.tolist()
                    )
                    if len(validation_ids)
                    else None
                ),
                "identity_overlap_count": 0,
                "candidate_universe_count": len(universe),
                "candidate_universe_sha256": _ordered_name_hash(universe),
                "full_dev_split_evidence": prepared.split_evidence,
                "fit_scope": REQUIRED_FIT_SCOPE if mode == "fold" else "full_dev_only",
            }
            fold_manifest_meta = write_json_atomic(
                run_dir / "fold_identity_manifest.json",
                fold_manifest_payload,
                overwrite=False,
            )
            checkpoint.transition(
                "data_validated", artifacts=(data_access_meta, fold_manifest_meta)
            )

            report("training_candidate_projection", fold_id if mode == "fold" else None)
            source_positions = fold["source_positions"][training_indices]
            X_source = prepared.X
            X_train_raw = X_source.iloc[source_positions].reset_index(drop=True)
            if list(X_train_raw.columns) != list(universe):
                raise ValueError("training candidate projection changed feature order")
            prepared.X = pd.DataFrame()
            del X_source, prepared
            gc.collect()
            _append_ownership_snapshot(
                ownership_rows,
                stage="training_candidate_projection_complete",
                branch=None,
                owned={"training_candidate_frame": X_train_raw},
            )

            report("selection_encoding", fold_id if mode == "fold" else None)
            selection_encoder = OriginalFeatureNumericEncoder()
            X_numeric = selection_encoder.fit_transform(X_train_raw)
            del X_train_raw, selection_encoder
            gc.collect()
            _append_ownership_snapshot(
                ownership_rows,
                stage="selection_encoding_source_released",
                branch=None,
                owned={"selection_numeric_matrix": X_numeric, "training_target": y_train},
            )

            voter_result = fit_voters_sequentially_memory_safe(
                X_numeric=X_numeric,
                y=y_train,
                seed=seed,
                estimator_threads=estimator_threads,
                stage_callback=report,
                fold_id=int(fold_id) if fold_id is not None else None,
            )
            del X_numeric
            gc.collect()
            _append_ownership_snapshot(
                ownership_rows,
                stage="voter_material_released",
                branch=None,
                owned={},
            )

            long_frame = build_long_voter_ranking_frame(
                dataset="lendingclub_v2",
                fold_id=int(fold_id or 0),
                eligible_features=universe,
                rankings=voter_result["rankings"],
                raw_scores=voter_result["raw_scores"],
                selector_configurations=voter_result["selector_configurations"],
                seed=seed,
                training_row_identity_sha256=fold_manifest_payload[
                    "training_identity_sha256"
                ],
                training_identity_target_sha256=fold_manifest_payload[
                    "training_identity_target_sha256"
                ],
                input_artifact_hash=input_artifact_hash,
                protocol_sha256=protocol_sha256,
                fit_scope=fold_manifest_payload["fit_scope"],
            )
            report("rank_aggregation", fold_id if mode == "fold" else None)
            aggregate = aggregate_cross_dataset_rank_voting(
                eligible_features=universe,
                rankings=voter_result["rankings"],
                fit_scopes={
                    voter: (
                        REQUIRED_FIT_SCOPE
                        if mode == "fold"
                        else "full_dev_only"
                    )
                    for voter in ELIGIBLE_VOTERS
                },
            )
            aggregate.insert(0, "fold_id", int(fold_id or 0))
            aggregate.insert(0, "dataset", "lendingclub_v2")
            aggregate["protocol_version"] = PROTOCOL_NAME
            aggregate["candidate_pool_membership"] = aggregate["aggregate_rank"].le(
                candidate_pool_budget
            )
            top_candidates = (
                aggregate.head(candidate_pool_budget)["feature"].astype(str).tolist()
            )
            if len(top_candidates) != candidate_pool_budget or len(set(top_candidates)) != candidate_pool_budget:
                raise ValueError("capacity aggregate top-K contract failed")
            candidate_frame = pd.DataFrame(
                {
                    "dataset": "lendingclub_v2",
                    "fold_id": int(fold_id or 0),
                    "feature": top_candidates,
                    "aggregate_rank": range(1, candidate_pool_budget + 1),
                    "candidate_pool_budget": candidate_pool_budget,
                }
            )
            voter_meta = write_csv_atomic(
                run_dir / "voter_rankings.csv", long_frame, overwrite=False
            )
            aggregate_meta = write_csv_atomic(
                run_dir / "aggregate_ranking.csv", aggregate, overwrite=False
            )
            candidate_meta = write_csv_atomic(
                run_dir / "candidate_features.csv", candidate_frame, overwrite=False
            )
            dtype_meta = write_json_atomic(
                run_dir / "dtype_manifest.json",
                {
                    "schema_version": "memory_refinement_dtype_manifest_v1",
                    "source_candidate_dtypes": full_dtypes,
                    "selection_output_dtype": "float32",
                    "numeric_precision_changed": False,
                    "candidate_universe_count": 675,
                    "candidate_pool_count": candidate_pool_budget,
                },
                overwrite=False,
            )
            checkpoint.transition(
                "selection_completed",
                artifacts=(voter_meta, aggregate_meta, candidate_meta, dtype_meta),
                completed_fold_id=fold_id if fold_id is not None else "full_dev",
            )
            additional.update(
                {
                    "voter_rankings": "voter_rankings.csv",
                    "aggregate_ranking": "aggregate_ranking.csv",
                    "candidate_features": "candidate_features.csv",
                    "dtype_manifest": "dtype_manifest.json",
                }
            )
            del long_frame, aggregate, candidate_frame, voter_result
            gc.collect()
        else:
            logger.info("Reusing compact validated voter/ranking checkpoint for %s", scenario_id)
            fold_manifest_payload = json.loads(
                (run_dir / "fold_identity_manifest.json").read_text(encoding="utf-8")
            )
            data_access_payload = json.loads(
                (run_dir / "data_access_log.json").read_text(encoding="utf-8")
            )
            input_artifact_hash = str(data_access_payload["combined_input_artifact_hash"])
            source_artifact_hashes = dict(data_access_payload["source_artifact_hashes"])
            top_candidates = pd.read_csv(run_dir / "candidate_features.csv")[
                "feature"
            ].astype(str).tolist()
            if len(top_candidates) != candidate_pool_budget:
                raise ValueError("resume candidate budget differs from checkpoint")
            additional.update(
                {
                    "voter_rankings": "voter_rankings.csv",
                    "aggregate_ranking": "aggregate_ranking.csv",
                    "candidate_features": "candidate_features.csv",
                    "dtype_manifest": "dtype_manifest.json",
                }
            )

        report("top_k_projected_reload", fold_id if mode == "fold" else None)
        projected = prepare_voting_pilot_dev_data(
            root,
            dataset="lendingclub_v2",
            csv_chunk_rows=25_000,
            projected_candidate_features=top_candidates,
            csv_low_memory=False,
        )
        projected_input_hash = hashlib.sha256(
            json.dumps(
                projected.source_artifact_hashes,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if projected_input_hash != input_artifact_hash:
            raise ValueError("top-K reload source hash differs from voter source")
        if tuple(projected.candidate_features) != tuple(top_candidates):
            raise ValueError("top-K reload changed aggregate candidate order")
        dtype_payload = json.loads(
            (run_dir / "dtype_manifest.json").read_text(encoding="utf-8")
        )
        expected_projected_dtypes = {
            feature: dtype_payload["source_candidate_dtypes"][feature]
            for feature in top_candidates
        }
        observed_projected_dtypes = {
            str(name): str(dtype) for name, dtype in projected.X.dtypes.items()
        }
        if observed_projected_dtypes != expected_projected_dtypes:
            raise ValueError(
                "top-K sparse reload changed a frozen source dtype: "
                f"expected={expected_projected_dtypes}, "
                f"observed={observed_projected_dtypes}"
            )
        fold = canonical_fold_projection(
            y=projected.y,
            stable_row_ids=projected.stable_row_ids,
            time_values=projected.time_values,
            fold_id=int(fold_id or 1),
        )
        if mode == "full_dev":
            training_indices = np.arange(len(fold["y"]), dtype=np.int64)
            validation_indices = np.asarray([], dtype=np.int64)
        else:
            training_indices = fold["training_indices"]
            validation_indices = fold["validation_indices"]
        training_ids = fold["ids"].iloc[training_indices].reset_index(drop=True)
        validation_ids = fold["ids"].iloc[validation_indices].reset_index(drop=True)
        y_train = fold["y"].iloc[training_indices].reset_index(drop=True)
        y_validation = fold["y"].iloc[validation_indices].reset_index(drop=True)
        if ordered_row_id_sha256(training_ids.tolist()) != fold_manifest_payload[
            "training_identity_sha256"
        ]:
            raise ValueError("top-K reload changed training identity order")
        if len(validation_ids) and ordered_row_id_sha256(
            validation_ids.tolist()
        ) != fold_manifest_payload["validation_identity_sha256"]:
            raise ValueError("top-K reload changed validation identity order")

        top_source = projected.X
        X_train_top = top_source.iloc[
            fold["source_positions"][training_indices]
        ].reset_index(drop=True)
        X_validation_top = (
            top_source.iloc[fold["source_positions"][validation_indices]].reset_index(
                drop=True
            )
            if len(validation_indices)
            else pd.DataFrame(columns=top_candidates)
        )
        projected.X = pd.DataFrame()
        del top_source, projected
        gc.collect()
        _append_ownership_snapshot(
            ownership_rows,
            stage="top_k_projection_source_released",
            branch=None,
            owned={
                "top_k_training_frame": X_train_top,
                "top_k_validation_frame": X_validation_top,
            },
        )
        top_encoder = OriginalFeatureNumericEncoder()
        X_top_numeric = top_encoder.fit_transform(X_train_top)
        del top_encoder
        gc.collect()
        _append_ownership_snapshot(
            ownership_rows,
            stage="top_k_selection_encoding_complete",
            branch=None,
            owned={
                "top_k_training_frame": X_train_top,
                "top_k_numeric_matrix": X_top_numeric,
                "top_k_validation_frame": X_validation_top,
            },
        )

        branch_results: dict[str, dict[str, Any]] = {}
        for branch in branches:
            report(f"rfe_{branch}", fold_id if mode == "fold" else None)
            branch_selection = fit_rfe_memory_safe(
                X_numeric=X_top_numeric,
                y=y_train,
                top_candidates=top_candidates,
                model_name=branch,
                seed=seed,
                estimator_threads=estimator_threads,
            )
            _append_ownership_snapshot(
                ownership_rows,
                stage="rfe_selector_released",
                branch=branch,
                owned={
                    "top_k_training_frame": X_train_top,
                    "top_k_numeric_matrix": X_top_numeric,
                },
            )
            report(f"final_model_fit_{branch}", fold_id if mode == "fold" else None)
            if mode == "fold":
                probabilities, effective_model = _fit_final_model(
                    repository_root=root,
                    dataset="lendingclub_v2",
                    model_name=branch,
                    selected_features=branch_selection["selected_features"],
                    X_train_raw=X_train_top,
                    y_train=y_train,
                    X_validation_raw=X_validation_top,
                    seed=seed,
                    estimator_threads=estimator_threads,
                )
            else:
                probabilities = None
                effective_model = _fit_full_dev_capacity_model(
                    repository_root=root,
                    dataset="lendingclub_v2",
                    model_name=branch,
                    selected_features=branch_selection["selected_features"],
                    X_train_raw=X_train_top,
                    y_train=y_train,
                    seed=seed,
                    estimator_threads=estimator_threads,
                )
            branch_results[branch] = {
                **branch_selection,
                "effective_model": effective_model,
                "probabilities": probabilities,
            }
            _append_ownership_snapshot(
                ownership_rows,
                stage="branch_model_released",
                branch=branch,
                owned={
                    "top_k_training_frame": X_train_top,
                    "top_k_numeric_matrix": X_top_numeric,
                    "top_k_validation_frame": X_validation_top,
                },
            )

        del X_top_numeric
        gc.collect()
        _append_ownership_snapshot(
            ownership_rows,
            stage="all_selector_matrices_released",
            branch=None,
            owned={
                "top_k_training_frame": X_train_top,
                "top_k_validation_frame": X_validation_top,
            },
        )

        model_artifacts = []
        prediction_artifacts = []
        for branch in branches:
            branch_dir = run_dir / "branches" / branch
            branch_dir.mkdir(parents=True, exist_ok=True)
            values = branch_results[branch]
            trace = values["rfe_trace"].copy()
            trace.insert(0, "model", branch)
            trace.insert(0, "scenario_id", scenario_id)
            trace_meta = write_csv_atomic(
                branch_dir / "rfe_selection_trace.csv", trace, overwrite=False
            )
            selected_frame = pd.DataFrame(
                {
                    "scenario_id": scenario_id,
                    "model": branch,
                    "feature": values["selected_features"],
                    "selection_order": range(1, len(values["selected_features"]) + 1),
                    "final_feature_budget": len(values["selected_features"]),
                }
            )
            selected_meta = write_csv_atomic(
                branch_dir / "selected_features.csv", selected_frame, overwrite=False
            )
            effective_meta = write_json_atomic(
                branch_dir / "effective_model_config.json",
                {
                    **values["effective_model"],
                    "rfe_effective_estimator_configuration": values[
                        "rfe_effective_config"
                    ],
                    "candidate_pool_budget": candidate_pool_budget,
                    "selector_seed": seed,
                    "protocol_sha256": protocol_sha256,
                    "checkpoint_resolved_config_hash": checkpoint_identity[
                        "resolved_config_hash"
                    ],
                    "refinement_version": "lendingclub_memory_safe_refinement_v1",
                    "research_eligible": False,
                    "comparison_eligible": False,
                },
                overwrite=False,
            )
            model_artifacts.extend((trace_meta, selected_meta, effective_meta))
            additional.update(
                {
                    f"{branch}_rfe_selection_trace": f"branches/{branch}/rfe_selection_trace.csv",
                    f"{branch}_selected_features": f"branches/{branch}/selected_features.csv",
                    f"{branch}_effective_model_config": f"branches/{branch}/effective_model_config.json",
                }
            )
            if mode == "fold":
                probabilities = np.asarray(values["probabilities"], dtype=float)
                prediction_frame = pd.DataFrame(
                    {
                        "stable_row_id": validation_ids.astype(str),
                        "target": y_validation.astype("int8"),
                        "prediction_probability": probabilities,
                        "predicted_class": (probabilities >= 0.5).astype("int8"),
                        "fold_id": int(fold_id),
                        "split": "DEV",
                        "row_position_or_order_key": range(1, len(validation_ids) + 1),
                        "dataset": "lendingclub_v2",
                        "run_id": scenario_id,
                        "method": "rank_voting_v1",
                        "model": branch,
                        "seed": seed,
                        "coverage_type": CAPACITY_SINGLE_FOLD_COVERAGE,
                        "research_eligible": False,
                        "comparison_eligible": False,
                        "probability_orientation": PROBABILITY_ORIENTATION,
                    }
                )
                prediction_meta, sidecar_meta, _ = publish_prediction_artifact(
                    path=branch_dir / "predictions_dev.csv",
                    metadata_path=branch_dir / "prediction_metadata.json",
                    frame=prediction_frame,
                    expected_identities=validation_ids,
                    expected_targets=y_validation,
                    coverage_type=CAPACITY_SINGLE_FOLD_COVERAGE,
                    expected_split="DEV",
                    research_eligible=False,
                    comparison_eligible=False,
                    context={
                        "scenario_id": scenario_id,
                        "dataset": "lendingclub_v2",
                        "method": "rank_voting_v1",
                        "model": branch,
                        "seed": seed,
                        "protocol_hash": protocol_sha256,
                        "configuration_hash": checkpoint_identity[
                            "resolved_config_hash"
                        ],
                        "split": "DEV",
                        "purpose": "memory_capacity_validation",
                    },
                )
                prediction_artifacts.extend((prediction_meta, sidecar_meta))
                additional.update(
                    {
                        f"{branch}_predictions_dev": f"branches/{branch}/predictions_dev.csv",
                        f"{branch}_prediction_metadata": f"branches/{branch}/prediction_metadata.json",
                    }
                )
        checkpoint.transition("model_fit_completed", artifacts=model_artifacts)
        if prediction_artifacts:
            checkpoint.transition(
                "dev_prediction_completed", artifacts=prediction_artifacts
            )

        del branch_results, X_train_top, X_validation_top
        gc.collect()
        _append_ownership_snapshot(
            ownership_rows,
            stage="worker_material_released",
            branch=None,
            owned={},
        )
        report("capacity_validation_publication", fold_id if mode == "fold" else None)
        validation_payload = {
            "schema_version": "lendingclub_memory_capacity_validation_v1",
            "status": "passed",
            "scenario_id": scenario_id,
            "purpose": "memory_capacity_validation",
            "research_eligible": False,
            "comparison_eligible": False,
            "load_oot": False,
            "oot_scored": False,
            "opened_oot_paths": [],
            "retained_oot_rows": 0,
            "dataset": "lendingclub_v2",
            "mode": mode,
            "fold_id": fold_id,
            "training_rows": len(training_ids),
            "validation_rows": len(validation_ids),
            "candidate_universe_count": 675,
            "candidate_pool_count": candidate_pool_budget,
            "branches": branches,
            "final_feature_budgets": {"lr": 20, "catboost": 40},
            "seed": seed,
            "selection_checkpoint_reused": selection_reused,
            "numeric_precision_changed": False,
            "gpu_training": False,
            "api_calls": 0,
            "clip_embedding_shap_workloads": 0,
        }
        validation_meta = write_json_atomic(
            run_dir / "capacity_validation.json", validation_payload, overwrite=False
        )
        timings[-1]["elapsed_seconds"] = time.perf_counter() - stage_started
        timing_meta = write_csv_atomic(
            run_dir / "stage_timings.csv", pd.DataFrame(timings), overwrite=False
        )
        ownership_meta = write_csv_atomic(
            run_dir / "memory_ownership_trace.csv",
            pd.DataFrame(ownership_rows),
            overwrite=False,
        )
        final_stage = "dev_prediction_completed" if mode == "fold" else "model_fit_completed"
        checkpoint.transition(
            final_stage,
            artifacts=(validation_meta, timing_meta, ownership_meta),
        )
        additional.update(
            {
                "fold_identity_manifest": "fold_identity_manifest.json",
                "data_access_log": "data_access_log.json",
                "capacity_validation": "capacity_validation.json",
                "stage_timings": "stage_timings.csv",
                "memory_ownership_trace": "memory_ownership_trace.csv",
            }
        )

    for relative in additional.values():
        inspect_artifact(run_dir / relative)
    return {
        "summary": {
            "scenario_id": scenario_id,
            "purpose": "memory_capacity_validation",
            "research_eligible": False,
            "comparison_eligible": False,
            "load_oot": False,
            "oot_scored": False,
            "mode": mode,
            "fold_id": fold_id,
            "training_rows": len(training_ids),
            "validation_rows": len(validation_ids),
            "candidate_universe_count": 675,
            "candidate_pool_count": candidate_pool_budget,
            "branches": branches,
            "selection_checkpoint_reused": selection_reused,
        },
        "additional_artifacts": additional,
    }


__all__ = [
    "ELIGIBLE_VOTERS",
    "PROTOCOL_NAME",
    "REQUIRED_FIT_SCOPE",
    "aggregate_cross_dataset_rank_voting",
    "build_long_voter_ranking_frame",
    "fit_fold_local_voting_adapter",
    "fit_rfe_memory_safe",
    "fit_voters_sequentially_memory_safe",
    "canonical_fold_projection",
    "lendingclub_memory_capacity_worker",
    "voting_pilot_worker",
]
