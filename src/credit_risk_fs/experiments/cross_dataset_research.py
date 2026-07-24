"""Canonical phase worker for the frozen cross-dataset voting matrix.

The parent orchestration owns the global DEV barrier.  This worker owns one
registered run and never loads OOT while ``phase == 'dev'``.
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

from credit_risk_fs.evaluation.metrics import determine_threshold, evaluate_model
from credit_risk_fs.experiments.atomic_io import (
    inspect_artifact,
    write_csv_atomic,
    write_json_atomic,
)
from credit_risk_fs.experiments.checkpointing import CheckpointManager
from credit_risk_fs.experiments.prediction_contract import (
    COMPLETE_OOF_COVERAGE,
    COMPLETE_OOT_COVERAGE,
    PROBABILITY_ORIENTATION,
    publish_prediction_artifact,
)
from credit_risk_fs.experiments.rank_voting import (
    ELIGIBLE_VOTERS,
    PROTOCOL_NAME,
    REQUIRED_FIT_SCOPE,
    _fit_final_model,
    aggregate_cross_dataset_rank_voting,
    build_long_voter_ranking_frame,
    canonical_fold_projection,
    fit_rfe_memory_safe,
    fit_voters_sequentially_memory_safe,
)
from credit_risk_fs.experiments.row_alignment import (
    ordered_row_id_sha256,
    ordered_row_id_target_sha256,
)
from credit_risk_fs.pipelines.common import (
    prepare_voting_pilot_dev_data,
    prepare_voting_research_oot_data,
)
from credit_risk_fs.preprocessing.encoding import OriginalFeatureNumericEncoder, Preprocessor
from credit_risk_fs.utils.logging import run_log_context


def _report(
    stop_event: Any,
    stage_queue: Any,
    stage: str,
    fold_id: Any = None,
    **fields: Any,
) -> None:
    if stop_event.is_set():
        raise RuntimeError(f"cooperative stop requested before stage {stage}")
    stage_queue.put({"stage": stage, "fold_id": fold_id, **fields})


def _reference_selection(
    X_numeric: pd.DataFrame,
    y: pd.Series,
    *,
    budget: int,
    seed: int,
    estimator_threads: int,
) -> dict[str, Any]:
    from credit_risk_fs.selectors.mrmr import RandomForestRelevanceMRMRSelector

    selector = RandomForestRelevanceMRMRSelector(
        k=budget, method="mrmr", random_state=seed, n_jobs=estimator_threads
    )
    selector.fit(X_numeric, y)
    selected = list(selector.selected_features_ or [])
    if len(selected) != budget or len(set(selected)) != budget:
        raise ValueError("reference selector did not produce the exact final budget")
    importances = getattr(selector, "rf_importances_", pd.Series(dtype=float))
    ranking = selected
    rows = []
    for position, feature in enumerate(X_numeric.columns, start=1):
        rank = ranking.index(feature) + 1 if feature in ranking else pd.NA
        rows.append(
            {
                "dataset": "",
                "fold_id": 0,
                "voter_id": "rf_corr_mrmr",
                "original_feature_name": str(feature),
                "normalized_feature_name": str(feature),
                "raw_rank": rank,
                "raw_score_if_available": (
                    float(importances.get(feature)) if feature in importances.index else pd.NA
                ),
                "normalized_score": (
                    1.0 - (int(rank) - 1) / max(len(X_numeric.columns) - 1, 1)
                    if not pd.isna(rank)
                    else 0.0
                ),
                "present": not pd.isna(rank),
                "fit_scope": REQUIRED_FIT_SCOPE,
                "seed": seed,
            }
        )
    aggregate = pd.DataFrame(
        {
            "feature": list(X_numeric.columns),
            "aggregate_score": [
                float(importances.get(feature, 0.0)) for feature in X_numeric.columns
            ],
        }
    )
    order = {feature: index for index, feature in enumerate(ranking)}
    aggregate["__rank"] = aggregate["feature"].map(order).fillna(len(aggregate))
    aggregate = aggregate.sort_values(
        ["__rank", "feature"], kind="mergesort"
    ).drop(columns="__rank").reset_index(drop=True)
    aggregate["aggregate_rank"] = np.arange(1, len(aggregate) + 1)
    aggregate["presence_count"] = aggregate["feature"].isin(ranking).astype(int)
    aggregate["best_individual_rank"] = aggregate["feature"].map(
        {value: index + 1 for index, value in enumerate(ranking)}
    )
    return {
        "voter_rankings": pd.DataFrame(rows),
        "aggregate_ranking": aggregate,
        "candidate_features": selected,
        "selected_features": selected,
        "rfe_trace": pd.DataFrame(
            {
                "feature": selected,
                "selected": True,
                "selection_rank": range(1, budget + 1),
            }
        ),
        "rfe_effective_config": None,
    }


def _select_on_boundary(
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    training_ids: pd.Series,
    dataset: str,
    model_name: str,
    method_id: str,
    candidate_pool_budget: int | None,
    final_feature_budget: int,
    seed: int,
    estimator_threads: int,
    protocol_sha256: str,
    input_artifact_hash: str,
    fold_id: int,
    stop_event: Any,
    stage_queue: Any,
) -> dict[str, Any]:
    candidates = list(map(str, X_train.columns))
    _report(
        stop_event,
        stage_queue,
        "selection_encoding",
        fold_id,
        input_row_count=len(X_train),
        input_feature_count=X_train.shape[1],
        component="original_feature_numeric_encoder",
    )
    encoder = OriginalFeatureNumericEncoder()
    X_numeric = encoder.fit_transform(X_train)
    del encoder
    if method_id == "rf_corr_mrmr":
        _report(stop_event, stage_queue, "reference_rf_corr_mrmr", fold_id)
        result = _reference_selection(
            X_numeric,
            y_train,
            budget=final_feature_budget,
            seed=seed,
            estimator_threads=estimator_threads,
        )
        result["voter_rankings"]["dataset"] = dataset
        result["voter_rankings"]["fold_id"] = fold_id
        result["aggregate_ranking"].insert(0, "fold_id", fold_id)
        result["aggregate_ranking"].insert(0, "dataset", dataset)
        del X_numeric
        gc.collect()
        return result

    if candidate_pool_budget not in {100, 200, 300}:
        raise ValueError("voting run requires frozen K=100, K=200, or K=300")
    voter_result = fit_voters_sequentially_memory_safe(
        X_numeric=X_numeric,
        y=y_train,
        seed=seed,
        estimator_threads=estimator_threads,
        stage_callback=lambda stage, current: _report(
            stop_event, stage_queue, stage, current
        ),
        fold_id=fold_id,
    )
    long_frame = build_long_voter_ranking_frame(
        dataset=dataset,
        fold_id=fold_id,
        eligible_features=candidates,
        rankings=voter_result["rankings"],
        raw_scores=voter_result["raw_scores"],
        selector_configurations=voter_result["selector_configurations"],
        seed=seed,
        training_row_identity_sha256=ordered_row_id_sha256(
            training_ids.astype(str).tolist()
        ),
        training_identity_target_sha256=ordered_row_id_target_sha256(
            training_ids.astype(str).tolist(), y_train.tolist()
        ),
        input_artifact_hash=input_artifact_hash,
        protocol_sha256=protocol_sha256,
        fit_scope="full_dev_only" if fold_id == 0 else REQUIRED_FIT_SCOPE,
    )
    _report(
        stop_event,
        stage_queue,
        "rank_aggregation",
        fold_id,
        input_feature_count=len(candidates),
        voter_count=len(ELIGIBLE_VOTERS),
        component="rank_voting",
    )
    aggregate = aggregate_cross_dataset_rank_voting(
        eligible_features=candidates,
        rankings=voter_result["rankings"],
        fit_scopes={
            voter: "full_dev_only" if fold_id == 0 else REQUIRED_FIT_SCOPE
            for voter in ELIGIBLE_VOTERS
        },
    )
    aggregate.insert(0, "fold_id", fold_id)
    aggregate.insert(0, "dataset", dataset)
    aggregate["protocol_version"] = PROTOCOL_NAME
    aggregate["candidate_pool_membership"] = aggregate["aggregate_rank"].le(
        candidate_pool_budget
    )
    top = aggregate.head(candidate_pool_budget)["feature"].astype(str).tolist()
    del X_numeric, voter_result
    gc.collect()
    _report(
        stop_event,
        stage_queue,
        "rfe_encoding",
        fold_id,
        input_row_count=len(X_train),
        input_feature_count=len(top),
        component="original_feature_numeric_encoder",
    )
    top_encoder = OriginalFeatureNumericEncoder()
    X_top_numeric = top_encoder.fit_transform(X_train.loc[:, top])
    _report(
        stop_event,
        stage_queue,
        "rfe",
        fold_id,
        input_row_count=len(X_top_numeric),
        input_feature_count=X_top_numeric.shape[1],
        final_feature_budget=final_feature_budget,
        component=f"{model_name}_rfe",
    )
    rfe = fit_rfe_memory_safe(
        X_numeric=X_top_numeric,
        y=y_train,
        top_candidates=top,
        model_name=model_name,
        seed=seed,
        estimator_threads=estimator_threads,
    )
    del X_top_numeric, top_encoder
    gc.collect()
    return {
        "voter_rankings": long_frame,
        "aggregate_ranking": aggregate,
        "candidate_features": top,
        **rfe,
    }


def _input_hash(source_hashes: Mapping[str, str]) -> str:
    import hashlib

    return hashlib.sha256(
        json.dumps(source_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _write_fold_artifacts(
    run_dir: Path,
    fold_id: int,
    selection: Mapping[str, Any],
    probabilities: np.ndarray,
    validation_ids: pd.Series,
    y_validation: pd.Series,
    *,
    dataset: str,
    run_id: str,
    method_id: str,
    model_name: str,
    seed: int,
    effective_model: Mapping[str, Any],
    protocol_sha256: str,
    configuration_hash: str,
) -> tuple[list[Any], pd.DataFrame]:
    fold_dir = run_dir / "folds" / f"fold_{fold_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    artifacts = [
        write_csv_atomic(fold_dir / "voter_rankings.csv", selection["voter_rankings"], overwrite=False),
        write_csv_atomic(fold_dir / "aggregate_ranking.csv", selection["aggregate_ranking"], overwrite=False),
        write_csv_atomic(
            fold_dir / "candidate_features.csv",
            pd.DataFrame(
                {
                    "feature": selection["candidate_features"],
                    "candidate_rank": range(1, len(selection["candidate_features"]) + 1),
                }
            ),
            overwrite=False,
        ),
        write_csv_atomic(fold_dir / "rfe_selection_trace.csv", selection["rfe_trace"], overwrite=False),
        write_csv_atomic(
            fold_dir / "selected_features.csv",
            pd.DataFrame(
                {
                    "feature": selection["selected_features"],
                    "selection_rank": range(1, len(selection["selected_features"]) + 1),
                    "fold_id": fold_id,
                }
            ),
            overwrite=False,
        ),
        write_json_atomic(
            fold_dir / "effective_model_config.json",
            {
                **dict(effective_model),
                "rfe_effective_estimator_configuration": selection.get(
                    "rfe_effective_config"
                ),
            },
            overwrite=False,
        ),
    ]
    frame = pd.DataFrame(
        {
            "stable_row_id": validation_ids.astype(str),
            "target": y_validation.astype("int8"),
            "prediction_probability": probabilities,
            "predicted_class": (probabilities >= 0.5).astype("int8"),
            "fold_id": fold_id,
            "split": "DEV",
            "row_position_or_order_key": range(1, len(validation_ids) + 1),
            "dataset": dataset,
            "run_id": run_id,
            "method": method_id,
            "model": model_name,
            "seed": seed,
            "coverage_type": COMPLETE_OOF_COVERAGE,
            "research_eligible": True,
            "comparison_eligible": True,
            "probability_orientation": PROBABILITY_ORIENTATION,
        }
    )
    prediction, sidecar, _ = publish_prediction_artifact(
        path=fold_dir / "predictions_dev.csv",
        metadata_path=fold_dir / "prediction_metadata.json",
        frame=frame.assign(
            coverage_type="single_dev_fold_pilot",
            research_eligible=False,
            comparison_eligible=False,
        ),
        expected_identities=validation_ids,
        expected_targets=y_validation,
        coverage_type="single_dev_fold_pilot",
        expected_split="DEV",
        research_eligible=False,
        comparison_eligible=False,
        context={
            "run_id": run_id,
            "dataset": dataset,
            "model": model_name,
            "method": method_id,
            "protocol_hash": protocol_sha256,
            "configuration_hash": configuration_hash,
            "split": "DEV_FOLD",
        },
    )
    artifacts.extend((prediction, sidecar))
    return artifacts, frame


def _run_dev_phase(
    *,
    stop_event: Any,
    stage_queue: Any,
    root: Path,
    run_dir: Path,
    checkpoint: CheckpointManager,
    checkpoint_identity: Mapping[str, Any],
    spec: Mapping[str, Any],
    protocol_sha256: str,
    estimator_threads: int,
) -> dict[str, Any]:
    completed_folds = set(checkpoint.load().get("completed_fold_ids", []))
    for fold_id in range(1, 6):
        if str(fold_id) in completed_folds:
            continue
        _report(
            stop_event,
            stage_queue,
            "dev_data_loading",
            fold_id,
            component="prepare_voting_pilot_dev_data",
        )
        prepared = prepare_voting_pilot_dev_data(
            root,
            dataset=str(spec["dataset"]),
            csv_chunk_rows=25_000,
            csv_low_memory=False,
        )
        _report(
            stop_event,
            stage_queue,
            "target_extraction",
            fold_id,
            component="validated_target_projection",
            target_row_count=len(prepared.y),
            target_class_count=int(prepared.y.nunique()),
        )
        _report(
            stop_event,
            stage_queue,
            "feature_filtering_sanitization",
            fold_id,
            component="candidate_feature_contract",
            candidate_feature_count=prepared.X.shape[1],
        )
        if fold_id == 1 and "data_validated" not in checkpoint.load()["completed_stages"]:
            data_meta = write_json_atomic(
                run_dir / "data_access_dev.json",
                {
                    "split": "DEV",
                    "opened_oot_paths": [],
                    "retained_oot_rows": 0,
                    "split_evidence": prepared.split_evidence,
                    "data_access_log": prepared.data_access_log,
                    "source_artifact_hashes": prepared.source_artifact_hashes,
                },
                overwrite=False,
            )
            checkpoint.transition("data_validated", artifacts=(data_meta,))
        projection = canonical_fold_projection(
            y=prepared.y,
            stable_row_ids=prepared.stable_row_ids,
            time_values=prepared.time_values,
            fold_id=fold_id,
        )
        tr = projection["training_indices"]
        va = projection["validation_indices"]
        positions = projection["source_positions"]
        _report(
            stop_event,
            stage_queue,
            "row_boundary_selection",
            fold_id,
            total_dev_row_count=len(projection["y"]),
            training_row_count=len(tr),
            validation_row_count=len(va),
            input_feature_count=prepared.X.shape[1],
            component="canonical_fold_projection",
        )
        X_train = prepared.X.iloc[positions[tr]].reset_index(drop=True)
        y_train = projection["y"].iloc[tr].reset_index(drop=True)
        selection = _select_on_boundary(
            X_train=X_train,
            y_train=y_train,
            training_ids=projection["ids"].iloc[tr].reset_index(drop=True),
            dataset=str(spec["dataset"]),
            model_name=str(spec["model"]),
            method_id=str(spec["method_id"]),
            candidate_pool_budget=spec.get("candidate_pool_budget"),
            final_feature_budget=int(spec["final_feature_budget"]),
            seed=42,
            estimator_threads=estimator_threads,
            protocol_sha256=protocol_sha256,
            input_artifact_hash=_input_hash(prepared.source_artifact_hashes),
            fold_id=fold_id,
            stop_event=stop_event,
            stage_queue=stage_queue,
        )
        selected = list(selection["selected_features"])
        del X_train, prepared
        gc.collect()
        _report(
            stop_event,
            stage_queue,
            "selected_projection_reload",
            fold_id,
            selected_feature_count=len(selected),
            component="prepare_voting_pilot_dev_data",
        )
        projected = prepare_voting_pilot_dev_data(
            root,
            dataset=str(spec["dataset"]),
            projected_candidate_features=selected,
            csv_chunk_rows=25_000,
            csv_low_memory=False,
        )
        projected_fold = canonical_fold_projection(
            y=projected.y,
            stable_row_ids=projected.stable_row_ids,
            time_values=projected.time_values,
            fold_id=fold_id,
        )
        ptr = projected_fold["training_indices"]
        pva = projected_fold["validation_indices"]
        ppos = projected_fold["source_positions"]
        probabilities, effective = _fit_final_model(
            repository_root=root,
            dataset=str(spec["dataset"]),
            model_name=str(spec["model"]),
            selected_features=selected,
            X_train_raw=projected.X.iloc[ppos[ptr]].reset_index(drop=True),
            y_train=projected_fold["y"].iloc[ptr].reset_index(drop=True),
            X_validation_raw=projected.X.iloc[ppos[pva]].reset_index(drop=True),
            seed=42,
            estimator_threads=estimator_threads,
            stage_callback=lambda stage, current, **details: _report(
                stop_event, stage_queue, stage, current, **details
            ),
            fold_id=fold_id,
        )
        _report(
            stop_event,
            stage_queue,
            "fold_artifact_writing",
            fold_id,
            component="atomic_artifact_writer",
        )
        artifacts, _ = _write_fold_artifacts(
            run_dir,
            fold_id,
            selection,
            probabilities,
            projected_fold["ids"].iloc[pva].reset_index(drop=True),
            projected_fold["y"].iloc[pva].reset_index(drop=True),
            dataset=str(spec["dataset"]),
            run_id=str(spec["run_id"]),
            method_id=str(spec["method_id"]),
            model_name=str(spec["model"]),
            seed=42,
            effective_model=effective,
            protocol_sha256=protocol_sha256,
            configuration_hash=str(checkpoint_identity["resolved_config_hash"]),
        )
        _report(
            stop_event,
            stage_queue,
            "fold_checkpoint_finalization",
            fold_id,
            artifact_count=len(artifacts),
            component="checkpoint_manager",
        )
        checkpoint.transition("selection_completed", artifacts=artifacts[:5])
        checkpoint.transition("model_fit_completed", artifacts=(artifacts[5],))
        checkpoint.transition(
            "dev_prediction_completed", artifacts=artifacts[6:], completed_fold_id=fold_id
        )
        del projected, selection, probabilities
        gc.collect()

    _report(
        stop_event,
        stage_queue,
        "dev_oof_aggregation",
        None,
        fold_count=5,
        component="prediction_contract",
    )
    fold_predictions = [
        pd.read_csv(run_dir / "folds" / f"fold_{fold_id}" / "predictions_dev.csv")
        for fold_id in range(1, 6)
    ]
    complete = pd.concat(fold_predictions, ignore_index=True)
    complete["coverage_type"] = COMPLETE_OOF_COVERAGE
    complete["research_eligible"] = True
    complete["comparison_eligible"] = True
    _report(
        stop_event,
        stage_queue,
        "dev_artifact_writing",
        None,
        component="atomic_artifact_writer",
    )
    prediction, sidecar, metadata = publish_prediction_artifact(
        path=run_dir / "results" / "dev_predictions.csv",
        metadata_path=run_dir / "results" / "dev_prediction_metadata.json",
        frame=complete,
        expected_identities=complete["stable_row_id"],
        expected_targets=complete["target"],
        coverage_type=COMPLETE_OOF_COVERAGE,
        expected_split="DEV",
        research_eligible=True,
        comparison_eligible=True,
        context={
            "run_id": spec["run_id"],
            "dataset": spec["dataset"],
            "model": spec["model"],
            "method": spec["method_id"],
            "split": "DEV_OOF",
            "fold_definition": "grouped_time_series_cv_5_splits_gap_1_expanding",
            "protocol_hash": protocol_sha256,
            "configuration_hash": checkpoint_identity["resolved_config_hash"],
        },
    )
    fold_selected = pd.concat(
        [
            pd.read_csv(run_dir / "folds" / f"fold_{fold_id}" / "selected_features.csv")
            for fold_id in range(1, 6)
        ],
        ignore_index=True,
    )
    selected_meta = write_csv_atomic(
        run_dir / "features" / "fold_selected_features.csv",
        fold_selected,
        overwrite=False,
    )
    _report(
        stop_event,
        stage_queue,
        "dev_evaluation",
        None,
        fold_count=5,
        component="evaluation_metrics",
    )
    fold_metrics = []
    selected_sets = []
    for fold_id, frame in enumerate(fold_predictions, start=1):
        fold_metrics.append(
            {"fold_id": fold_id, **evaluate_model(frame.target, frame.prediction_probability)}
        )
        selected_sets.append(
            set(fold_selected.loc[fold_selected.fold_id.eq(fold_id), "feature"].astype(str))
        )
    metrics_meta = write_csv_atomic(
        run_dir / "results" / "dev_fold_metrics.csv",
        pd.DataFrame(fold_metrics),
        overwrite=False,
    )
    stability_meta = write_csv_atomic(
        run_dir / "features" / "feature_stability_metrics.csv",
        pd.DataFrame(
            [
                {
                    "left_fold": index,
                    "right_fold": index + 1,
                    "jaccard": len(selected_sets[index - 1] & selected_sets[index])
                    / max(1, len(selected_sets[index - 1] | selected_sets[index])),
                }
                for index in range(1, 5)
            ]
        ),
        overwrite=False,
    )
    _report(
        stop_event,
        stage_queue,
        "dev_checkpoint_finalization",
        None,
        component="checkpoint_manager",
    )
    checkpoint.transition(
        "dev_prediction_completed",
        artifacts=(prediction, sidecar, selected_meta, metrics_meta, stability_meta),
    )
    return {
        "summary": {
            "phase": "dev",
            "run_id": spec["run_id"],
            "completed_folds": 5,
            "oof_rows": metadata["row_count"],
            "oot_opened": False,
        },
        "additional_artifacts": {
            "dev_prediction_metadata": "results/dev_prediction_metadata.json",
            "dev_fold_metrics": "results/dev_fold_metrics.csv",
        },
    }


def _atomic_joblib(path: Path, payload: Any) -> Any:
    partial = path.with_name(path.name + f".{os.getpid()}.partial")
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, partial)
    os.replace(partial, path)
    return inspect_artifact(path)


def _run_oot_phase(
    *,
    stop_event: Any,
    stage_queue: Any,
    root: Path,
    run_dir: Path,
    checkpoint: CheckpointManager,
    checkpoint_identity: Mapping[str, Any],
    spec: Mapping[str, Any],
    protocol_sha256: str,
    estimator_threads: int,
) -> dict[str, Any]:
    checkpoint_payload = checkpoint.load()
    if set(checkpoint_payload.get("completed_fold_ids", [])) != {"1", "2", "3", "4", "5"}:
        raise RuntimeError("OOT barrier closed: five validated DEV folds are required")
    if not (run_dir / "results" / "dev_predictions.csv").is_file():
        raise RuntimeError("OOT barrier closed: complete DEV OOF artifact is missing")

    existing_oot_path = run_dir / "results" / "oot_predictions.csv"
    existing_metrics_path = run_dir / "results" / "prediction_metrics.csv"
    if existing_oot_path.is_file():
        # Resume after a validated OOT publication reuses it and performs only
        # the remaining deterministic evaluation/publication work.
        oot_frame = pd.read_csv(existing_oot_path)
        oof = pd.read_csv(run_dir / "results" / "dev_predictions.csv")
        metric_artifacts = []
        if not existing_metrics_path.is_file():
            _report(
                stop_event,
                stage_queue,
                "oot_evaluation",
                None,
                component="evaluation_metrics",
            )
            metrics_meta = write_csv_atomic(
                existing_metrics_path,
                pd.DataFrame(
                    [
                        {
                            "split": "DEV_OOF",
                            **evaluate_model(oof.target, oof.prediction_probability),
                        },
                        {
                            "split": "OOT",
                            **evaluate_model(
                                oot_frame.target,
                                oot_frame.prediction_probability,
                                y_pred=oot_frame.predicted_class,
                            ),
                        },
                    ]
                ),
                overwrite=False,
            )
            metric_artifacts.append(metrics_meta)
        else:
            metric_artifacts.append(inspect_artifact(existing_metrics_path))
        _report(
            stop_event,
            stage_queue,
            "oot_checkpoint_finalization",
            None,
            component="checkpoint_manager",
        )
        checkpoint.transition("evaluation_completed", artifacts=metric_artifacts)
        return {
            "summary": {
                "phase": "oot",
                "run_id": spec["run_id"],
                "oot_rows": len(oot_frame),
                "configuration_frozen_before_oot": True,
                "oot_prediction_reused_after_integrity_validation": True,
            },
            "additional_artifacts": {
                "final_voter_rankings": "features/final_voter_rankings.csv",
                "final_aggregate_ranking": "features/final_aggregate_ranking.csv",
                "final_candidate_features": "features/final_candidate_features.csv",
                "oot_prediction_metadata": "results/oot_prediction_metadata.json",
                "oot_access_log": "data_access_oot.json",
            },
        }

    final_selected_path = run_dir / "features" / "final_selected_features.csv"
    selection: Mapping[str, Any] | None
    if final_selected_path.is_file():
        selected = pd.read_csv(final_selected_path)["feature"].astype(str).tolist()
        if len(selected) != int(spec["final_feature_budget"]):
            raise ValueError("reusable final selection has the wrong frozen budget")
        selection = None
    else:
        _report(stop_event, stage_queue, "full_dev_data_loading", None)
        dev = prepare_voting_pilot_dev_data(
            root,
            dataset=str(spec["dataset"]),
            csv_chunk_rows=25_000,
            csv_low_memory=False,
        )
        _report(
            stop_event,
            stage_queue,
            "full_dev_target_extraction",
            None,
            component="validated_target_projection",
            target_row_count=len(dev.y),
            target_class_count=int(dev.y.nunique()),
        )
        _report(
            stop_event,
            stage_queue,
            "full_dev_feature_filtering_sanitization",
            None,
            component="candidate_feature_contract",
            candidate_feature_count=dev.X.shape[1],
        )
        selection = _select_on_boundary(
            X_train=dev.X.reset_index(drop=True),
            y_train=dev.y.reset_index(drop=True),
            training_ids=dev.stable_row_ids.reset_index(drop=True),
            dataset=str(spec["dataset"]),
            model_name=str(spec["model"]),
            method_id=str(spec["method_id"]),
            candidate_pool_budget=spec.get("candidate_pool_budget"),
            final_feature_budget=int(spec["final_feature_budget"]),
            seed=42,
            estimator_threads=estimator_threads,
            protocol_sha256=protocol_sha256,
            input_artifact_hash=_input_hash(dev.source_artifact_hashes),
            fold_id=0,
            stop_event=stop_event,
            stage_queue=stage_queue,
        )
        selected = list(selection["selected_features"])
        del dev
        gc.collect()
    _report(stop_event, stage_queue, "full_dev_selected_projection_reload", None)
    dev_selected = prepare_voting_pilot_dev_data(
        root,
        dataset=str(spec["dataset"]),
        projected_candidate_features=selected,
        csv_chunk_rows=25_000,
        csv_low_memory=False,
    )
    _report(
        stop_event,
        stage_queue,
        "full_dev_selected_feature_validation",
        None,
        component="candidate_feature_contract",
        selected_feature_count=dev_selected.X.shape[1],
        target_row_count=len(dev_selected.y),
    )
    dev_source_hashes = dict(dev_selected.source_artifact_hashes)

    # This is the first and only OOT loader call in a run, after the global gate.
    _report(stop_event, stage_queue, "locked_oot_data_loading", None)
    oot = prepare_voting_research_oot_data(
        root,
        dataset=str(spec["dataset"]),
        projected_candidate_features=selected,
        csv_chunk_rows=25_000,
        csv_low_memory=False,
    )
    _report(
        stop_event,
        stage_queue,
        "oot_target_extraction",
        None,
        component="validated_target_projection",
        target_row_count=len(oot.y),
        target_class_count=int(oot.y.nunique()),
    )
    _report(
        stop_event,
        stage_queue,
        "oot_feature_filtering_sanitization",
        None,
        component="candidate_feature_contract",
        selected_feature_count=oot.X.shape[1],
    )
    if _input_hash(oot.source_artifact_hashes) != _input_hash(dev_source_hashes):
        raise ValueError("locked OOT projection source provenance changed")

    import yaml
    from credit_risk_fs.models.registry import get_model_bundle

    base = yaml.safe_load((root / "configs/base.yaml").read_text(encoding="utf-8"))
    model_kwargs = dict(base["model_params"][str(spec["model"])])
    model_kwargs["random_state"] = 42
    if spec["model"] == "catboost":
        model_kwargs["thread_count"] = estimator_threads
    dataset_config = yaml.safe_load(
        (root / f"configs/experiments/{spec['dataset']}_matrix.yaml").read_text(
            encoding="utf-8"
        )
    )
    model_bundle_path = run_dir / "models" / "final_model_bundle.joblib"
    effective_path = run_dir / "models" / "final_model_metadata.json"
    model_artifacts = []
    if model_bundle_path.is_file() and effective_path.is_file():
        bundle = joblib.load(model_bundle_path)
        if list(bundle.get("selected_features", [])) != selected or bundle.get(
            "configuration_hash"
        ) != checkpoint_identity["resolved_config_hash"]:
            raise ValueError("reusable full-DEV model provenance differs")
        model = bundle["model"]
        preprocessor = bundle["preprocessor"]
        _report(
            stop_event,
            stage_queue,
            "full_dev_preprocessing",
            None,
            component="preprocessor",
            selected_feature_count=len(selected),
        )
        X_dev = preprocessor.transform(dev_selected.X.loc[:, selected])
        X_oot = preprocessor.transform(oot.X.loc[:, selected])
        model_artifacts.extend(
            (inspect_artifact(model_bundle_path), inspect_artifact(effective_path))
        )
    else:
        preprocessor = Preprocessor(**dict(dataset_config.get("preprocessor_kwargs", {})))
        _report(
            stop_event,
            stage_queue,
            "full_dev_preprocessing",
            None,
            component="preprocessor",
            selected_feature_count=len(selected),
        )
        X_dev = preprocessor.fit_transform(dev_selected.X.loc[:, selected])
        X_oot = preprocessor.transform(oot.X.loc[:, selected])
        get_model, _, _, _ = get_model_bundle(str(spec["model"]), model_kwargs)
        model = get_model()
        _report(stop_event, stage_queue, "full_dev_model_fit", None)
        model.fit(X_dev, dev_selected.y, eval_set=None)
    _report(
        stop_event,
        stage_queue,
        "full_dev_prediction",
        None,
        component=str(spec["model"]),
        prediction_row_count=len(X_dev),
    )
    dev_probabilities = np.asarray(model.predict_proba(X_dev), dtype=float)
    _report(
        stop_event,
        stage_queue,
        "oot_prediction",
        None,
        component=str(spec["model"]),
        prediction_row_count=len(X_oot),
    )
    oot_probabilities = np.asarray(model.predict_proba(X_oot), dtype=float)
    if not np.isfinite(dev_probabilities).all() or not np.isfinite(oot_probabilities).all():
        raise ValueError("full-DEV model produced non-finite probabilities")
    classes = list(map(int, model.model.classes_))
    if classes != [0, 1]:
        raise ValueError("full-DEV model probability orientation is not class 1")
    if not model_artifacts:
        model_meta = _atomic_joblib(
            model_bundle_path,
            {
                "model": model,
                "preprocessor": preprocessor,
                "selected_features": selected,
                "configuration_hash": checkpoint_identity["resolved_config_hash"],
            },
        )
        effective_meta = write_json_atomic(
            effective_path,
            {
                "model": spec["model"],
                "method": spec["method_id"],
                "selected_features": selected,
                "feature_budget": len(selected),
                "training_scope": "full_DEV",
                "oot_used_for_fit": False,
                "requested_model_configuration": model_kwargs,
                "actual_estimator_configuration": model.model.get_params(),
                "probability_classes": classes,
                "probability_orientation": PROBABILITY_ORIENTATION,
                "estimator_threads_maximum": estimator_threads,
            },
            overwrite=False,
        )
        model_artifacts.extend((model_meta, effective_meta))
    threshold = determine_threshold(dev_selected.y, dev_probabilities)
    oot_frame = pd.DataFrame(
        {
            "stable_row_id": oot.stable_row_ids.astype(str),
            "target": oot.y.astype("int8"),
            "prediction_probability": oot_probabilities,
            "predicted_class": (oot_probabilities >= threshold).astype("int8"),
            "fold_id": "final",
            "split": "OOT",
            "row_position_or_order_key": range(1, len(oot.y) + 1),
            "dataset": spec["dataset"],
            "run_id": spec["run_id"],
            "method": spec["method_id"],
            "model": spec["model"],
            "seed": 42,
            "coverage_type": COMPLETE_OOT_COVERAGE,
            "research_eligible": True,
            "comparison_eligible": True,
            "probability_orientation": PROBABILITY_ORIENTATION,
        }
    )
    _report(
        stop_event,
        stage_queue,
        "oot_artifact_writing",
        None,
        component="atomic_artifact_writer",
    )
    prediction, sidecar, prediction_metadata = publish_prediction_artifact(
        path=run_dir / "results" / "oot_predictions.csv",
        metadata_path=run_dir / "results" / "oot_prediction_metadata.json",
        frame=oot_frame,
        expected_identities=oot.stable_row_ids,
        expected_targets=oot.y,
        coverage_type=COMPLETE_OOT_COVERAGE,
        expected_split="OOT",
        research_eligible=True,
        comparison_eligible=True,
        context={
            "run_id": spec["run_id"],
            "dataset": spec["dataset"],
            "model": spec["model"],
            "method": spec["method_id"],
            "split": "OOT",
            "fold_definition": "locked_single_final_evaluation",
            "protocol_hash": protocol_sha256,
            "configuration_hash": checkpoint_identity["resolved_config_hash"],
        },
    )
    selection_artifacts = []
    if selection is None:
        for relative in (
            "features/final_selected_features.csv",
            "features/final_voter_rankings.csv",
            "features/final_aggregate_ranking.csv",
            "features/final_candidate_features.csv",
        ):
            selection_artifacts.append(inspect_artifact(run_dir / relative))
    else:
        final_selected_meta = write_csv_atomic(
            final_selected_path,
            pd.DataFrame(
                {
                    "feature": selected,
                    "selection_rank": range(1, len(selected) + 1),
                    "scope": "full_dev",
                }
            ),
            overwrite=False,
        )
        final_voter_meta = write_csv_atomic(
            run_dir / "features" / "final_voter_rankings.csv",
            selection["voter_rankings"],
            overwrite=False,
        )
        final_aggregate_meta = write_csv_atomic(
            run_dir / "features" / "final_aggregate_ranking.csv",
            selection["aggregate_ranking"],
            overwrite=False,
        )
        final_candidate_meta = write_csv_atomic(
            run_dir / "features" / "final_candidate_features.csv",
            pd.DataFrame(
                {
                    "feature": selection["candidate_features"],
                    "candidate_rank": range(
                        1, len(selection["candidate_features"]) + 1
                    ),
                }
            ),
            overwrite=False,
        )
        selection_artifacts.extend(
            (
                final_selected_meta,
                final_voter_meta,
                final_aggregate_meta,
                final_candidate_meta,
            )
        )
    _report(
        stop_event,
        stage_queue,
        "oot_evaluation",
        None,
        component="evaluation_metrics",
    )
    oof = pd.read_csv(run_dir / "results" / "dev_predictions.csv")
    metrics = pd.DataFrame(
        [
            {"split": "DEV_OOF", **evaluate_model(oof.target, oof.prediction_probability)},
            {
                "split": "OOT",
                **evaluate_model(oot.y, oot_probabilities, threshold=threshold),
            },
        ]
    )
    metrics_meta = write_csv_atomic(
        run_dir / "results" / "prediction_metrics.csv", metrics, overwrite=False
    )
    access_meta = write_json_atomic(
        run_dir / "data_access_oot.json",
        {
            "split": "OOT",
            "configuration_frozen_before_open": True,
            "projected_features": selected,
            "split_evidence": oot.split_evidence,
            "data_access_log": oot.data_access_log,
        },
        overwrite=False,
    )
    _report(
        stop_event,
        stage_queue,
        "oot_checkpoint_finalization",
        None,
        component="checkpoint_manager",
    )
    checkpoint.transition(
        "selection_completed",
        artifacts=selection_artifacts,
    )
    checkpoint.transition("model_fit_completed", artifacts=model_artifacts)
    checkpoint.transition("oot_prediction_completed", artifacts=(prediction, sidecar, access_meta))
    checkpoint.transition("evaluation_completed", artifacts=(metrics_meta,))
    return {
        "summary": {
            "phase": "oot",
            "run_id": spec["run_id"],
            "oot_rows": prediction_metadata["row_count"],
            "configuration_frozen_before_oot": True,
        },
        "additional_artifacts": {
            "final_voter_rankings": "features/final_voter_rankings.csv",
            "final_aggregate_ranking": "features/final_aggregate_ranking.csv",
            "final_candidate_features": "features/final_candidate_features.csv",
            "oot_prediction_metadata": "results/oot_prediction_metadata.json",
            "oot_access_log": "data_access_oot.json",
        },
    }


def cross_dataset_research_phase_worker(
    *,
    stop_event: Any,
    stage_queue: Any,
    checkpoint_identity: Mapping[str, Any],
    run_directory: str,
    repository_root: str,
    phase: str,
    spec: Mapping[str, Any],
    protocol_sha256: str,
    estimator_threads: int,
    frozen_set_sha256: str | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Execute one resumable phase; OOT is unreachable from the DEV branch."""

    if phase not in {"dev", "oot"}:
        raise ValueError("research phase must be dev or oot")
    if estimator_threads < 1 or estimator_threads > 4:
        raise ValueError("research estimator thread limit must remain within [1, 4]")
    root = Path(repository_root).resolve()
    run_dir = Path(run_directory).resolve()
    checkpoint = CheckpointManager(run_dir)
    with run_log_context(run_dir / "run.log"):
        if phase == "dev":
            if frozen_set_sha256 is not None:
                raise ValueError("DEV phase must not receive an OOT configuration lock")
            return _run_dev_phase(
                stop_event=stop_event,
                stage_queue=stage_queue,
                root=root,
                run_dir=run_dir,
                checkpoint=checkpoint,
                checkpoint_identity=checkpoint_identity,
                spec=spec,
                protocol_sha256=protocol_sha256,
                estimator_threads=estimator_threads,
            )
        if not frozen_set_sha256:
            raise RuntimeError("OOT phase requires the frozen validated configuration-set hash")
        return _run_oot_phase(
            stop_event=stop_event,
            stage_queue=stage_queue,
            root=root,
            run_dir=run_dir,
            checkpoint=checkpoint,
            checkpoint_identity=checkpoint_identity,
            spec=spec,
            protocol_sha256=protocol_sha256,
            estimator_threads=estimator_threads,
        )


__all__ = ["cross_dataset_research_phase_worker"]
