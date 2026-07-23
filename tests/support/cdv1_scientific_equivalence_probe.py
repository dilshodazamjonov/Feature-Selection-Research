"""Small deterministic scientific probe; it never opens a project dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from credit_risk_fs.evaluation.metrics import evaluate_model
from credit_risk_fs.experiments.rank_voting import (
    REQUIRED_FIT_SCOPE,
    _fit_final_model,
    aggregate_cross_dataset_rank_voting,
    fit_rfe_memory_safe,
)


def _hash_json(value) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_probe(repository_root: str | Path) -> dict:
    root = Path(repository_root).resolve()
    rng = np.random.default_rng(42017)
    features = [f"feature_{index:02d}" for index in range(30)]
    matrix = rng.normal(size=(320, len(features))).astype("float32")
    linear = (
        1.4 * matrix[:, 0]
        - 1.1 * matrix[:, 3]
        + 0.8 * matrix[:, 7]
        + rng.normal(scale=0.35, size=len(matrix))
    )
    target = pd.Series((linear > np.median(linear)).astype("int8"))
    frame = pd.DataFrame(matrix, columns=features)
    rankings = {
        "rf_corr_mrmr": features[::2] + features[1::2],
        "boruta": list(reversed(features[5:])) + list(reversed(features[:5])),
    }
    aggregate = aggregate_cross_dataset_rank_voting(
        eligible_features=features,
        rankings=rankings,
        fit_scopes={
            "rf_corr_mrmr": REQUIRED_FIT_SCOPE,
            "boruta": REQUIRED_FIT_SCOPE,
        },
    )
    candidates = aggregate["feature"].astype(str).tolist()
    train = frame.iloc[:240].reset_index(drop=True)
    validation = frame.iloc[240:].reset_index(drop=True)
    y_train = target.iloc[:240].reset_index(drop=True)
    y_validation = target.iloc[240:].reset_index(drop=True)
    rfe = fit_rfe_memory_safe(
        X_numeric=train.loc[:, candidates].astype("float32"),
        y=y_train,
        top_candidates=candidates,
        model_name="lr",
        seed=42,
        estimator_threads=1,
    )
    selected = list(rfe["selected_features"])
    probabilities, effective = _fit_final_model(
        repository_root=root,
        dataset="homecredit",
        model_name="lr",
        selected_features=selected,
        X_train_raw=train,
        y_train=y_train,
        X_validation_raw=validation,
        seed=42,
        estimator_threads=1,
    )
    metrics = evaluate_model(y_validation, probabilities)
    ranking_bytes = aggregate.to_csv(index=False, lineterminator="\n").encode("utf-8")
    probability_bytes = np.asarray(probabilities, dtype="<f8").tobytes()
    return {
        "schema_version": "cdv1_scientific_equivalence_probe_v1",
        "synthetic_rows": len(frame),
        "synthetic_features": len(features),
        "ranking_sha256": hashlib.sha256(ranking_bytes).hexdigest(),
        "ranking_feature_order": candidates,
        "selected_features": selected,
        "selected_features_sha256": _hash_json(selected),
        "class_1_probability_sha256": hashlib.sha256(probability_bytes).hexdigest(),
        "metrics": metrics,
        "metrics_sha256": _hash_json(metrics),
        "probability_orientation": effective["probability_orientation"],
        "model_configuration": effective["requested_model_configuration"],
        "preprocessing_configuration": effective["preprocessing"]["configuration"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_probe(args.repository_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
