from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from credit_risk_fs.pipelines.common import (
    DERIVED_METRIC_VALIDATION_TOLERANCE,
    PredictionMetadata,
    compute_score_psi,
    export_prediction_artifact,
    generate_metric_provenance_artifacts,
    prediction_metrics_from_saved_files,
    validate_metric_provenance,
)
from credit_risk_fs.utils.hashing import sha256_file


def _metadata(model: str, split: str) -> PredictionMetadata:
    return PredictionMetadata(
        dataset="homecredit",
        split=split,
        run_id=f"metric-run-{model}",
        method="reverse_transfer",
        model=model,
        source_training_dataset="lendingclub_v2",
        external_dataset="homecredit",
        configuration_hash="c" * 64,
        data_manifest_hash="d" * 64,
        pairing_policy_version="identity_equivalence_v2",
        source_identity_manifest_hash="i" * 64,
        stable_row_id_column="SK_ID_CURR",
        source_stable_id_values_hash="s" * 64,
    )


def _fold_manifest(ids: list[str]) -> list[dict[str, object]]:
    folds = []
    for fold_id, validation_ids in enumerate((ids[:10], ids[10:]), start=1):
        training_ids = ids[10:] if fold_id == 1 else ids[:10]
        folds.append(
            {
                "fold_id": fold_id,
                "training_id_hash": hashlib.sha256(
                    json.dumps(
                        sorted(training_ids), separators=(",", ":")
                    ).encode()
                ).hexdigest(),
                "validation_id_hash": hashlib.sha256(
                    json.dumps(
                        sorted(validation_ids), separators=(",", ":")
                    ).encode()
                ).hexdigest(),
                "validation_row_count": len(validation_ids),
                "model_fit_scope": "fold_training_ids_only",
                "training_ids": training_ids,
                "validation_ids": validation_ids,
            }
        )
    return folds


def _prediction_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    dev_ids = [str(100000 + index) for index in range(20)]
    oot_ids = [str(200000 + index) for index in range(20)]
    targets = np.tile([0, 1], 10)
    dev_zero = iter(
        [0.05, 0.07, 0.09, 0.11, 0.13, 0.70, 0.72, 0.74, 0.76, 0.78]
    )
    dev_one = iter(
        [0.15, 0.17, 0.19, 0.21, 0.23, 0.80, 0.82, 0.84, 0.86, 0.88]
    )
    oot_zero = iter([0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70])
    oot_one = iter([0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85])
    dev_probabilities = [
        next(dev_one) if target else next(dev_zero) for target in targets
    ]
    oot_probabilities = [
        next(oot_one) if target else next(oot_zero) for target in targets
    ]
    dev = pd.DataFrame(
        {
            "stable_row_id": dev_ids,
            "fold_id": np.repeat([1, 2], 10),
            "target": targets,
            "prediction_probability": dev_probabilities,
            "predicted_class": (np.asarray(dev_probabilities) >= 0.5).astype(int),
        }
    )
    oot = pd.DataFrame(
        {
            "stable_row_id": oot_ids,
            "target": targets,
            "prediction_probability": oot_probabilities,
            "predicted_class": (np.asarray(oot_probabilities) >= 0.5).astype(int),
        }
    )
    return dev, oot


def _generated_artifacts(
    root: Path, model: str = "lr"
) -> tuple[dict, pd.DataFrame]:
    dev, oot = _prediction_frames()
    dev_ids = dev["stable_row_id"].tolist()
    dev_targets = dict(zip(dev_ids, dev["target"]))
    oot_targets = dict(zip(oot["stable_row_id"], oot["target"]))
    dev_path = root / "dev_oof_predictions.csv"
    oot_path = root / "oot_predictions.csv"
    _, dev_manifest = export_prediction_artifact(
        dev,
        metadata=_metadata(model, "DEV_OOF"),
        path=dev_path,
        threshold=0.5,
        expected_ids=set(dev_ids),
        expected_targets=dev_targets,
        fold_manifest=_fold_manifest(dev_ids),
        forbidden_ids=set(oot["stable_row_id"]),
    )
    _, oot_manifest = export_prediction_artifact(
        oot,
        metadata=_metadata(model, "oot"),
        path=oot_path,
        threshold=0.5,
        expected_ids=set(oot["stable_row_id"]),
        expected_targets=oot_targets,
    )
    metrics, manifest = generate_metric_provenance_artifacts(
        dev_prediction_path=dev_path,
        oot_prediction_path=oot_path,
        prediction_manifests=[dev_manifest, oot_manifest],
        metrics_path=root / "prediction_metrics.csv",
        psi_details_path=root / "psi_details.csv",
        metric_manifest_path=root / "metric_manifest.json",
        threshold=0.5,
        run_id=f"metric-run-{model}",
        model=model,
        configuration_hash="c" * 64,
        data_manifest_hash="d" * 64,
        expected_dev_targets=dev_targets,
    )
    return manifest, metrics


@pytest.mark.parametrize("model", ["lr", "catboost"])
def test_valid_derived_metric_artifacts_reproduce_for_both_models(
    tmp_path: Path, model: str
) -> None:
    manifest, metrics = _generated_artifacts(tmp_path / model, model)
    validate_metric_provenance(manifest)
    regenerated = prediction_metrics_from_saved_files(
        manifest["prediction_manifests"][0]["prediction_path"],
        manifest["prediction_manifests"][1]["prediction_path"],
        threshold=0.5,
    )
    pd.testing.assert_frame_equal(
        metrics[regenerated.columns].reset_index(drop=True),
        regenerated.reset_index(drop=True),
    )
    indexed = metrics.set_index("split")
    expected_drop = indexed.loc["DEV_OOF", "auc"] - indexed.loc["oot", "auc"]
    assert manifest["auc_drop"] == pytest.approx(expected_drop, abs=1e-15)
    assert manifest["score_psi"] == pytest.approx(
        indexed.loc["DEV_OOF", "score_psi"], abs=1e-15
    )
    details = pd.read_csv(
        manifest["derived_metrics"]["score_psi"]["psi_details_path"]
    )
    assert details["psi_contribution"].sum() == pytest.approx(
        manifest["score_psi"], abs=DERIVED_METRIC_VALIDATION_TOLERANCE
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "plus",
        "minus",
        "reversed",
        "mean_fold",
        "dev_auc",
        "oot_auc",
        "dev_prediction",
        "oot_prediction",
        "dev_hash",
        "oot_hash",
        "dev_count",
        "oot_count",
    ],
)
def test_auc_drop_adversarial_claims_are_rejected(
    tmp_path: Path, mutation: str
) -> None:
    manifest, _ = _generated_artifacts(tmp_path)
    bad = deepcopy(manifest)
    auc = bad["derived_metrics"]["auc_drop"]
    dev_manifest, oot_manifest = bad["prediction_manifests"]
    if mutation == "plus":
        bad["auc_drop"] += 0.01
    elif mutation == "minus":
        bad["auc_drop"] -= 0.01
    elif mutation == "reversed":
        bad["auc_drop"] = -bad["auc_drop"]
        bad["auc_drop_convention"] = "OOT_AUC_MINUS_DEV_OOF_AUC"
    elif mutation == "mean_fold":
        dev_frame = pd.read_csv(dev_manifest["prediction_path"])
        mean_fold_auc = np.mean(
            [
                roc_auc_score(
                    fold["target"], fold["prediction_probability"]
                )
                for _, fold in dev_frame.groupby("fold_id")
            ]
        )
        assert mean_fold_auc != pytest.approx(auc["dev_oof_auc"])
        auc["dev_oof_auc"] = float(mean_fold_auc)
    elif mutation == "dev_auc":
        auc["dev_oof_auc"] += 0.01
    elif mutation == "oot_auc":
        auc["oot_auc"] += 0.01
    elif mutation in {"dev_prediction", "oot_prediction"}:
        item = dev_manifest if mutation.startswith("dev") else oot_manifest
        frame = pd.read_csv(item["prediction_path"])
        frame.loc[0, "prediction_probability"] += 0.001
        frame.to_csv(item["prediction_path"], index=False)
    elif mutation in {"dev_hash", "oot_hash"}:
        item = dev_manifest if mutation.startswith("dev") else oot_manifest
        item["prediction_hash"] = "f" * 64
    elif mutation in {"dev_count", "oot_count"}:
        item = dev_manifest if mutation.startswith("dev") else oot_manifest
        item["row_count"] += 1
    with pytest.raises(ValueError):
        validate_metric_provenance(bad)


@pytest.mark.parametrize(
    "mutation",
    [
        "plus",
        "minus",
        "zero",
        "reversed_scopes",
        "in_sample_reference",
        "oot_fitted_bins",
        "edge",
        "effective_bins",
        "epsilon",
        "contribution",
        "contribution_sum",
        "dev_prediction",
        "oot_prediction",
        "reference_hash",
        "comparison_hash",
        "reference_count",
        "comparison_count",
        "missing_details",
    ],
)
def test_psi_adversarial_claims_are_rejected(
    tmp_path: Path, mutation: str
) -> None:
    manifest, _ = _generated_artifacts(tmp_path)
    bad = deepcopy(manifest)
    psi = bad["derived_metrics"]["score_psi"]
    dev_manifest, oot_manifest = bad["prediction_manifests"]
    if mutation == "plus":
        bad["score_psi"] += 0.01
    elif mutation == "minus":
        bad["score_psi"] -= 0.01
    elif mutation == "zero":
        psi["metric_value"] = 0.0
    elif mutation == "reversed_scopes":
        bad["psi_reference_split"], bad["psi_comparison_split"] = (
            bad["psi_comparison_split"],
            bad["psi_reference_split"],
        )
    elif mutation == "in_sample_reference":
        psi["reference_scope"] = "dev"
    elif mutation == "oot_fitted_bins":
        dev = pd.read_csv(dev_manifest["prediction_path"])
        oot = pd.read_csv(oot_manifest["prediction_path"])
        _, _, definition = compute_score_psi(
            oot["prediction_probability"], dev["prediction_probability"]
        )
        psi["bin_edges"] = definition["bin_edges"]
    elif mutation == "edge":
        psi["bin_edges"][1] += 0.001
    elif mutation == "effective_bins":
        psi["effective_bin_count"] += 1
    elif mutation == "epsilon":
        psi["smoothing_epsilon"] = 1e-3
    elif mutation in {"contribution", "contribution_sum"}:
        details_path = Path(psi["psi_details_path"])
        details = pd.read_csv(details_path)
        details.loc[0, "psi_contribution"] += (
            0.001 if mutation == "contribution" else 0.01
        )
        details.to_csv(details_path, index=False)
        psi["psi_details_hash"] = sha256_file(details_path)
    elif mutation in {"dev_prediction", "oot_prediction"}:
        item = dev_manifest if mutation.startswith("dev") else oot_manifest
        frame = pd.read_csv(item["prediction_path"])
        frame.loc[0, "prediction_probability"] += 0.001
        frame.to_csv(item["prediction_path"], index=False)
    elif mutation == "reference_hash":
        psi["reference_prediction_hash"] = "f" * 64
    elif mutation == "comparison_hash":
        psi["comparison_prediction_hash"] = "f" * 64
    elif mutation == "reference_count":
        psi["reference_row_count"] += 1
    elif mutation == "comparison_count":
        psi["comparison_row_count"] += 1
    elif mutation == "missing_details":
        Path(psi["psi_details_path"]).unlink()
    with pytest.raises(ValueError):
        validate_metric_provenance(bad)


@pytest.mark.parametrize("invalid_value", [1.01, np.inf])
def test_invalid_saved_probabilities_fail_metric_generation(
    tmp_path: Path, invalid_value: float
) -> None:
    manifest, _ = _generated_artifacts(tmp_path)
    dev = manifest["prediction_manifests"][0]
    frame = pd.read_csv(dev["prediction_path"])
    frame.loc[0, "prediction_probability"] = invalid_value
    frame.to_csv(dev["prediction_path"], index=False)
    with pytest.raises(ValueError, match="probabilities"):
        generate_metric_provenance_artifacts(
            dev_prediction_path=dev["prediction_path"],
            oot_prediction_path=manifest["prediction_manifests"][1][
                "prediction_path"
            ],
            prediction_manifests=manifest["prediction_manifests"],
            metrics_path=tmp_path / "invalid" / "prediction_metrics.csv",
            psi_details_path=tmp_path / "invalid" / "psi_details.csv",
            metric_manifest_path=tmp_path
            / "invalid"
            / "metric_manifest.json",
            threshold=0.5,
            run_id="metric-run-lr",
            model="lr",
            configuration_hash="c" * 64,
            data_manifest_hash="d" * 64,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "model",
        "run_id",
        "configuration_hash",
        "data_manifest_hash",
        "source_identity_manifest_hash",
        "missing_metric_version",
        "missing_psi_version",
    ],
)
def test_derived_metric_provenance_mismatches_fail_closed(
    tmp_path: Path, mutation: str
) -> None:
    manifest, _ = _generated_artifacts(tmp_path)
    bad = deepcopy(manifest)
    if mutation in {
        "model",
        "run_id",
        "configuration_hash",
        "data_manifest_hash",
        "source_identity_manifest_hash",
    }:
        bad["derived_metrics"]["auc_drop"][mutation] = "wrong"
    elif mutation == "missing_metric_version":
        del bad["metric_implementation_version"]
    elif mutation == "missing_psi_version":
        del bad["derived_metrics"]["score_psi"][
            "psi_implementation_version"
        ]
    with pytest.raises(ValueError):
        validate_metric_provenance(bad)


def test_psi_is_zero_for_identical_distributions_and_positive_for_shift(
) -> None:
    reference = np.linspace(0.01, 0.99, 100)
    identical, _, _ = compute_score_psi(reference, reference[::-1])
    shifted, details, definition = compute_score_psi(
        reference, np.clip(reference + 0.2, 0.0, 1.0)
    )
    assert identical == pytest.approx(0.0, abs=1e-15)
    assert shifted > 0.0
    assert details["psi_contribution"].sum() == pytest.approx(
        shifted, abs=1e-15
    )
    assert definition["effective_bin_count"] == len(
        definition["bin_edges"]
    ) - 1


def test_duplicate_quantile_edges_and_repeated_validation_are_deterministic(
    tmp_path: Path,
) -> None:
    reference = np.array([0.1] * 30 + [0.5] * 40 + [0.9] * 30)
    comparison = np.array([0.1] * 20 + [0.5] * 30 + [0.9] * 50)
    first = compute_score_psi(reference, comparison)
    second = compute_score_psi(reference[::-1], comparison[::-1])
    assert first[0] == second[0]
    assert first[2]["bin_edges"] == second[2]["bin_edges"]
    pd.testing.assert_frame_equal(first[1], second[1])
    manifest, _ = _generated_artifacts(tmp_path)
    validate_metric_provenance(manifest)
    validate_metric_provenance(manifest)


def test_row_shuffling_preserves_auc_drop_and_psi(tmp_path: Path) -> None:
    manifest, original = _generated_artifacts(tmp_path / "original")
    shuffled_manifests = deepcopy(manifest["prediction_manifests"])
    for item in shuffled_manifests:
        path = Path(item["prediction_path"])
        pd.read_csv(path).sample(frac=1.0, random_state=7).to_csv(
            path, index=False
        )
        item["prediction_hash"] = sha256_file(path)
    shuffled, shuffled_manifest = generate_metric_provenance_artifacts(
        dev_prediction_path=shuffled_manifests[0]["prediction_path"],
        oot_prediction_path=shuffled_manifests[1]["prediction_path"],
        prediction_manifests=shuffled_manifests,
        metrics_path=tmp_path / "shuffled" / "prediction_metrics.csv",
        psi_details_path=tmp_path / "shuffled" / "psi_details.csv",
        metric_manifest_path=tmp_path / "shuffled" / "metric_manifest.json",
        threshold=0.5,
        run_id="metric-run-lr",
        model="lr",
        configuration_hash="c" * 64,
        data_manifest_hash="d" * 64,
    )
    assert shuffled_manifest["auc_drop"] == pytest.approx(
        manifest["auc_drop"], abs=1e-15
    )
    assert shuffled_manifest["score_psi"] == pytest.approx(
        manifest["score_psi"], abs=1e-15
    )
    assert shuffled["auc"].tolist() == pytest.approx(original["auc"].tolist())
