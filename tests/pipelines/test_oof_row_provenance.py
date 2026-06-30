from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from credit_risk_fs.models import training
from credit_risk_fs.models._cv_utils import GroupedTimeSeriesSplit
from credit_risk_fs.models.training import (
    build_fold_prediction_frame,
    evaluate_reconciled_oof,
    reconcile_oof_predictions,
)
from credit_risk_fs.pipelines import reverse_transfer
from credit_risk_fs.pipelines.common import _stable_row_ids
from credit_risk_fs.pipelines.reverse_transfer import (
    TransferStageError,
    execute_plan,
    load_config_dir,
)


def _manifest(fold: int, ids: list[str]) -> dict[str, object]:
    return {
        "fold_id": fold,
        "training_ids": [f"train-{fold}"],
        "validation_ids": ids,
        "validation_row_count": len(ids),
        "training_id_hash": hashlib.sha256(
            json.dumps([f"train-{fold}"], separators=(",", ":")).encode()
        ).hexdigest(),
        "validation_id_hash": hashlib.sha256(
            json.dumps(sorted(ids), separators=(",", ":")).encode()
        ).hexdigest(),
    }


def _fold(fold: int, ids: list[str]) -> pd.DataFrame:
    index = pd.Index(range(100 * fold, 100 * fold + len(ids)))
    return build_fold_prediction_frame(
        validation_ids=pd.Series(ids, index=index),
        validation_targets=pd.Series(
            [value % 2 for value in range(len(ids))], index=index
        ),
        prediction_probabilities=np.linspace(0.1, 0.9, len(ids)),
        fold=fold,
        threshold=0.5,
    )


def test_fold_predictions_map_one_to_one_to_validation_ids() -> None:
    frame = _fold(1, ["1001", "1002", "1003"])
    assert len(frame) == 3
    assert frame["stable_row_id"].nunique() == 3
    assert set(frame["stable_row_id"]) == {"1001", "1002", "1003"}


def test_prediction_series_index_cannot_align_away_rows() -> None:
    ids = pd.Series(["1001", "1002"], index=[10, 20])
    targets = pd.Series([0, 1], index=[10, 20])
    probabilities = pd.Series([0.2, 0.8], index=[999, 998])
    frame = build_fold_prediction_frame(
        validation_ids=ids,
        validation_targets=targets,
        prediction_probabilities=probabilities,
        fold=1,
        threshold=0.5,
    )
    assert frame["prediction_probability"].tolist() == [0.2, 0.8]


def test_one_missing_prediction_is_rejected() -> None:
    frame = _fold(1, ["1001", "1002"]).iloc[:1]
    with pytest.raises(RuntimeError, match="reconciliation failed"):
        reconcile_oof_predictions([frame], [_manifest(1, ["1001", "1002"])])


def test_one_extra_prediction_is_rejected() -> None:
    frame = _fold(1, ["1001", "1002"])
    extra = frame.iloc[-1].to_dict()
    extra["stable_row_id"] = "1003"
    frame.loc[len(frame)] = extra
    with pytest.raises(RuntimeError, match="reconciliation failed"):
        reconcile_oof_predictions([frame], [_manifest(1, ["1001", "1002"])])


def test_duplicate_prediction_id_is_rejected() -> None:
    frame = _fold(1, ["1001", "1002"])
    frame.loc[1, "stable_row_id"] = "1001"
    with pytest.raises(RuntimeError, match="reconciliation failed"):
        reconcile_oof_predictions([frame], [_manifest(1, ["1001", "1002"])])


def test_duplicate_validation_id_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="duplicated"):
        build_fold_prediction_frame(
            validation_ids=["1001", "1001"],
            validation_targets=[0, 1],
            prediction_probabilities=[0.2, 0.8],
            fold=1,
            threshold=0.5,
        )


@pytest.mark.parametrize("probability", [np.nan, np.inf, -np.inf])
def test_non_finite_prediction_is_rejected(probability: float) -> None:
    with pytest.raises(RuntimeError, match="finite probabilities"):
        build_fold_prediction_frame(
            validation_ids=["1001"],
            validation_targets=[0],
            prediction_probabilities=[probability],
            fold=1,
            threshold=0.5,
        )


def test_target_and_id_index_misalignment_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="indexes are misaligned"):
        build_fold_prediction_frame(
            validation_ids=pd.Series(["1001", "1002"], index=[1, 2]),
            validation_targets=pd.Series([0, 1], index=[2, 1]),
            prediction_probabilities=[0.2, 0.8],
            fold=1,
            threshold=0.5,
        )


def test_oof_union_equals_all_fold_validation_ids() -> None:
    frames = [_fold(1, ["1001", "1002"]), _fold(2, ["1003", "1004"])]
    oof, reconciliation = reconcile_oof_predictions(
        frames,
        [_manifest(1, ["1001", "1002"]), _manifest(2, ["1003", "1004"])],
    )
    assert len(oof) == 4
    assert oof["stable_row_id"].nunique() == 4
    assert reconciliation["missing_prediction_ids"].sum() == 0


def test_oof_union_missing_one_fold_row_is_rejected() -> None:
    frames = [_fold(1, ["1001", "1002"]), _fold(2, ["1003"])]
    with pytest.raises(RuntimeError, match="reconciliation failed"):
        reconcile_oof_predictions(
            frames,
            [
                _manifest(1, ["1001", "1002"]),
                _manifest(2, ["1003", "1004"]),
            ],
        )


def test_oof_metrics_are_not_called_before_reconciliation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def fake_evaluate(*args: object, **kwargs: object) -> dict[str, float]:
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(training, "evaluate_model", fake_evaluate)
    with pytest.raises(RuntimeError, match="reconciliation failed"):
        evaluate_reconciled_oof(
            [_fold(1, ["1001"])],
            [_manifest(1, ["1001", "1002"])],
        )
    assert not called


def test_failed_evaluate_cannot_write_complete_stage_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "out"
    artifacts = {
        stage: output / f"{stage}.txt"
        for stage in ("prepare", "train", "project", "evaluate")
    }

    def successful(stage: str):
        def handler(**kwargs: object) -> dict[str, object]:
            artifacts[stage].parent.mkdir(parents=True, exist_ok=True)
            artifacts[stage].write_text(stage, encoding="utf-8")
            return {}

        return handler

    monkeypatch.setattr(reverse_transfer, "DEFAULT_OUTPUT_ROOT", output)
    monkeypatch.setattr(reverse_transfer, "_prepare", successful("prepare"))
    monkeypatch.setattr(reverse_transfer, "_train", successful("train"))
    monkeypatch.setattr(reverse_transfer, "_project", successful("project"))
    monkeypatch.setattr(
        reverse_transfer,
        "_stage_artifact_paths",
        lambda stage, *args, **kwargs: [artifacts[stage]],
    )
    config = load_config_dir("configs/corrected_lendingclub_to_homecredit")
    common = {
        "config": config,
        "seeds": (11, 22, 33, 44, 55),
        "models": ("lr", "catboost"),
        "output_dir": output,
        "resume": False,
        "skip_existing": False,
    }
    execute_plan(**common, stages=("prepare",))
    execute_plan(**common, stages=("train",))
    execute_plan(**common, stages=("project",))

    def fail_evaluate(**kwargs: object) -> dict[str, object]:
        raise RuntimeError("OOF reconciliation failed")

    monkeypatch.setattr(reverse_transfer, "_evaluate", fail_evaluate)
    with pytest.raises(RuntimeError, match="OOF reconciliation failed"):
        execute_plan(**common, stages=("evaluate",))
    manifest = reverse_transfer.read_json(
        output / "manifests/evaluate_stage_manifest.json"
    )
    assert manifest["status"] == "in_progress"


def test_authentic_stable_id_values_are_preserved() -> None:
    frame = _fold(1, ["307001", "100002"])
    assert set(frame["stable_row_id"]) == {"307001", "100002"}


def test_generated_positional_identifier_is_rejected() -> None:
    frame = pd.DataFrame({"index": [0, 1], "TARGET": [0, 1]})
    with pytest.raises(ValueError, match="persistent source column"):
        _stable_row_ids(
            frame,
            dataset="homecredit",
            stable_row_id_column="index",
        )


def test_grouped_temporal_validation_folds_are_disjoint() -> None:
    values = np.array([1] * 5 + [2] * 5 + [3] * 5 + [4] * 5)
    validations = [
        set(validation.tolist())
        for _, validation in GroupedTimeSeriesSplit(n_splits=2).split(values)
    ]
    assert validations[0].isdisjoint(validations[1])
