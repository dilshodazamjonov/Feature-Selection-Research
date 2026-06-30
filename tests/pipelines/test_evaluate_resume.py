from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from credit_risk_fs.pipelines import reverse_transfer
from credit_risk_fs.pipelines.reverse_transfer import (
    TransferStageError,
    _evaluate,
    _evaluate_model_resume_state,
    execute_plan,
    load_config_dir,
)
from credit_risk_fs.utils.io import write_json


SEEDS = (11, 22, 33, 44, 55)
MODELS = ("lr", "catboost")


def _exercise_model_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    states: dict[str, str],
) -> list[str]:
    output = tmp_path / "out"
    (output / "reverse_projection").mkdir(parents=True)
    pd.DataFrame({"feature_name": ["x"]}).to_csv(
        output / "reverse_projection/homecredit_reverse_scores.csv",
        index=False,
    )
    write_json(
        output / "pairing/data_manifest.json",
        {"raw_dev_statistical_evidence_hash": "raw"},
    )
    config = load_config_dir("configs/corrected_lendingclub_to_homecredit")
    project = {
        "data_dir": str(tmp_path / "data"),
        "description_path": str(tmp_path / "description.csv"),
        "dev_start_day": -600,
        "oot_start_day": -240,
        "oot_end_day": 0,
        "n_splits": 5,
        "cv_gap_groups": 1,
        "random_seed": 42,
        "excluded_feature_columns": [],
        "preprocessor_kwargs": {},
    }
    monkeypatch.setattr(
        reverse_transfer, "load_named_project_config", lambda name: project
    )
    monkeypatch.setattr(
        reverse_transfer,
        "fixed_candidate_pool",
        lambda ranking, model, **kwargs: pd.DataFrame(
            {
                "feature_id": [f"{model}-1"],
                "feature_name": ["x"],
                "configuration_hash": ["unused"],
                "model": [model],
            }
        ),
    )
    monkeypatch.setattr(
        reverse_transfer,
        "_evaluate_model_resume_state",
        lambda **kwargs: states[kwargs["model"]],
    )
    monkeypatch.setattr(
        reverse_transfer, "_validate_completed_evaluate_model", lambda **kwargs: None
    )
    monkeypatch.setattr(
        reverse_transfer,
        "_evaluate_comparison_row",
        lambda run_dir, model, budget: {"model": model},
    )
    monkeypatch.setattr(reverse_transfer, "build_data_version", lambda path: {})
    monkeypatch.setattr(
        reverse_transfer, "resolve_model_kwargs", lambda project, model: {}
    )
    monkeypatch.setattr(
        reverse_transfer, "prepare_modeling_data", lambda experiment: object()
    )
    executed: list[str] = []

    def fake_run(experiment, prepared_data):
        executed.append(experiment.model_name)
        run_dir = Path(experiment.experiment_output_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(exp_dir=run_dir)

    monkeypatch.setattr(reverse_transfer, "run_experiment", fake_run)
    _evaluate(
        config=config,
        output_dir=output,
        seeds=SEEDS,
        models=MODELS,
        resume=True,
    )
    return executed


def test_authenticated_lr_and_absent_catboost_runs_only_catboost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executed = _exercise_model_resume(
        tmp_path,
        monkeypatch,
        {"lr": "complete", "catboost": "absent"},
    )
    assert executed == ["catboost"]


def test_completed_authenticated_lr_is_not_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert "lr" not in _exercise_model_resume(
        tmp_path,
        monkeypatch,
        {"lr": "complete", "catboost": "absent"},
    )


def test_tampered_candidate_pool_is_rejected(
    tmp_path: Path,
) -> None:
    expected = pd.DataFrame(
        {
            "feature_id": ["a"],
            "feature_name": ["x"],
            "configuration_hash": ["config"],
            "model": ["lr"],
        }
    )
    path = tmp_path / "pool.csv"
    expected.assign(feature_name="tampered").to_csv(path, index=False)
    with pytest.raises(TransferStageError, match="candidate pool mismatch"):
        reverse_transfer._validate_candidate_pool(
            actual_path=path,
            expected=expected,
            model="lr",
            config_hash="config",
        )


@pytest.mark.parametrize(
    "missing_name",
    ["prediction_manifest.json", "oof_reconciliation.csv"],
)
def test_incomplete_model_or_missing_reconciliation_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing_name: str,
) -> None:
    paths = [tmp_path / "pool.csv", tmp_path / "run" / "complete.txt"]
    monkeypatch.setattr(
        reverse_transfer, "_evaluate_model_artifact_paths", lambda *args: paths
    )
    paths[0].write_text("pool", encoding="utf-8")
    paths[1].parent.mkdir(parents=True)
    paths[1].write_text("partial", encoding="utf-8")
    paths.append(tmp_path / "run" / missing_name)
    with pytest.raises(TransferStageError, match="incomplete existing"):
        _evaluate_model_resume_state(
            config={},
            output_dir=tmp_path,
            model="lr",
            project={},
            config_hash="config",
            expected_pool=pd.DataFrame(),
            budget={},
        )


def _seed_partial_evaluate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict, Path, dict[str, Path]]:
    output = tmp_path / "out"
    artifacts = {
        stage: output / f"{stage}.txt"
        for stage in ("prepare", "train", "project", "evaluate")
    }

    def success(stage: str):
        def handler(**kwargs):
            artifacts[stage].parent.mkdir(parents=True, exist_ok=True)
            artifacts[stage].write_text(stage, encoding="utf-8")
            return {}

        return handler

    monkeypatch.setattr(reverse_transfer, "DEFAULT_OUTPUT_ROOT", output)
    monkeypatch.setattr(reverse_transfer, "_prepare", success("prepare"))
    monkeypatch.setattr(reverse_transfer, "_train", success("train"))
    monkeypatch.setattr(reverse_transfer, "_project", success("project"))
    monkeypatch.setattr(
        reverse_transfer,
        "_stage_artifact_paths",
        lambda stage, *args, **kwargs: [artifacts[stage]],
    )
    config = load_config_dir("configs/corrected_lendingclub_to_homecredit")
    common = {
        "config": config,
        "seeds": SEEDS,
        "models": MODELS,
        "output_dir": output,
        "resume": False,
        "skip_existing": False,
    }
    for stage in ("prepare", "train", "project"):
        execute_plan(**common, stages=(stage,))
    write_json(
        output / "manifests/evaluate_stage_manifest.json",
        {
            "stage": "evaluate",
            "status": "in_progress",
            "configuration_hash": reverse_transfer._configuration_hash(config),
            "source_dataset": reverse_transfer.SOURCE_DATASET,
            "external_dataset": reverse_transfer.EXTERNAL_DATASET,
            "pairing_policy_version": reverse_transfer.PAIRING_POLICY_VERSION,
            "requested_seeds": list(SEEDS),
            "requested_models": list(MODELS),
        },
    )
    return common, output, artifacts


def test_catboost_failure_leaves_evaluate_in_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    common, output, _ = _seed_partial_evaluate(tmp_path, monkeypatch)

    def fail(**kwargs):
        raise RuntimeError("catboost failed")

    monkeypatch.setattr(reverse_transfer, "_evaluate", fail)
    with pytest.raises(RuntimeError, match="catboost failed"):
        execute_plan(**{**common, "resume": True}, stages=("evaluate",))
    assert (
        reverse_transfer.read_json(
            output / "manifests/evaluate_stage_manifest.json"
        )["status"]
        == "in_progress"
    )


def test_both_valid_models_produce_complete_evaluate_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    common, output, artifacts = _seed_partial_evaluate(tmp_path, monkeypatch)

    def finish(**kwargs):
        artifacts["evaluate"].write_text("both valid", encoding="utf-8")
        return {"runs": [{"model": "lr"}, {"model": "catboost"}]}

    monkeypatch.setattr(reverse_transfer, "_evaluate", finish)
    execute_plan(**{**common, "resume": True}, stages=("evaluate",))
    assert (
        reverse_transfer.read_json(
            output / "manifests/evaluate_stage_manifest.json"
        )["status"]
        == "complete"
    )


def test_without_resume_partial_evaluate_is_still_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    common, _, _ = _seed_partial_evaluate(tmp_path, monkeypatch)
    with pytest.raises(TransferStageError, match="use --resume"):
        execute_plan(**common, stages=("evaluate",))


def test_evaluate_resume_never_invokes_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    common, _, artifacts = _seed_partial_evaluate(tmp_path, monkeypatch)
    monkeypatch.setattr(
        reverse_transfer,
        "_register",
        lambda **kwargs: pytest.fail("registry must not run"),
    )

    def finish(**kwargs):
        artifacts["evaluate"].write_text("complete", encoding="utf-8")
        return {}

    monkeypatch.setattr(reverse_transfer, "_evaluate", finish)
    execute_plan(**{**common, "resume": True}, stages=("evaluate",))
