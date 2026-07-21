from __future__ import annotations

from credit_risk_fs.clip.training_validation import load_and_validate_training_inputs, load_training_config


def test_training_inputs_enforce_homecredit_train_and_external_lendingclub_only(
    legacy_config_paths,
):
    config = legacy_config_paths(load_training_config("configs/corrected_homecredit_clip/training.yaml"))
    data = load_and_validate_training_inputs(config)

    assert set(data.train_pairs["dataset"]) == {"homecredit"}
    assert set(data.train_pairs["split"]) == {"train"}
    assert data.train_pairs["allowed_for_training"].all()
    assert set(data.validation_pairs["split"]) == {"validation"}
    assert not data.validation_pairs["allowed_for_training"].any()
    assert set(data.external_pairs["dataset"]) == {"lendingclub_v2"}
    assert not data.external_pairs["allowed_for_training"].any()
    assert data.statistical_dim == 13
    assert config.statistical_view_scope == "compact_target_free_v2"


def test_forbidden_fields_are_absent_from_pair_inputs(legacy_config_paths):
    config = legacy_config_paths(load_training_config("configs/corrected_homecredit_clip/training.yaml"))
    data = load_and_validate_training_inputs(config)
    columns = set(data.train_pairs.columns).union(data.validation_pairs.columns).union(data.external_pairs.columns)

    forbidden_tokens = ["llm", "oot", "psi", "target", "label", "prediction", "stable_core"]
    assert not [column for column in columns if any(token in column.lower() for token in forbidden_tokens)]
