from __future__ import annotations

from credit_risk_fs.clip.trainer import train_seed
from credit_risk_fs.clip.training_validation import load_and_validate_training_inputs, load_training_config


def test_same_seed_smoke_training_reproduces_checkpoint_hash(
    tmp_path, legacy_config_paths
):
    config = legacy_config_paths(load_training_config("configs/corrected_homecredit_clip/training.yaml"))
    data = load_and_validate_training_inputs(config)
    snapshot = "test_config_snapshot: true\n"

    first = train_seed(
        config=config,
        data=data,
        seed=11,
        output_dir=tmp_path / "first",
        config_snapshot_text=snapshot,
        smoke_test=True,
    )
    second = train_seed(
        config=config,
        data=data,
        seed=11,
        output_dir=tmp_path / "second",
        config_snapshot_text=snapshot,
        smoke_test=True,
    )

    assert first.checkpoint_hash == second.checkpoint_hash
    assert first.best_epoch == second.best_epoch
