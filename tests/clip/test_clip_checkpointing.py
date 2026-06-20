from __future__ import annotations

import pytest

from credit_risk_fs.clip.checkpointing import load_checkpoint
from credit_risk_fs.clip.training_validation import load_and_validate_training_inputs, load_training_config
from credit_risk_fs.utils.io import read_json


def test_selected_checkpoint_loads_with_current_upstream_hashes():
    config = load_training_config()
    data = load_and_validate_training_inputs(config)
    selection = read_json("results/clip/training/model_selection_manifest.json")

    model = load_checkpoint(
        checkpoint_path=selection["selected_checkpoint_path"],
        manifest_path=f"results/clip/training/seeds/seed_{selection['selected_seed']}/checkpoint_manifest.json",
        config=config,
        upstream_hashes=data.upstream_hashes,
    )

    assert model.config.text_input_dim == 384
    assert model.config.statistical_input_dim == 1


def test_stale_upstream_hash_prevents_checkpoint_loading():
    config = load_training_config()
    data = load_and_validate_training_inputs(config)
    selection = read_json("results/clip/training/model_selection_manifest.json")
    stale = dict(data.upstream_hashes)
    stale["negative_policy_manifest_hash"] = "stale"

    with pytest.raises(RuntimeError, match="upstream artifact hash mismatch"):
        load_checkpoint(
            checkpoint_path=selection["selected_checkpoint_path"],
            manifest_path=f"results/clip/training/seeds/seed_{selection['selected_seed']}/checkpoint_manifest.json",
            config=config,
            upstream_hashes=stale,
        )
