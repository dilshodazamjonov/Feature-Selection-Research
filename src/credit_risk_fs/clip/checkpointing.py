from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from credit_risk_fs.clip.model import (
    ClipModelConfig,
    SemanticStatisticalContrastiveEncoder,
)
from credit_risk_fs.clip.training_validation import ClipTrainingConfig
from credit_risk_fs.utils.hashing import sha256_file
from credit_risk_fs.utils.io import read_json, write_json


def save_checkpoint(
    *,
    model: SemanticStatisticalContrastiveEncoder,
    path: Path,
    manifest_path: Path,
    seed: int,
    epoch: int,
    validation_metric: str,
    validation_value: float,
    parameter_count: int,
    upstream_hashes: dict[str, str],
    git_commit: str,
    statistical_view_scope: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model.config.__dict__,
            "seed": int(seed),
            "epoch": int(epoch),
        },
        path,
    )
    checkpoint_hash = sha256_file(path)
    manifest = {
        "checkpoint_sha256": checkpoint_hash,
        "seed": int(seed),
        "epoch": int(epoch),
        "validation_criterion": validation_metric,
        "validation_value": float(validation_value),
        "architecture": model.config.__dict__,
        "parameter_count": int(parameter_count),
        "temperature": float(model.temperature().detach().cpu().item()),
        "git_commit": git_commit,
        "statistical_view_scope": statistical_view_scope,
        **upstream_hashes,
    }
    if extra:
        manifest.update(extra)
    write_json(manifest_path, manifest)
    return manifest


def load_checkpoint(
    *,
    checkpoint_path: str | Path,
    manifest_path: str | Path,
    config: ClipTrainingConfig,
    upstream_hashes: dict[str, str],
    expected_metadata: dict[str, Any] | None = None,
    map_location: str | torch.device = "cpu",
) -> SemanticStatisticalContrastiveEncoder:
    manifest = read_json(manifest_path)
    observed_checkpoint_hash = sha256_file(checkpoint_path)
    if observed_checkpoint_hash != manifest.get("checkpoint_sha256"):
        raise RuntimeError("checkpoint hash mismatch")
    for key, observed in upstream_hashes.items():
        if manifest.get(key) != observed:
            raise RuntimeError(f"upstream artifact hash mismatch for {key}")
    for key, expected in (expected_metadata or {}).items():
        if manifest.get(key) != expected:
            raise RuntimeError(
                f"checkpoint metadata mismatch for {key}: "
                f"expected {expected!r}, observed {manifest.get(key)!r}"
            )
    payload = torch.load(checkpoint_path, map_location=map_location)
    model_config = ClipModelConfig(**payload["model_config"])
    if model_config.text_input_dim != config.model.text_input_dim:
        raise RuntimeError("checkpoint text dimension does not match config")
    if model_config.statistical_input_dim != config.model.statistical_input_dim:
        raise RuntimeError("checkpoint statistical dimension does not match config")
    model = SemanticStatisticalContrastiveEncoder(model_config)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model
