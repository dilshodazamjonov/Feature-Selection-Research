from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from credit_risk_fs.clip.checkpointing import load_checkpoint
from credit_risk_fs.clip.training_validation import (
    ClipTrainingConfig,
    TrainingDataBundle,
    false_negative_mask,
    tensors_for_pairs,
)
from credit_risk_fs.utils.hashing import sha256_text
from credit_risk_fs.utils.io import write_json


def build_learned_outputs(
    *,
    config: ClipTrainingConfig,
    data: TrainingDataBundle,
    selected_checkpoint_path: Path,
    selected_checkpoint_manifest_path: Path,
    output_dir: Path,
) -> dict[str, Path | dict[str, Any]]:
    model = load_checkpoint(
        checkpoint_path=selected_checkpoint_path,
        manifest_path=selected_checkpoint_manifest_path,
        config=config,
        upstream_hashes=data.upstream_hashes,
        map_location="cpu",
    )
    home_text, home_stat = tensors_for_pairs(data.homecredit_pairs, data.homecredit_text, data.homecredit_stat)
    lc_text, lc_stat = tensors_for_pairs(data.external_pairs, data.lendingclub_text, data.lendingclub_stat)
    with torch.no_grad():
        home_text_proj, home_stat_proj = model(home_text, home_stat)
        lc_text_proj, lc_stat_proj = model(lc_text, lc_stat)
    home_joint = _joint(home_text_proj, home_stat_proj)
    lc_joint = _joint(lc_text_proj, lc_stat_proj)
    checkpoint_hash = str(json.loads(selected_checkpoint_manifest_path.read_text(encoding="utf-8"))["checkpoint_sha256"])

    home_embeddings = _embedding_frame(data.homecredit_pairs, home_text_proj, home_stat_proj, home_joint, checkpoint_hash, config)
    lc_embeddings = _embedding_frame(data.external_pairs, lc_text_proj, lc_stat_proj, lc_joint, checkpoint_hash, config)
    home_path = output_dir / "homecredit_joint_embeddings.parquet"
    lc_path = output_dir / "lendingclub_v2_joint_embeddings.parquet"
    home_embeddings.to_parquet(home_path, index=False)
    lc_embeddings.to_parquet(lc_path, index=False)

    anchor_names = _training_anchor_names(data.homecredit_pairs)
    anchor_frame = home_embeddings[home_embeddings["feature_name"].isin(anchor_names)].copy()
    if anchor_frame.empty:
        raise RuntimeError("no Home Credit training-split anchors available for learned scoring")
    joint_cols = [col for col in home_embeddings.columns if str(col).startswith("joint_") and len(str(col)) == 10]
    centroid = anchor_frame[joint_cols].to_numpy(dtype=float).mean(axis=0)
    centroid = centroid / max(float(np.linalg.norm(centroid)), 1e-12)
    anchor_hash = sha256_text(",".join(f"{value:.12g}" for value in centroid.tolist()))

    home_scores = _score_frame(home_embeddings, centroid, checkpoint_hash, anchor_hash, config)
    lc_scores = _score_frame(lc_embeddings, centroid, checkpoint_hash, anchor_hash, config)
    home_score_path = output_dir / "homecredit_learned_scores.csv"
    lc_score_path = output_dir / "lendingclub_v2_learned_scores.csv"
    home_scores.to_csv(home_score_path, index=False)
    lc_scores.to_csv(lc_score_path, index=False)
    anchor_manifest = {
        "anchor_dataset": "homecredit",
        "anchor_policy": "Home Credit training-split stable-core anchor only",
        "lendingclub_v2_anchor_policy": "uses unchanged Home Credit learned anchor",
        "anchor_count": int(len(anchor_frame)),
        "anchor_hash": anchor_hash,
        "checkpoint_hash": checkpoint_hash,
        "fusion_rule": "L2-normalized average of projected text and projected statistical embeddings",
        "statistical_view_scope": config.statistical_view_scope,
        "anchor_features": sorted(anchor_frame["feature_name"].astype(str).tolist()),
    }
    anchor_path = write_json(output_dir / "learned_anchor_manifest.json", anchor_manifest)
    return {
        "homecredit_joint_embeddings": home_path,
        "lendingclub_v2_joint_embeddings": lc_path,
        "homecredit_learned_scores": home_score_path,
        "lendingclub_v2_learned_scores": lc_score_path,
        "learned_anchor_manifest": anchor_path,
        "anchor_manifest": anchor_manifest,
    }


def _joint(text_projection: torch.Tensor, statistical_projection: torch.Tensor) -> np.ndarray:
    joint = torch.nn.functional.normalize((text_projection + statistical_projection) / 2.0, p=2, dim=-1)
    return joint.detach().cpu().numpy().astype("float32")


def _embedding_frame(
    pairs: pd.DataFrame,
    text_projection: torch.Tensor,
    statistical_projection: torch.Tensor,
    joint: np.ndarray,
    checkpoint_hash: str,
    config: ClipTrainingConfig,
) -> pd.DataFrame:
    text_np = text_projection.detach().cpu().numpy().astype("float32")
    stat_np = statistical_projection.detach().cpu().numpy().astype("float32")
    frame = pairs[["dataset", "feature_name", "split", "semantic_group", "source_manifest_hash"]].copy()
    frame["projected_text_hash"] = [_vector_hash(row) for row in text_np]
    frame["projected_statistical_hash"] = [_vector_hash(row) for row in stat_np]
    frame["joint_embedding_hash"] = [_vector_hash(row) for row in joint]
    frame["checkpoint_hash"] = checkpoint_hash
    frame["statistical_view_scope"] = config.statistical_view_scope
    joint_cols = pd.DataFrame(joint, columns=[f"joint_{idx:04d}" for idx in range(joint.shape[1])])
    return pd.concat([frame.reset_index(drop=True), joint_cols], axis=1)


def _score_frame(embeddings: pd.DataFrame, centroid: np.ndarray, checkpoint_hash: str, anchor_hash: str, config: ClipTrainingConfig) -> pd.DataFrame:
    joint_cols = [col for col in embeddings.columns if str(col).startswith("joint_") and len(str(col)) == 10]
    values = embeddings[joint_cols].to_numpy(dtype=float)
    similarity = values @ centroid
    out = embeddings[
        [
            "dataset",
            "feature_name",
            "split",
            "projected_text_hash",
            "projected_statistical_hash",
            "joint_embedding_hash",
            "source_manifest_hash",
        ]
    ].copy()
    out["learned_similarity"] = similarity.astype(float)
    out["checkpoint_hash"] = checkpoint_hash
    out["anchor_hash"] = anchor_hash
    out["statistical_view_scope"] = config.statistical_view_scope
    out = out.sort_values(["learned_similarity", "feature_name"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    out["learned_rank"] = range(1, len(out) + 1)
    return out[
        [
            "dataset",
            "feature_name",
            "split",
            "learned_similarity",
            "learned_rank",
            "projected_text_hash",
            "projected_statistical_hash",
            "joint_embedding_hash",
            "checkpoint_hash",
            "anchor_hash",
            "source_manifest_hash",
            "statistical_view_scope",
        ]
    ]


def _training_anchor_names(homecredit_pairs: pd.DataFrame) -> set[str]:
    candidates = [
        Path("results/clip_v2/statistical_view/homecredit_statistical_anchor_features.csv"),
        Path("results/clip/statistical_baseline/homecredit_statistical_anchor_features.csv"),
    ]
    anchor_path = next((path for path in candidates if path.exists()), None)
    if anchor_path is None:
        raise RuntimeError(
            "no leakage-safe Home Credit statistical anchor feature artifact exists"
        )
    anchors = pd.read_csv(anchor_path)
    train_features = set(homecredit_pairs.loc[homecredit_pairs["split"].eq("train"), "feature_name"].astype(str))
    return set(anchors["feature_name"].astype(str)).intersection(train_features)


def _vector_hash(values: np.ndarray) -> str:
    return sha256_text(json.dumps([round(float(value), 10) for value in values.tolist()], sort_keys=False))
