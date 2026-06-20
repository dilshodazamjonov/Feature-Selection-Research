from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from credit_risk_fs.clip.selector_validation import ClipSelectorConfig, FUSION_RULE
from credit_risk_fs.utils.hashing import sha256_text


SCORE_CACHE_VERSION = "clip_selector_score_cache_v1"


def score_cache_key(
    *,
    dataset: str,
    feature_name: str,
    checkpoint_hash: str,
    anchor_hash: str,
    text_embedding_hash: str,
    statistical_vector_hash: str,
    preprocessor_hash: str,
    fusion_rule: str,
    statistical_view_scope: str,
    code_version: str,
) -> str:
    return sha256_text(
        "|".join(
            [
                SCORE_CACHE_VERSION,
                dataset,
                feature_name,
                checkpoint_hash,
                anchor_hash,
                text_embedding_hash,
                statistical_vector_hash,
                preprocessor_hash,
                fusion_rule,
                statistical_view_scope,
                code_version,
            ]
        )
    )


def build_score_cache_frame(
    *,
    scores: pd.DataFrame,
    joint_embeddings: pd.DataFrame,
    config: ClipSelectorConfig,
    checkpoint_hash: str,
    anchor_hash: str,
    preprocessor_hash: str,
    code_version: str,
) -> pd.DataFrame:
    merged = scores.merge(
        joint_embeddings[
            [
                "dataset",
                "feature_name",
                "projected_text_hash",
                "projected_statistical_hash",
                "joint_embedding_hash",
                "checkpoint_hash",
                "statistical_view_scope",
            ]
        ],
        on=["dataset", "feature_name", "projected_text_hash", "projected_statistical_hash", "joint_embedding_hash", "checkpoint_hash", "statistical_view_scope"],
        how="inner",
    )
    if len(merged) != len(scores):
        raise RuntimeError("score cache alignment failed")
    merged["score_cache_key"] = [
        score_cache_key(
            dataset=str(row.dataset),
            feature_name=str(row.feature_name),
            checkpoint_hash=checkpoint_hash,
            anchor_hash=anchor_hash,
            text_embedding_hash=str(row.projected_text_hash),
            statistical_vector_hash=str(row.projected_statistical_hash),
            preprocessor_hash=preprocessor_hash,
            fusion_rule=FUSION_RULE,
            statistical_view_scope=str(row.statistical_view_scope),
            code_version=code_version,
        )
        for row in merged.itertuples(index=False)
    ]
    merged["score_cache_version"] = SCORE_CACHE_VERSION
    merged["fusion_rule"] = FUSION_RULE
    merged["preprocessor_hash"] = preprocessor_hash
    merged["code_version"] = code_version
    return merged.sort_values(["dataset", "learned_rank", "feature_name"], kind="mergesort").reset_index(drop=True)


def write_score_cache(frame: pd.DataFrame, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return output


def validate_score_cache(frame: pd.DataFrame, *, checkpoint_hash: str, anchor_hash: str) -> None:
    required = {
        "dataset",
        "score_cache_key",
        "score_cache_version",
        "checkpoint_hash",
        "anchor_hash",
        "feature_name",
        "learned_similarity",
        "learned_rank",
        "source_manifest_hash",
        "statistical_view_scope",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"CLIP score cache missing columns: {missing}")
    if frame["feature_name"].duplicated().any():
        raise RuntimeError("CLIP score cache has duplicate feature names")
    if frame["score_cache_key"].duplicated().any():
        raise RuntimeError("CLIP score cache has duplicate cache keys")
    if not frame["checkpoint_hash"].astype(str).eq(checkpoint_hash).all():
        raise RuntimeError("stale CLIP score cache: checkpoint hash mismatch")
    if not frame["anchor_hash"].astype(str).eq(anchor_hash).all():
        raise RuntimeError("stale CLIP score cache: anchor hash mismatch")
    if not frame["score_cache_version"].astype(str).eq(SCORE_CACHE_VERSION).all():
        raise RuntimeError("stale CLIP score cache: version mismatch")
    if not np.isfinite(pd.to_numeric(frame["learned_similarity"], errors="coerce")).all():
        raise RuntimeError("CLIP score cache contains non-finite learned scores")


def select_cache_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[
        [
            "dataset",
            "feature_name",
            "learned_similarity",
            "learned_rank",
            "projected_text_hash",
            "projected_statistical_hash",
            "joint_embedding_hash",
            "checkpoint_hash",
            "anchor_hash",
            "source_manifest_hash",
            "statistical_view_scope",
            "score_cache_key",
            "score_cache_version",
            "fusion_rule",
            "preprocessor_hash",
            "code_version",
        ]
    ].copy()
