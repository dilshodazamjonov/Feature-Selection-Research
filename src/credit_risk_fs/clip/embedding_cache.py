from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd

from credit_risk_fs.utils.hashing import sha256_text
from credit_risk_fs.utils.io import write_json


@dataclass(frozen=True)
class EmbeddingCacheSpec:
    encoder_model: str
    encoder_revision: str
    normalize_embeddings: bool
    text_template_version: str


def embedding_cache_key(
    *,
    dataset: str,
    feature_name: str,
    feature_text_hash: str,
    spec: EmbeddingCacheSpec,
) -> str:
    raw = "|".join(
        [
            dataset,
            feature_name,
            feature_text_hash,
            spec.encoder_model,
            spec.encoder_revision,
            str(bool(spec.normalize_embeddings)),
            spec.text_template_version,
        ]
    )
    return sha256_text(raw)


def build_embedding_frame(
    *,
    text_frame: pd.DataFrame,
    embeddings: np.ndarray,
    spec: EmbeddingCacheSpec,
) -> pd.DataFrame:
    if len(text_frame) != int(embeddings.shape[0]):
        raise ValueError("embedding row count does not match text frame row count")
    records = text_frame[
        ["dataset", "feature_name", "feature_text_hash", "source_manifest_hash", "text_template_version"]
    ].copy()
    records["encoder_model"] = spec.encoder_model
    records["encoder_revision"] = spec.encoder_revision
    records["normalize_embeddings"] = bool(spec.normalize_embeddings)
    records["embedding_cache_key"] = [
        embedding_cache_key(
            dataset=str(row["dataset"]),
            feature_name=str(row["feature_name"]),
            feature_text_hash=str(row["feature_text_hash"]),
            spec=spec,
        )
        for row in records.to_dict("records")
    ]
    embedding_columns = pd.DataFrame(
        embeddings,
        columns=[f"embedding_{idx:04d}" for idx in range(int(embeddings.shape[1]))],
        index=records.index,
    )
    frame = pd.concat([records, embedding_columns], axis=1)
    return frame.sort_values(["dataset", "feature_name"], kind="mergesort").reset_index(drop=True)


def save_embedding_frame(frame: pd.DataFrame, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_parquet(output, index=False)
    except ImportError as exc:
        raise RuntimeError(
            "Parquet output requires pyarrow or fastparquet. Install a parquet engine before generating embeddings."
        ) from exc
    return output


def load_embedding_frame(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def write_embedding_cache_manifest(path: str | Path, *, frames: Iterable[pd.DataFrame], spec: EmbeddingCacheSpec) -> Path:
    payload = {
        "encoder_model": spec.encoder_model,
        "encoder_revision": spec.encoder_revision,
        "normalize_embeddings": spec.normalize_embeddings,
        "text_template_version": spec.text_template_version,
        "datasets": {},
    }
    for frame in frames:
        dataset = str(frame["dataset"].iloc[0]) if len(frame) else "unknown"
        payload["datasets"][dataset] = {
            "row_count": int(len(frame)),
            "embedding_dimension": len(_embedding_columns(frame)),
            "cache_key_count": int(frame["embedding_cache_key"].nunique()) if "embedding_cache_key" in frame else 0,
        }
    return write_json(path, payload)


def _embedding_columns(frame: pd.DataFrame) -> list[str]:
    return [col for col in frame.columns if re.fullmatch(r"embedding_\d{4}", str(col))]
