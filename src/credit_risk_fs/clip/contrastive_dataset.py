from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from credit_risk_fs.clip.pair_validation import embedding_columns, statistical_columns


class ContrastiveFeatureDataset:
    """Lazy research dataset over positive contrastive pair indexes."""

    def __init__(
        self,
        *,
        pairs_path: str | Path,
        text_embeddings_path: str | Path,
        statistical_vectors_path: str | Path,
        mode: str,
    ) -> None:
        if mode not in {"train", "validation", "external"}:
            raise ValueError("mode must be train, validation, or external")
        self.mode = mode
        self.pairs = pd.read_parquet(pairs_path).sort_values(["dataset", "feature_name"], kind="mergesort").reset_index(drop=True)
        self.text = pd.read_parquet(text_embeddings_path)
        self.stat = pd.read_parquet(statistical_vectors_path)
        if mode == "train" and not self.pairs["dataset"].eq("homecredit").all():
            raise ValueError("training dataset can only be built from HomeCredit pairs")
        if mode == "train" and not self.pairs["allowed_for_training"].all():
            raise ValueError("train mode received pairs not allowed for training")
        if mode == "validation" and not self.pairs["allowed_for_validation"].all():
            raise ValueError("validation mode received pairs not allowed for validation")
        if mode == "external" and not self.pairs["allowed_for_external_evaluation"].all():
            raise ValueError("external mode received pairs not allowed for external evaluation")
        self._text_cols = embedding_columns(self.text)
        self._stat_cols = statistical_columns(self.stat)
        self._text_by_key = self.text.set_index("embedding_cache_key", drop=False)
        self._stat_by_key = self.stat.set_index("stable_row_id", drop=False)

    def __len__(self) -> int:
        return int(len(self.pairs))

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.pairs.iloc[int(index)]
        text_row = self._text_by_key.loc[str(row["text_embedding_row_id"])]
        stat_row = self._stat_by_key.loc[str(row["statistical_vector_row_id"])]
        if str(text_row["feature_name"]) != str(row["feature_name"]):
            raise ValueError("text embedding feature alignment failed")
        if str(stat_row["feature_name"]) != str(row["feature_name"]):
            raise ValueError("statistical vector feature alignment failed")
        if str(text_row["feature_text_hash"]) != str(row["text_hash"]):
            raise ValueError("text hash alignment failed")
        if str(stat_row["statistical_vector_hash"]) != str(row["statistical_vector_hash"]):
            raise ValueError("statistical vector hash alignment failed")
        text_embedding = text_row[self._text_cols].to_numpy(dtype=np.float32)
        statistical_vector = stat_row[self._stat_cols].to_numpy(dtype=np.float32)
        if not np.isfinite(text_embedding).all() or not np.isfinite(statistical_vector).all():
            raise ValueError("non-finite contrastive tensors")
        return {
            "text_embedding": text_embedding,
            "statistical_vector": statistical_vector,
            "feature_name": str(row["feature_name"]),
            "pair_id": str(row["pair_id"]),
            "negative_exclusion_key": str(row["base_feature_family"]),
            "metadata": {
                "dataset": str(row["dataset"]),
                "split": str(row["split"]),
                "group_key": str(row["group_key"]),
                "semantic_group": str(row["semantic_group"]),
                "source_table_or_formula": str(row["source_table_or_formula"]),
                "pair_role": str(row["pair_role"]),
            },
        }
