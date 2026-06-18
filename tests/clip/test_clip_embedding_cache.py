from __future__ import annotations

import numpy as np
import pandas as pd
import re

from credit_risk_fs.clip.embedding_cache import EmbeddingCacheSpec, build_embedding_frame, embedding_cache_key


def test_cache_identity_is_deterministic_and_changes_with_text_or_model():
    spec = EmbeddingCacheSpec("model-a", "main", True, "feature_text_v1")
    key = embedding_cache_key(dataset="homecredit", feature_name="f", feature_text_hash="abc", spec=spec)

    assert key == embedding_cache_key(dataset="homecredit", feature_name="f", feature_text_hash="abc", spec=spec)
    assert key != embedding_cache_key(dataset="homecredit", feature_name="f", feature_text_hash="def", spec=spec)
    assert key != embedding_cache_key(
        dataset="homecredit",
        feature_name="f",
        feature_text_hash="abc",
        spec=EmbeddingCacheSpec("model-b", "main", True, "feature_text_v1"),
    )
    assert key != embedding_cache_key(
        dataset="homecredit",
        feature_name="f",
        feature_text_hash="abc",
        spec=EmbeddingCacheSpec("model-a", "rev2", True, "feature_text_v1"),
    )


def test_embedding_feature_alignment_is_explicit():
    text = pd.DataFrame(
        {
            "dataset": ["homecredit", "homecredit"],
            "feature_name": ["a", "b"],
            "feature_text_hash": ["ha", "hb"],
            "source_manifest_hash": ["m", "m"],
            "text_template_version": ["v", "v"],
        }
    )
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    frame = build_embedding_frame(
        text_frame=text,
        embeddings=embeddings,
        spec=EmbeddingCacheSpec("mock", "test", True, "v"),
    )

    assert list(frame["feature_name"]) == ["a", "b"]
    assert ["embedding_0000", "embedding_0001"] == [
        col for col in frame.columns if re.fullmatch(r"embedding_\d{4}", col)
    ]
