from __future__ import annotations

from scripts.run_clip_final_evaluation import _feature_set_hash


def test_feature_set_hash_is_order_stable():
    assert _feature_set_hash(["b", "a"]) == _feature_set_hash(["a", "b"])
