from __future__ import annotations

import pytest


LEGACY_CLIP_V1_MODULES = {
    "test_clip_checkpointing.py",
    "test_clip_contrastive_dataset.py",
    "test_clip_final_analysis.py",
    "test_clip_learned_scoring.py",
    "test_clip_negative_policy.py",
    "test_clip_pair_builder.py",
    "test_clip_pair_validation.py",
    "test_clip_selector_behavior.py",
    "test_clip_selector_cache.py",
    "test_clip_selector_integration.py",
    "test_clip_training_boundaries.py",
    "test_clip_training_determinism.py",
    "test_clip_v1_freeze_and_v2_boundaries.py",
}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    marker = pytest.mark.legacy_clip_v1
    for item in items:
        if item.path.name in LEGACY_CLIP_V1_MODULES:
            item.add_marker(marker)
