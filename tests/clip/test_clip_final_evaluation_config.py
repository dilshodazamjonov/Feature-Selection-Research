from __future__ import annotations

import pytest

from scripts.run_clip_final_evaluation import ACTIVE_DATASETS, LEGACY_DATASET, _run_specs, _selected_datasets


def test_clip_final_evaluation_only_uses_active_datasets():
    specs = _run_specs(list(ACTIVE_DATASETS))

    assert {spec["dataset"] for spec in specs} == {"homecredit", "lendingclub_v2"}
    assert LEGACY_DATASET not in {spec["dataset"] for spec in specs}
    assert len(specs) == 8


def test_clip_final_evaluation_rejects_legacy_lendingclub():
    class Args:
        dataset = "lendingclub"
        all = False
        dry_run = False

    with pytest.raises(RuntimeError, match="legacy LendingClub"):
        _selected_datasets(Args())
