from __future__ import annotations

import pandas as pd

from scripts.run_clip_final_evaluation import BASELINE_SELECTORS


def test_baseline_selector_allowlist_is_focused():
    assert set(BASELINE_SELECTORS) == {"mrmr", "llm", "llm_then_mrmr", "stable_core_llm_fill"}


def test_baseline_rows_are_loaded_from_canonical_shape(tmp_path, monkeypatch):
    root = tmp_path / "results" / "homecredit"
    root.mkdir(parents=True)
    pd.DataFrame(
        [
            {"dataset_name": "homecredit", "model": "lr", "selector": "mrmr", "output_folder": "x"},
            {"dataset_name": "homecredit", "model": "lr", "selector": "pca", "output_folder": "y"},
        ]
    ).to_csv(root / "final_comparison_table.csv", index=False)
    monkeypatch.chdir(tmp_path)

    from scripts.run_clip_final_evaluation import _load_baselines

    loaded = _load_baselines(["homecredit"])

    assert loaded["selector"].tolist() == ["mrmr"]
    assert loaded["source"].tolist() == ["frozen_baseline"]
