from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import build_clip_v2_final_analysis as analysis


def _selected(dataset: str, selector: str, features: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset": dataset,
            "model": "catboost",
            "selector": selector,
            "feature_name": features,
            "final_selected": True,
            "final_rank": range(1, len(features) + 1),
            "clip_rank": range(1, len(features) + 1),
            "semantic_group": ["credit"] * len(features),
            "source_table_or_formula": ["src"] * len(features),
        }
    )


def test_semantic_map_uses_common_text_pca_and_is_deterministic(tmp_path, monkeypatch):
    datasets = ["homecredit", "lendingclub_v2"]
    feature_rows = {}
    for dataset in datasets:
        feature_rows[dataset] = pd.DataFrame(
            {
                "dataset": dataset,
                "feature_name": [f"{dataset}_f{i}" for i in range(5)],
                "semantic_group": ["credit", "credit", "income", "income", "delinq"],
                "embedding_cache_key": [f"{dataset}_hash_{i}" for i in range(5)],
                "embedding_0000": [0.0, 1.0, 2.0, 3.0, 4.0],
                "embedding_0001": [1.0, 1.0, 0.0, 0.0, 2.0],
                "embedding_0002": [0.5, 0.0, 0.5, 1.0, 1.5],
            }
        )
    v1 = pd.concat(
        [
            _selected("homecredit", "clip", ["homecredit_f0", "homecredit_f1"]),
            _selected("homecredit", "clip_then_mrmr", ["homecredit_f1", "homecredit_f2"]),
            _selected("lendingclub_v2", "clip", ["lendingclub_v2_f0", "lendingclub_v2_f1"]),
            _selected("lendingclub_v2", "clip_then_mrmr", ["lendingclub_v2_f1", "lendingclub_v2_f2"]),
        ],
        ignore_index=True,
    )
    v2 = pd.concat(
        [
            _selected("homecredit", "clip_v2", ["homecredit_f1", "homecredit_f3"]),
            _selected("homecredit", "clip_v2_then_mrmr", ["homecredit_f2", "homecredit_f3"]),
            _selected("lendingclub_v2", "clip_v2", ["lendingclub_v2_f1", "lendingclub_v2_f3"]),
            _selected("lendingclub_v2", "clip_v2_then_mrmr", ["lendingclub_v2_f2", "lendingclub_v2_f3"]),
        ],
        ignore_index=True,
    )

    def fake_read_parquet(path):
        text = str(path)
        if "homecredit" in text:
            return feature_rows["homecredit"].copy()
        return feature_rows["lendingclub_v2"].copy()

    def fake_read_csv(path, *args, **kwargs):
        text = str(path)
        if "results/clip/final_evaluation" in text:
            return v1.copy()
        if "selected_features_long" in text:
            return v2.copy()
        raise FileNotFoundError(text)

    agg_root = tmp_path / "final_evaluation"
    agg_root.mkdir()
    (agg_root / "selected_features_long.csv").write_text("placeholder\n", encoding="utf-8")
    monkeypatch.setattr(analysis, "ANALYSIS_ROOT", tmp_path / "analysis")
    monkeypatch.setattr(analysis, "AGG_ROOT", agg_root)
    monkeypatch.setattr(analysis.pd, "read_parquet", fake_read_parquet)
    monkeypatch.setattr(analysis.pd, "read_csv", fake_read_csv)

    first, plot = analysis.build_feature_semantic_map()
    second, _ = analysis.build_feature_semantic_map()

    assert plot.exists()
    assert len(first) == 10
    assert first[["dataset", "feature_name", "pca_1", "pca_2"]].equals(second[["dataset", "feature_name", "pca_1", "pca_2"]])
    assert set(first.columns).issuperset(
        {
            "selected_by_clip_v1",
            "selected_by_clip_v2",
            "selected_by_clip_v1_then_mrmr",
            "selected_by_clip_v2_then_mrmr",
            "clip_v1_rank",
            "clip_v2_rank",
            "source_table",
            "base_family",
        }
    )
    assert "y_true" not in first.columns
    home = first[first["dataset"].eq("homecredit")]
    v1_set = set(home.loc[home["selected_by_clip_v1"], "feature_name"])
    v2_set = set(home.loc[home["selected_by_clip_v2"], "feature_name"])
    assert len(v1_set & v2_set) / len(v1_set | v2_set) == pytest.approx(1 / 3)
    assert int(home["selected_by_clip_v1"].sum()) == 2
    assert int(home["selected_by_clip_v2"].sum()) == 2


def test_semantic_map_fails_when_required_selections_are_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(analysis, "ANALYSIS_ROOT", tmp_path / "analysis")
    monkeypatch.setattr(analysis.pd, "read_csv", lambda *args, **kwargs: pd.DataFrame(columns=["dataset", "model", "selector", "feature_name"]))

    with pytest.raises(RuntimeError, match="missing CLIP-v2 selected features|missing CLIP-v1 or CLIP-v2"):
        analysis.build_feature_semantic_map()


def test_markdown_table_does_not_require_tabulate():
    table = analysis._markdown_table(pd.DataFrame({"limitation": ["a|b"], "impact": ["line\nbreak"]}))

    assert "| limitation | impact |" in table
    assert "a\\|b" in table
    assert "line break" in table
