from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_risk_fs.clip.selector_adapter import ClipScoreAdapter
from credit_risk_fs.selectors.clip_screening import ClipScreeningSelector
from credit_risk_fs.selectors.clip_then_mrmr import ClipThenMRMRSelector


def _candidate_frame(dataset: str = "homecredit", *, n_features: int = 30, n_rows: int = 48) -> tuple[pd.DataFrame, pd.Series]:
    scores = ClipScoreAdapter(dataset=dataset).score_frame(use_cache=False)
    features = (
        scores.sort_values(["learned_rank", "feature_name"], kind="mergesort")["feature_name"]
        .astype(str)
        .head(n_features)
        .tolist()
    )
    rng = np.random.default_rng(123)
    X = pd.DataFrame(rng.normal(size=(n_rows, len(features))), columns=features)
    y = pd.Series((X.iloc[:, :5].sum(axis=1) > 0).astype(int), name="TARGET")
    return X, y


def test_clip_selector_selects_budgeted_top_features():
    X, y = _candidate_frame(n_features=25)
    selector = ClipScreeningSelector(feature_budget=7, model_name="lr")

    selected = selector.fit_transform(X, y)

    assert selected.shape[1] == 7
    assert selector.selection_manifest_ is not None
    assert selector.selection_manifest_["final_selected"].all()
    assert selector.selection_manifest_["model"].unique().tolist() == ["lr"]


def test_clip_then_mrmr_screens_before_mrmr_and_uses_dev_rows_only():
    X, y = _candidate_frame(n_features=24, n_rows=36)
    selector = ClipThenMRMRSelector(feature_budget=5, screening_pool_size=12, model_name="catboost")

    X_screened = selector.fit(X, y).transform(X)
    X_final = selector.fit_postprocess(X_screened, y)

    assert X_screened.shape[1] == 12
    assert X_final.shape[1] == 5
    assert selector.mrmr_input_row_count_ == 36
    assert selector.selection_manifest_ is not None
    assert int(selector.selection_manifest_["final_selected"].sum()) == 5


def test_duplicate_or_missing_candidates_are_rejected():
    adapter = ClipScoreAdapter(dataset="homecredit")
    frame = adapter.score_frame(use_cache=False)
    feature = str(frame.iloc[0]["feature_name"])

    with pytest.raises(RuntimeError, match="duplicate candidate"):
        adapter.rank_candidates([feature, feature])

    with pytest.raises(RuntimeError, match="missing CLIP scores"):
        adapter.rank_candidates([feature, "__not_scored__"])


def test_legacy_lendingclub_is_rejected():
    with pytest.raises(RuntimeError, match="legacy LendingClub"):
        ClipScoreAdapter(dataset="lendingclub").score_frame()
