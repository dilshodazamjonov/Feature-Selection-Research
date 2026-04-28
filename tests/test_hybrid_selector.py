import pandas as pd
import pytest

from feature_selection.hybrid import LLMThenStatSelector, StableCoreLLMFillSelector


class FakeLLMSelector:
    def __init__(self, **kwargs):
        self.feature_budget = kwargs.get("feature_budget")
        self.selected_features = None
        self.artifact_dir = None

    def set_artifact_dir(self, artifact_dir):
        self.artifact_dir = artifact_dir

    def fit(self, X, y=None):
        if self.feature_budget is None:
            self.selected_features = ["f1", "f3"]
        else:
            self.selected_features = X.columns[: int(self.feature_budget)].tolist()
        self.ranked_features_ = list(self.selected_features)
        return self

    def transform(self, X):
        return X[self.selected_features]

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class FakeStatSelector:
    def __init__(self, keep: int = 1):
        self.keep = keep
        self.selected_features = None

    def fit(self, X, y=None):
        self.selected_features = X.columns[: self.keep].tolist()
        return self

    def transform(self, X):
        return X[self.selected_features]

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


def test_llm_then_stat_selector_applies_both_stages(tmp_path):
    X = pd.DataFrame(
        {
            "f1": [1.0, 2.0, 3.0],
            "f2": [10.0, 11.0, 12.0],
            "f3": [4.0, 5.0, 6.0],
        }
    )
    y = pd.Series([0, 1, 0])

    selector = LLMThenStatSelector(
        description_csv_path="dummy.csv",
        stat_selector_cls=FakeStatSelector,
        stat_selector_kwargs={"keep": 1},
        cache_dir=str(tmp_path / "cache"),
        llm_selector_cls=FakeLLMSelector,
        llm_candidate_pool_budget=2,
        final_feature_budget=1,
    )
    selector.set_artifact_dir(tmp_path / "artifacts")

    transformed = selector.fit_transform(X, y)

    assert selector.llm_selected_features_ == ["f1", "f2"]
    assert selector.selected_features_ == ["f1"]
    assert list(transformed.columns) == ["f1"]
    assert not (tmp_path / "artifacts" / "llm_preselected_features.csv").exists()
    assert not (tmp_path / "artifacts" / "hybrid_selected_features.csv").exists()


def test_llm_then_stat_selector_requires_target():
    selector = LLMThenStatSelector(
        description_csv_path="dummy.csv",
        stat_selector_cls=FakeStatSelector,
        llm_selector_cls=FakeLLMSelector,
    )

    with pytest.raises(ValueError):
        selector.fit(pd.DataFrame({"f1": [1.0]}), None)


def test_lr_hybrid_uses_top60_candidate_pool_and_final20_output(tmp_path):
    X = pd.DataFrame({f"f{i}": [float(i), float(i + 1), float(i + 2)] for i in range(80)})
    y = pd.Series([0, 1, 0])

    selector = LLMThenStatSelector(
        description_csv_path="dummy.csv",
        stat_selector_cls=FakeStatSelector,
        stat_selector_kwargs={"keep": 20},
        cache_dir=str(tmp_path / "cache"),
        llm_selector_cls=FakeLLMSelector,
        llm_candidate_pool_budget=60,
        final_feature_budget=20,
    )

    transformed = selector.fit_transform(X, y)

    assert len(selector.llm_selected_features_) == 60
    assert len(selector.selected_features_) == 20
    assert list(transformed.columns) == [f"f{i}" for i in range(20)]


def test_catboost_hybrid_uses_top100_candidate_pool_and_final40_output(tmp_path):
    X = pd.DataFrame({f"f{i}": [float(i), float(i + 1), float(i + 2)] for i in range(120)})
    y = pd.Series([0, 1, 0])

    selector = LLMThenStatSelector(
        description_csv_path="dummy.csv",
        stat_selector_cls=FakeStatSelector,
        stat_selector_kwargs={"keep": 40},
        cache_dir=str(tmp_path / "cache"),
        llm_selector_cls=FakeLLMSelector,
        llm_candidate_pool_budget=100,
        final_feature_budget=40,
    )

    transformed = selector.fit_transform(X, y)

    assert len(selector.llm_selected_features_) == 100
    assert len(selector.selected_features_) == 40
    assert list(transformed.columns) == [f"f{i}" for i in range(40)]


def test_hybrid_output_never_exceeds_final_budget(tmp_path):
    X = pd.DataFrame({f"f{i}": [float(i), float(i + 1), float(i + 2)] for i in range(40)})
    y = pd.Series([0, 1, 0])

    selector = LLMThenStatSelector(
        description_csv_path="dummy.csv",
        stat_selector_cls=FakeStatSelector,
        stat_selector_kwargs={"keep": 30},
        cache_dir=str(tmp_path / "cache"),
        llm_selector_cls=FakeLLMSelector,
        llm_candidate_pool_budget=30,
        final_feature_budget=20,
    )

    transformed = selector.fit_transform(X, y)

    assert len(selector.selected_features_) == 20
    assert transformed.shape[1] == 20


def test_boruta_like_hybrid_backfills_to_budget_from_llm_order(tmp_path):
    X = pd.DataFrame({f"f{i}": [float(i), float(i + 1), float(i + 2)] for i in range(30)})
    y = pd.Series([0, 1, 0])

    selector = LLMThenStatSelector(
        description_csv_path="dummy.csv",
        stat_selector_cls=FakeStatSelector,
        stat_selector_kwargs={"keep": 5},
        cache_dir=str(tmp_path / "cache"),
        llm_selector_cls=FakeLLMSelector,
        llm_candidate_pool_budget=20,
        final_feature_budget=20,
    )

    transformed = selector.fit_transform(X, y)

    assert len(selector.selected_features_) == 20
    assert list(transformed.columns) == [f"f{i}" for i in range(20)]


def test_stable_core_llm_fill_runs_and_respects_budget(tmp_path):
    X = pd.DataFrame({f"f{i}": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0] for i in range(12)})
    X["signal"] = [0.1, 0.9, 0.2, 0.8, 0.3, 0.7]
    y = pd.Series([0, 1, 0, 1, 0, 1])

    selector = StableCoreLLMFillSelector(
        description_csv_path="dummy.csv",
        cache_dir=str(tmp_path / "cache"),
        llm_selector_cls=FakeLLMSelector,
        final_feature_budget=5,
        bootstrap_iterations=3,
        bootstrap_fraction=1.0,
        random_state=7,
    )

    selector.fit(X, y)
    X_selected = selector.fit_postprocess(X, y)

    assert len(selector.selected_features_) == 5
    assert X_selected.shape[1] == 5
