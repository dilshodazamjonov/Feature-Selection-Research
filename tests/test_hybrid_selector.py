import pandas as pd
import pytest

from feature_selection.hybrid import LLMThenStatSelector


class FakeLLMSelector:
    def __init__(self, **kwargs):
        self.selected_features = None
        self.artifact_dir = None

    def set_artifact_dir(self, artifact_dir):
        self.artifact_dir = artifact_dir

    def fit(self, X, y=None):
        self.selected_features = ["f1", "f3"]
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
    )
    selector.set_artifact_dir(tmp_path / "artifacts")

    transformed = selector.fit_transform(X, y)

    assert selector.llm_selected_features_ == ["f1", "f3"]
    assert selector.selected_features_ == ["f1"]
    assert list(transformed.columns) == ["f1"]
    assert (tmp_path / "artifacts" / "llm_preselected_features.csv").exists()
    assert (tmp_path / "artifacts" / "hybrid_selected_features.csv").exists()


def test_llm_then_stat_selector_requires_target():
    selector = LLMThenStatSelector(
        description_csv_path="dummy.csv",
        stat_selector_cls=FakeStatSelector,
        llm_selector_cls=FakeLLMSelector,
    )

    with pytest.raises(ValueError):
        selector.fit(pd.DataFrame({"f1": [1.0]}), None)
