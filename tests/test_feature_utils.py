import pandas as pd

from Models.logistic_regression_model import LogisticRegressionModel
from evaluation.feature_utils import _extract_feature_importance


def test_extract_feature_importance_unwraps_model_wrapper():
    X = pd.DataFrame(
        {
            "feature_a": [0.0, 0.0, 1.0, 1.0],
            "feature_b": [0.0, 1.0, 0.0, 1.0],
        }
    )
    y = pd.Series([0, 0, 1, 1])

    model = LogisticRegressionModel()
    model.fit(X, y)

    importance_df = _extract_feature_importance(model, X.columns.tolist())

    assert list(importance_df.columns) == ["feature", "importance"]
    assert set(importance_df["feature"]) == {"feature_a", "feature_b"}
    assert (importance_df["importance"] >= 0).all()
