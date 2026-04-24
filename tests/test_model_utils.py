import pandas as pd

import Models.utils as model_utils


def test_train_model_does_not_forward_external_eval_set_by_default(monkeypatch):
    class DummyCatBoostModel:
        def __init__(self):
            self.received_eval_set = "unset"

        def fit(self, X, y, eval_set=None):
            self.received_eval_set = eval_set
            return self

        def predict_proba(self, X):
            return pd.Series([0.5] * len(X))

        def save(self, path):
            return None

    monkeypatch.setattr(model_utils, "CatBoostModel", DummyCatBoostModel)

    get_model, train_model, _, _ = model_utils.get_model_bundle("catboost")
    model = get_model()

    X_train = pd.DataFrame({"feature": [0.0, 1.0, 0.0]})
    y_train = pd.Series([0, 1, 0])
    X_val = pd.DataFrame({"feature": [1.0]})
    y_val = pd.Series([1])

    train_model(model, X_train, y_train, X_val, y_val)

    assert model.received_eval_set is None
