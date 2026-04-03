
from Models.catboost_model import CatBoostModel
from Models.random_forest_model import RandomForestModel
from Models.logistic_regression_model import LogisticRegressionModel

from feature_selection.boruta_rfe import BorutaRFESelector
from feature_selection.mrmr import MRMR
from feature_selection.pca import PCASelector



def get_selector(selector_name):
    """
    Returns the selector class and its default kwargs.

    Updated logic:
        - 'boruta' -> Boruta + RFE pipeline
        - 'rfe' removed
    """
    name = selector_name.lower()

    if name == "boruta_rfe":
        return BorutaRFESelector, {
            "boruta_kwargs": {"max_iter": 20, "random_state": 42},
            "rfe_kwargs": {"n_features": 40, "step": 10, "random_state": 42},
        }

    if name == "mrmr":
        return MRMR, {"k": 50, "method": "mrmr", "random_state": 42}

    if name == "pca":
        return PCASelector, {"n_components": 0.95, "save_dir": None}

    if name == "none":
        return None, {}

    raise ValueError(f"Unsupported selector: {selector_name}")


def get_model_bundle(model_name):
    """
    Returns model factory and adapter functions.
    """
    name = model_name.lower()

    if name == "catboost":
        model_cls = CatBoostModel
    elif name == "rf":
        model_cls = RandomForestModel
    elif name == "lr":
        model_cls = LogisticRegressionModel
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    def get_model():
        return model_cls()

    def train_model(model, X_train, y_train, X_val=None, y_val=None):
        eval_set = (X_val, y_val) if X_val is not None and y_val is not None else None
        return model.fit(X_train, y_train, eval_set=eval_set)

    def predict_proba(model, X):
        return model.predict_proba(X)

    def save_model(model, path):
        return model.save(path)

    return get_model, train_model, predict_proba, save_model