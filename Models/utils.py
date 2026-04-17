
from Models.catboost_model import CatBoostModel
from Models.random_forest_model import RandomForestModel
from Models.logistic_regression_model import LogisticRegressionModel

from feature_selection.boruta_rfe import BorutaRFESelector
from feature_selection.mrmr import MRMR
from feature_selection.pca import PCASelector
from feature_selection.llm_selector import LLMSelector


def get_selector(selector_name: str):
    """
    Returns the selector class and its default kwargs.

    Supported selectors:
        - 'boruta' / 'boruta_rfe' -> Boruta + RFE
        - 'mrmr' -> Minimum Redundancy Maximum Relevance
        - 'pca' -> Principal Component Analysis
        - 'llm' -> LLM-based feature selection
        - 'llm_mrmr' -> Hybrid (LLM → mRMR)
        - 'none' -> No feature selection
    """

    name = selector_name.lower()

    if name in ("boruta", "boruta_rfe"):
        return BorutaRFESelector, {
            "boruta_kwargs": {"max_iter": 20, "random_state": 42},
            "rfe_kwargs": {"n_features": 40, "step": 10, "random_state": 42},
        }

    elif name == "mrmr":
        return MRMR, {
            "k": 50,
            "method": "mrmr",
            "random_state": 42,
        }

    elif name == "pca":
        return PCASelector, {
            "n_components": 0.95,
            "save_dir": None,
        }

    elif name == "llm":
        return LLMSelector, {
            # MUST be passed from outside or filled later
            "feature_metadata": None,
            "cache_path": "outputs/llm_selected_features.json",
            "model": "gpt-4.1-mini",
            "temperature": 0.0
        }

    # TODO  

    # elif name == "llm_mrmr":
    #     return LLM_MRMR_Selector, {
    #         "llm_selector": None,   # will be injected
    #         "mrmr_selector": None, # will be injected
    #     }

    elif name == "none":
        return None, {}

    else:
        raise ValueError(
            f"Unsupported selector: {selector_name}. "
            f"Available: boruta, boruta_rfe, mrmr, pca, llm, llm_mrmr, none"
        )


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