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
        from feature_selection.boruta_rfe import BorutaRFESelector

        return BorutaRFESelector, {
            "boruta_kwargs": {"max_iter": 20, "random_state": 42},
            "rfe_kwargs": {"n_features": 40, "step": 10, "random_state": 42},
        }

    elif name == "mrmr":
        from feature_selection.mrmr import MRMR

        return MRMR, {
            "k": 50,
            "method": "mrmr",
            "random_state": 42,
        }

    elif name == "pca":
        from feature_selection.pca import PCASelector

        return PCASelector, {
            "n_components": 0.95,
            "save_dir": None,
        }

    elif name == "llm":
        from feature_selection.llm_selector import LLMSelector

        return LLMSelector, {
            "description_csv_path": None,
            "cache_dir": "outputs/llm_selector_cache",
            "model": "gpt-4.1-mini",
            "temperature": 0.0,
            "max_features": 50,
            "max_missing_rate": 0.95,
            "iv_filter_kwargs": {
                "min_iv": 0.01,
                "max_iv_for_leakage": 0.5,
                "encode": True,
                "n_jobs": 1,
                "verbose": False,
            },
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


def get_model_bundle(model_name, model_kwargs=None):
    """
    Returns model factory and adapter functions.
    """
    name = model_name.lower()
    model_kwargs = dict(model_kwargs or {})

    if name == "catboost":
        from Models.catboost_model import CatBoostModel

        model_cls = CatBoostModel
    elif name == "rf":
        from Models.random_forest_model import RandomForestModel

        model_cls = RandomForestModel
    elif name == "lr":
        from Models.logistic_regression_model import LogisticRegressionModel

        model_cls = LogisticRegressionModel
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    def get_model():
        return model_cls(**model_kwargs)

    def train_model(model, X_train, y_train, X_val=None, y_val=None):
        # Keep held-out fold data strictly for evaluation unless a model
        # explicitly opts in to consuming an external eval set.
        use_external_eval_set = bool(getattr(model, "supports_external_eval_set", False))
        eval_set = (
            (X_val, y_val)
            if use_external_eval_set and X_val is not None and y_val is not None
            else None
        )
        return model.fit(X_train, y_train, eval_set=eval_set)

    def predict_proba(model, X):
        return model.predict_proba(X)

    def save_model(model, path):
        return model.save(path)

    return get_model, train_model, predict_proba, save_model
