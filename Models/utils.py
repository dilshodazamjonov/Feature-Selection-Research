CatBoostModel = None
RandomForestModel = None
LogisticRegressionModel = None


def get_selector(selector_name: str):
    """
    Returns the selector class and its default kwargs.

    Supported selectors:
        - 'boruta' / 'boruta_rfe' -> Boruta, with optional RFE refinement
        - 'mrmr' -> Minimum Redundancy Maximum Relevance
        - 'pca' -> Principal Component Analysis
        - 'llm' -> LLM-based feature selection
        - 'domain_rule_baseline' -> lightweight domain-rule screener
        - 'none' -> No feature selection
    """

    name = selector_name.lower()

    if name in ("boruta", "boruta_rfe"):
        from feature_selection.boruta_rfe import BorutaRFESelector

        return BorutaRFESelector, {
            "boruta_kwargs": {"max_iter": 15, "random_state": 42},
            "rfe_kwargs": {"n_features": 40, "step": 10, "random_state": 42},
            "use_rfe": False,
            "n_features": 40,
        }

    if name == "mrmr":
        from feature_selection.mrmr import MRMR

        return MRMR, {
            "k": 50,
            "method": "mrmr",
            "random_state": 42,
        }

    if name == "pca":
        from feature_selection.pca import PCASelector

        return PCASelector, {
            "n_components": 0.95,
            "save_dir": None,
            "random_state": 42,
        }

    if name == "llm":
        from feature_selection.llm_selector import LLMSelector

        return LLMSelector, {
            "description_csv_path": None,
            "cache_dir": "results/_llm_rankings_cache",
            "model": "gpt-4.1-mini",
            "temperature": 0.0,
            "max_features": 100,
            "max_missing_rate": 0.95,
            "iv_filter_kwargs": {
                "min_iv": 0.01,
                "max_iv_for_leakage": 0.5,
                "encode": True,
                "n_jobs": 1,
                "verbose": False,
            },
        }

    if name == "domain_rule_baseline":
        from feature_selection.hybrid import DomainRuleBaselineSelector

        return DomainRuleBaselineSelector, {
            "description_csv_path": None,
            "feature_budget": 40,
        }

    if name == "none":
        return None, {}

    raise ValueError(
        f"Unsupported selector: {selector_name}. "
        f"Available: boruta, boruta_rfe, mrmr, pca, llm, domain_rule_baseline, none"
    )


def get_model_bundle(model_name, model_kwargs=None):
    """
    Returns model factory and adapter functions.
    """
    name = model_name.lower()
    model_kwargs = dict(model_kwargs or {})

    if name == "catboost":
        global CatBoostModel
        if CatBoostModel is None:
            from Models.catboost_model import CatBoostModel as _CatBoostModel

            CatBoostModel = _CatBoostModel

        model_cls = CatBoostModel
    elif name == "rf":
        global RandomForestModel
        if RandomForestModel is None:
            from Models.random_forest_model import RandomForestModel as _RandomForestModel

            RandomForestModel = _RandomForestModel

        model_cls = RandomForestModel
    elif name == "lr":
        global LogisticRegressionModel
        if LogisticRegressionModel is None:
            from Models.logistic_regression_model import LogisticRegressionModel as _LogisticRegressionModel

            LogisticRegressionModel = _LogisticRegressionModel

        model_cls = LogisticRegressionModel
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    def get_model():
        return model_cls(**model_kwargs)

    def train_model(model, X_train, y_train, X_val=None, y_val=None):
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
