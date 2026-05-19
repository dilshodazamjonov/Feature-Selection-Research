CatBoostModel = None
RandomForestModel = None
LogisticRegressionModel = None


from credit_risk_fs.selectors.registry import get_selector


def get_model_bundle(model_name, model_kwargs=None):
    """
    Returns model factory and adapter functions.
    """
    name = model_name.lower()
    model_kwargs = dict(model_kwargs or {})

    if name == "catboost":
        global CatBoostModel
        if CatBoostModel is None:
            from credit_risk_fs.models.catboost_model import CatBoostModel as _CatBoostModel

            CatBoostModel = _CatBoostModel

        model_cls = CatBoostModel
    elif name == "rf":
        global RandomForestModel
        if RandomForestModel is None:
            from credit_risk_fs.models.random_forest_model import (
                RandomForestModel as _RandomForestModel,
            )

            RandomForestModel = _RandomForestModel

        model_cls = RandomForestModel
    elif name == "lr":
        global LogisticRegressionModel
        if LogisticRegressionModel is None:
            from credit_risk_fs.models.logistic_regression import (
                LogisticRegressionModel as _LogisticRegressionModel,
            )

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
