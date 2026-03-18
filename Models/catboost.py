# models/catboost.py

from catboost import CatBoostClassifier


def get_catboost_model():
    return CatBoostClassifier(
        depth=10,
        learning_rate=0.01,
        l2_leaf_reg=95,
        min_data_in_leaf=290,
        colsample_bylevel=0.9,
        random_strength=0.125,
        grow_policy='Depthwise',
        one_hot_max_size=21,
        leaf_estimation_method='Newton',
        bootstrap_type='Bernoulli',
        subsample=0.55,
        loss_function='Logloss',
        eval_metric='AUC',
        auto_class_weights='Balanced',
        iterations=2200,
        early_stopping_rounds=150,
        verbose=100
    )


def train_catboost(model, X_train, y_train, X_val, y_val):
    model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val)
    )
    return model


def predict_proba(model, X):
    return model.predict_proba(X)[:, 1]


def save_model(model, path):
    model.save_model(path)


def get_feature_importance(model, feature_names):
    importances = model.get_feature_importance()
    return list(zip(feature_names, importances))