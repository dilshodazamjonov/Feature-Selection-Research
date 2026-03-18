# models/logistic_regression.py

import joblib
from sklearn.linear_model import LogisticRegression

# Return untrained Logistic Regression model
def get_lr_model():
    """
    Returns a LogisticRegression model instance with default settings.
    """
    return LogisticRegression(
        solver="liblinear",
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )

# Train LR model
def train_lr(model, X_train, y_train, X_val=None, y_val=None):
    """
    Fits the Logistic Regression model on training data.
    """
    model.fit(X_train, y_train)
    return model

# Predict probabilities
def predict_proba(model, X):
    """
    Returns predicted probabilities for the positive class.
    """
    return model.predict_proba(X)[:, 1]

# Save model
def save_model(model, path):
    """
    Saves the trained model to disk.
    """
    joblib.dump(model, path)

# Feature importance (absolute coefficient values)
def get_feature_importance(model, feature_names):
    """
    Returns list of tuples (feature, importance) sorted by importance descending.
    """
    coefs = model.coef_[0]
    return sorted(zip(feature_names, abs(coefs)), key=lambda x: x[1], reverse=True)