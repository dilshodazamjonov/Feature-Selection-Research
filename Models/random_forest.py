# models/random_forest.py

import joblib
from sklearn.ensemble import RandomForestClassifier

# Return untrained Random Forest model
def get_rf_model():
    """
    Returns a RandomForestClassifier instance with default settings.
    """
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

# Train RF model
def train_rf(model, X_train, y_train, X_val=None, y_val=None):
    """
    Fits the Random Forest model on training data.
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

# Feature importance from Random Forest
def get_feature_importance(model, feature_names):
    """
    Returns list of tuples (feature, importance) sorted by importance descending.
    """
    importances = model.feature_importances_
    return sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)