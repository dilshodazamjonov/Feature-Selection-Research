# rf_pca_pipeline.py
"""
PCA + Random Forest pipeline for credit scoring.
Uses metrics.py utilities to calculate and save performance metrics.
Shows progress with tqdm and uses all CPU cores for RF.
"""

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from Preprocessing.data_process import preprocess_data
from Models.metrics import calculate_metrics, save_metrics_to_csv
from tqdm import tqdm  
import numpy as np

# --- 1. Load and preprocess data ---
X_train_scaled, X_test_scaled, y_train, y_test, feature_names = preprocess_data(
    "data/inputs/Master_Data_with_filtering_updated.csv",
    target_col="TARGET"
)

# --- 2. PCA + Random Forest experiments ---
results = []

n_components_list = [10]

for n_components in tqdm(n_components_list, desc="Running PCA+RF experiments"):
    # PCA
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train_pca, y_train)
    
    # Predictions
    y_pred = clf.predict(X_test_pca)
    y_pred_proba = clf.predict_proba(X_test_pca)[:, 1]
    
    # Metrics
    metrics_dict = calculate_metrics(
        y_true=y_test,
        y_pred=y_pred,
        y_pred_proba=y_pred_proba,
        n_components=n_components,
        original_features=X_train_scaled.shape[1],
        pca_explained_variance=pca.explained_variance_ratio_
    )
    
    results.append(metrics_dict)

    total_variance = np.sum(pca.explained_variance_ratio_)
    print(f"Completed PCA with {n_components} components, Explained Variance: {total_variance:.4f}")

save_metrics_to_csv(results, filepath="data/output/pca_metrics_results.csv")
print("PCA + RF pipeline complete. Metrics saved.")