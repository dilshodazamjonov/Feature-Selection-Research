from feature_methods.pca.pca_entry import PCASelector
from feature_methods.preselection.iv_calc import IVFilter
from Preprocessing.data_process import Preprocessor
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np


DATA_DIR = 'data/inputs/Master_Data_with_filtering_updated.csv'
target = 'TARGET'
OUTPUT_DIR = 'data/output/pca'

raw_df = pd.read_csv(DATA_DIR)
print('Data Read')
X = raw_df.drop(columns=[target])
y = raw_df[target]

# ========= Train Test Split =========

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ========= Preprocess ==========

preprocess = Preprocessor()

preprocess.fit(X_train)

X_train_proc = preprocess.transform(X_train)
X_test_proc = preprocess.transform(X_test)


X_train_proc = X_train_proc.replace([np.inf, -np.inf], np.nan)
X_test_proc = X_test_proc.replace([np.inf, -np.inf], np.nan)

# fill using TRAIN statistics
X_train_proc = X_train_proc.fillna(X_train_proc.median())
X_test_proc = X_test_proc.fillna(X_train_proc.median())


print("Preprocessing finished")
print("NaNs:", X_train_proc.isna().sum().sum())
print("Infs:", np.isinf(X_train_proc.values).sum())
# ========= IV filter ===========

iv_filter = IVFilter()

iv_filter.fit(X_train_proc, y_train)

X_train_filtered = iv_filter.transform(X_train_proc)
X_test_filtered = iv_filter.transform(X_test_proc)

print("IV Filter finished")
print(f'N features kept: {len(X_train_filtered.columns)}')

# ========= PCA =============

pca = PCASelector(n_components=20, save_dir=OUTPUT_DIR)

pca.fit(X_train_filtered)

X_train_final = pca.transform(X_train_filtered)
X_test_final = pca.transform(X_test_filtered)

print("PCA finished")
print(f"PCA Selected Columns: {X_train_final.columns}")

# ========= Model ===========

ctb_model = CatBoostClassifier(
    iterations=1000,
    early_stopping_rounds=50,
    verbose=100
)

ctb_model.fit(
    X_train_final,
    y_train,
    eval_set=(X_test_final, y_test)
)

train_preds = ctb_model.predict(X_train_final)
test_preds = ctb_model.predict(X_test_final)

print(f'Train Roc Score {roc_auc_score(train_preds, y_train)}')
print(f'Test Roc Score {roc_auc_score(test_preds, y_test)}')