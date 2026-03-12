import pandas as pd

# ========== STEP FUNCTIONS ==========

def detect_categorical_columns(df: pd.DataFrame):
    """Return categorical column names."""
    return df.select_dtypes(include=["object", "category"]).columns.tolist()


def compute_numeric_medians(df: pd.DataFrame):
    """Compute medians for numeric columns."""
    numeric_cols = df.select_dtypes(exclude=["object", "category"]).columns
    return df[numeric_cols].median()


def detect_zero_variance_columns(df: pd.DataFrame):
    """Detect columns with no variance."""
    return df.columns[df.nunique() <= 1].tolist()


def add_missing_indicators(df: pd.DataFrame):
    """Add binary indicators for missing values."""
    df = df.copy()

    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df[f"{col}_missing"] = df[col].isnull().astype(int)

    return df


def impute_numeric(df: pd.DataFrame, medians: pd.Series):
    """Fill numeric missing values using stored medians."""
    df = df.copy()

    for col in medians.index:
        if col in df.columns:
            df[col] = df[col].fillna(medians[col])

    return df


def impute_categorical(df: pd.DataFrame, cat_columns):
    """Fill categorical missing values."""
    df = df.copy()

    for col in cat_columns:
        if col in df.columns:
            df[col] = df[col].fillna("missing")

    return df


def remove_zero_variance(df: pd.DataFrame, zero_var_cols):
    """Remove zero variance columns."""
    return df.drop(columns=zero_var_cols, errors="ignore")


def encode_categoricals(df: pd.DataFrame):
    """One-hot encode categorical variables."""
    return pd.get_dummies(df, drop_first=True)


def align_columns(df: pd.DataFrame, expected_columns):
    """Ensure same columns as training data."""
    return df.reindex(columns=expected_columns, fill_value=0)


# ========== PREPROCESSOR CLASS ==========

class Preprocessor:

    def __init__(self):
        self.medians = None
        self.cat_columns = None
        self.zero_var_cols = None
        self.dummy_columns = None

    def fit(self, X: pd.DataFrame):
        X = X.copy()
        
        self.cat_columns = detect_categorical_columns(X)
        self.medians = compute_numeric_medians(X)
        self.zero_var_cols = detect_zero_variance_columns(X)

        X_tmp = add_missing_indicators(X)
        X_tmp = impute_numeric(X_tmp, self.medians)
        X_tmp = impute_categorical(X_tmp, self.cat_columns)
        X_tmp = remove_zero_variance(X_tmp, self.zero_var_cols)
        X_tmp = encode_categoricals(X_tmp)

        X_tmp = X_tmp.drop(columns=['SK_ID_CURR'], errors='ignore')

        self.dummy_columns = X_tmp.columns

    def transform(self, X: pd.DataFrame):
        X = X.copy()

        X = add_missing_indicators(X)
        X = impute_numeric(X, self.medians)
        X = impute_categorical(X, self.cat_columns)
        X = remove_zero_variance(X, self.zero_var_cols)
        X = encode_categoricals(X)
        X = align_columns(X, self.dummy_columns)

        return X

    def fit_transform(self, X: pd.DataFrame):
        self.fit(X)
        return self.transform(X)



# ============= Usage ===============

# pre = Preprocessor()
# X_train = pre.fit_transform(X_train)
# X_val = pre.transform(X_val)