import numpy as np
import pandas as pd


def calculate_psi(
    expected: pd.Series,
    actual: pd.Series,
    bins: int = 10,
    eps: float = 1e-6
) -> float:
    """
    Compute Population Stability Index (PSI) between two numeric distributions.

    PSI measures how much a variable's distribution has shifted between
    a reference dataset (expected) and a comparison dataset (actual).

    Parameters
    ----------
    expected : pd.Series
        Reference distribution (typically training data).
    actual : pd.Series
        Comparison distribution (validation, test, or OOT data).
    bins : int, default=10
        Number of quantile-based bins constructed from the expected distribution.
    eps : float, default=1e-6
        Small constant added to avoid division by zero or log(0).

    Returns
    -------
    float
        PSI value. Returns np.nan if binning is not possible (e.g., constant feature).

    Notes
    -----
    - Binning is based ONLY on the expected distribution.
    - Handles NaN and infinite values by dropping them.
    - Aligns bin distributions to avoid index mismatch errors.
    - Interpretation:
        * PSI < 0.1   → Stable
        * 0.1–0.25    → Moderate drift
        * > 0.25      → Significant drift
    """
    try:
        def _to_series(x):
            return pd.Series(x).replace([np.inf, -np.inf], np.nan).dropna()

        expected = _to_series(expected)
        actual = _to_series(actual)

        # Edge cases: empty series
        if len(expected) == 0 or len(actual) == 0:
            return np.nan

        # Compute quantile-based breakpoints from the reference distribution.
        # Make the outer edges open-ended so comparison values outside the
        # training range are still assigned to a PSI bin instead of being
        # silently dropped from the comparison distribution.
        breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
        breakpoints = np.unique(breakpoints.astype(float))

        # Edge case: constant or low-variance feature
        if len(breakpoints) < 2:
            return np.nan

        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf

        # Bin both distributions using expected bins
        expected_bins = pd.cut(expected, bins=breakpoints, include_lowest=True)
        actual_bins = pd.cut(actual, bins=breakpoints, include_lowest=True)

        # Compute normalized distributions
        expected_dist = expected_bins.value_counts(normalize=True).sort_index()
        actual_dist = actual_bins.value_counts(normalize=True).sort_index()

        # Align bins (critical for correctness)
        all_bins = expected_dist.index.union(actual_dist.index)
        expected_dist = expected_dist.reindex(all_bins, fill_value=0)
        actual_dist = actual_dist.reindex(all_bins, fill_value=0)

        # PSI formula
        psi = np.sum(
            (expected_dist - actual_dist) *
            np.log((expected_dist + eps) / (actual_dist + eps))
        )

        return float(psi)
    
    except Exception:
        # Return nan on any calculation error instead of crashing
        return np.nan


def feature_psi(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    bins: int = 10
) -> pd.DataFrame:
    """
    Compute PSI for all features between two datasets.

    Applies `calculate_psi` column-wise to compare feature distributions
    between training and validation/test datasets.

    Parameters
    ----------
    X_train : pd.DataFrame
        Reference dataset (typically training features).
    X_val : pd.DataFrame
        Comparison dataset (validation, test, or OOT features).
    bins : int, default=10
        Number of quantile bins used in PSI calculation.

    Returns
    -------
    pd.DataFrame
        DataFrame with:
            - 'feature': feature name
            - 'psi': PSI score
        Sorted in descending order of PSI.

    Notes
    -----
    - Automatically handles exceptions per feature (returns NaN if failed).
    - Works best on numeric features.
    - High PSI values indicate unstable or drifting features.
    """

    psi_dict = {}

    for col in X_train.columns:
        try:
            psi_dict[col] = calculate_psi(X_train[col], X_val[col], bins)
        except Exception:
            psi_dict[col] = np.nan

    return (
        pd.DataFrame({
            "feature": list(psi_dict.keys()),
            "psi": list(psi_dict.values())
        })
        .sort_values("psi", ascending=False)
        .reset_index(drop=True)
    )


def add_psi_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add PSI stability classification labels to a PSI DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a 'psi' column.

    Returns
    -------
    pd.DataFrame
        Same DataFrame with an additional column:
            - 'psi_flag': categorical stability label

    Labels
    ------
    - 'stable'     : PSI < 0.1
    - 'moderate'   : 0.1 ≤ PSI < 0.25
    - 'unstable'   : PSI ≥ 0.25
    """

    df = df.copy()

    df["psi_flag"] = pd.cut(
        df["psi"],
        bins=[-np.inf, 0.1, 0.25, np.inf],
        labels=["stable", "moderate", "unstable"]
    )

    return df


def jaccard_similarity(set1: set, set2: set) -> float:
    """
    Compute Jaccard similarity between two sets.

    Jaccard similarity is defined as the size of the intersection divided
    by the size of the union of the two sets.

    Parameters
    ----------
    set1 : set
        First set of items.
    set2 : set
        Second set of items.

    Returns
    -------
    float
        Jaccard similarity score in [0, 1]. Returns 0 if both sets are empty.
    """

    if not set1 and not set2:
        return 1.0

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    return intersection / union if union > 0 else 0.0
    
