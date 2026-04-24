from sklearn.decomposition import PCA
import pandas as pd
import os
from typing import List


class PCASelector:
    def __init__(self, n_components=0.95, save_dir=None):
        """
        PCA based feature reduction.

        Parameters
        ----------
        n_components : int or float
            Number of components or variance ratio (0.95 = keep 95% variance)
        save_dir : str
            Directory to save PCA artifacts
        """

        self.n_components = n_components
        self.save_dir = save_dir
        self.pca = PCA(n_components=n_components)

        self.feature_names: List = None
        self.explained_variance: List = None

    def fit(self, X: pd.DataFrame, y=None):
        """Fit PCA model."""

        self.feature_names = X.columns

        self.pca.fit(X)

        self.explained_variance = self.pca.explained_variance_ratio_

        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)

            # save explained variance
            variance_df = pd.DataFrame({
                "component": range(len(self.explained_variance)),
                "explained_variance_ratio": self.explained_variance
            })

            variance_df.to_csv(
                os.path.join(self.save_dir, "pca_explained_variance.csv"),
                index=False
            )
        return self

    def transform(self, X: pd.DataFrame):
        """Transform dataset using fitted PCA."""

        if self.feature_names is None:
            raise ValueError("PCASelector not fitted yet")

        X_pca = self.pca.transform(X)

        columns = [f"PC{i+1}" for i in range(X_pca.shape[1])]

        return pd.DataFrame(X_pca, columns=columns, index=X.index)

    def fit_transform(self, X: pd.DataFrame, y=None):
        """Fit and transform."""

        self.fit(X, y=y)
        return self.transform(X)
