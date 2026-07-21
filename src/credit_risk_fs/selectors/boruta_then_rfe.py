from __future__ import annotations

import logging
import warnings
from typing import Any

import pandas as pd

from credit_risk_fs.selectors.base import SelectedFeaturesMixin, select_feature_frame
from credit_risk_fs.selectors.boruta import BorutaSelector
from credit_risk_fs.selectors.rfe import RFESelector


logger = logging.getLogger(__name__)


class BorutaThenRFESelector(SelectedFeaturesMixin):
    """Run Boruta and then CatBoost RFE on the confirmed training features."""

    def __init__(
        self,
        boruta_kwargs: dict[str, Any] | None = None,
        rfe_kwargs: dict[str, Any] | None = None,
        *,
        use_rfe: bool = True,
        n_features: int = 40,
    ) -> None:
        self.boruta_kwargs = dict(boruta_kwargs or {})
        self.rfe_kwargs = dict(rfe_kwargs or {})
        self.use_rfe = bool(use_rfe)
        self.n_features = int(n_features)
        if not self.use_rfe:
            warnings.warn(
                "use_rfe=False is retained only for BorutaRFESelector compatibility; "
                "use BorutaSelector for Boruta-only selection.",
                DeprecationWarning,
                stacklevel=2,
            )

        self.boruta = BorutaSelector(**self.boruta_kwargs)
        rfe_kwargs_with_budget = dict(self.rfe_kwargs)
        rfe_kwargs_with_budget.setdefault("n_features", self.n_features)
        self.rfe = RFESelector(**rfe_kwargs_with_budget) if self.use_rfe else None
        self.selected_features_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> BorutaThenRFESelector:
        """Fit both stages on the same training slice; never backfill features."""

        X_boruta = self.boruta.fit_transform(X, y)
        logger.info("After Boruta: %s", X_boruta.shape)

        if not self.use_rfe:
            self.selected_features_ = list(self.boruta.selected_features_ or [])[
                : self.n_features
            ]
            return self
        if X_boruta.shape[1] == 0:
            raise ValueError("Boruta selected zero features; RFE cannot be fitted.")

        if self.rfe is None:  # Defensive guard for deserialized legacy objects.
            raise ValueError("RFE stage is not configured.")
        self.rfe.fit(X_boruta, y)
        self.selected_features_ = list(self.rfe.selected_features_ or [])
        logger.info("After RFE: %d selected features", len(self.selected_features_))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return select_feature_frame(
            X,
            self.selected_features_,
            selector_name=self.__class__.__name__,
        )

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)


class BorutaRFESelector(BorutaThenRFESelector):
    """Deprecated class-name alias retained for saved callers and configurations."""

    def __init__(
        self,
        boruta_kwargs: dict[str, Any] | None = None,
        rfe_kwargs: dict[str, Any] | None = None,
        use_rfe: bool = False,
        n_features: int = 40,
    ) -> None:
        super().__init__(
            boruta_kwargs=boruta_kwargs,
            rfe_kwargs=rfe_kwargs,
            use_rfe=use_rfe,
            n_features=n_features,
        )


__all__ = ["BorutaThenRFESelector", "BorutaRFESelector"]
