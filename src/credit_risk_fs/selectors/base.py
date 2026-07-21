from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class FeatureSelector(Protocol):
    """Public contract implemented by fitted feature selectors."""

    selected_features_: list[str] | None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> FeatureSelector: ...

    def transform(self, X: pd.DataFrame) -> pd.DataFrame: ...

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> pd.DataFrame: ...


class SelectedFeaturesMixin:
    """Provide the canonical fitted attribute and its legacy compatibility alias."""

    selected_features_: list[str] | None = None

    @property
    def selected_features(self) -> list[str] | None:
        """Deprecated compatibility alias for :attr:`selected_features_`."""

        return self.selected_features_

    @selected_features.setter
    def selected_features(self, value: Sequence[str] | None) -> None:
        self.selected_features_ = None if value is None else [str(item) for item in value]

    def get_feature_names_out(self) -> list[str]:
        """Return fitted output names in deterministic selection order."""

        if self.selected_features_ is None:
            raise ValueError(f"{self.__class__.__name__} must be fitted first.")
        return list(self.selected_features_)


def validate_feature_frame(X: pd.DataFrame) -> list[str]:
    """Validate and return feature names without changing column identity."""

    if not isinstance(X, pd.DataFrame):
        raise TypeError("Selectors require X to be a pandas DataFrame.")
    names = [str(column) for column in X.columns]
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicates:
        raise ValueError(f"Feature names must be unique; duplicates: {duplicates[:10]}")
    return names


def resolve_feature_budget(requested: int, available: int) -> int:
    """Clamp a non-negative requested feature count to the available width."""

    requested = int(requested)
    available = int(available)
    if requested < 0:
        raise ValueError("Feature budget must be non-negative.")
    if available < 0:
        raise ValueError("Available feature count must be non-negative.")
    return min(requested, available)


def get_selected_features(selector: Any) -> list[str] | None:
    """Read canonical results, falling back to the documented legacy alias."""

    if selector is None:
        return None
    value = getattr(selector, "selected_features_", None)
    if value is None:
        value = getattr(selector, "selected_features", None)
    if value is None:
        return None
    return [str(feature) for feature in value]


def set_selected_features(selector: Any, features: Sequence[str]) -> list[str]:
    """Write canonical results and keep non-mixin legacy selectors usable."""

    normalized = [str(feature) for feature in features]
    selector.selected_features_ = list(normalized)
    if not isinstance(selector, SelectedFeaturesMixin):
        selector.selected_features = list(normalized)
    return normalized


def select_feature_frame(
    X: pd.DataFrame,
    selected_features: Sequence[str] | None,
    *,
    selector_name: str,
) -> pd.DataFrame:
    """Return selected columns or raise a specific fitted/schema error."""

    validate_feature_frame(X)
    if selected_features is None:
        raise ValueError(f"{selector_name} must be fitted before transform.")

    selected = [str(feature) for feature in selected_features]
    duplicate_selected = sorted(
        feature for feature, count in Counter(selected).items() if count > 1
    )
    if duplicate_selected:
        raise ValueError(
            f"{selector_name} produced duplicate selected features: "
            f"{duplicate_selected[:10]}"
        )

    available = {str(column): column for column in X.columns}
    missing = [feature for feature in selected if feature not in available]
    if missing:
        raise ValueError(
            f"Input is missing {len(missing)} features selected by {selector_name}: "
            f"{missing[:10]}"
        )
    return X.loc[:, [available[feature] for feature in selected]]
