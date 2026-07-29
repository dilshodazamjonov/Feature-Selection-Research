"""Fit numeric selectors on original features before final-model preprocessing.

The final-model preprocessor may one-hot encode a different set of categories in
each expanding fold. Feature-selection identities must nevertheless remain in
the frozen original-feature universe. This adapter supplies the one-column-per-
original-feature numeric view used by the authenticated DEV pilots, then projects
the selected original columns before the final-model preprocessor is fitted.
"""

from __future__ import annotations

import copy
from typing import Any

import pandas as pd

from credit_risk_fs.preprocessing.encoding import OriginalFeatureNumericEncoder
from credit_risk_fs.selectors.base import (
    SelectedFeaturesMixin,
    get_selected_features,
    select_feature_frame,
    validate_feature_frame,
)


class OriginalFeatureSelectorAdapter(SelectedFeaturesMixin):
    """Adapt a numeric selector to the original-feature selection boundary."""

    select_before_preprocessing = True
    apply_post_preprocessing = False
    selection_encoding = (
        "credit_risk_fs.preprocessing.encoding.OriginalFeatureNumericEncoder"
    )

    def __init__(
        self,
        *,
        selector_cls: type,
        selector_kwargs: dict[str, Any] | None = None,
        k: int | None = None,
        random_state: int = 42,
    ) -> None:
        if not isinstance(selector_cls, type):
            raise TypeError("selector_cls must be a selector class")
        kwargs = copy.deepcopy(dict(selector_kwargs or {}))
        if k is not None:
            kwargs["k"] = int(k)
        if "random_state" in kwargs:
            kwargs["random_state"] = int(random_state)
        self.selector_cls = selector_cls
        self.selector_kwargs = kwargs
        self.k = None if k is None else int(k)
        self.random_state = int(random_state)
        self.selector = selector_cls(**kwargs)
        self.encoder_: OriginalFeatureNumericEncoder | None = None
        self.selected_features_ = None
        self.original_candidate_features_: list[str] | None = None
        self.result_ = None

    @property
    def result(self):
        result = getattr(self.selector, "result", None)
        if result is None:
            raise AttributeError("wrapped selector has no result contract")
        return result

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> "OriginalFeatureSelectorAdapter":
        self.original_candidate_features_ = validate_feature_frame(X)
        self.encoder_ = OriginalFeatureNumericEncoder()
        numeric = self.encoder_.fit_transform(X)
        if list(numeric.columns) != self.original_candidate_features_:
            raise RuntimeError("selection encoding changed original feature identity")
        self.selector.fit(numeric, y)
        selected = get_selected_features(self.selector)
        if selected is None:
            raise RuntimeError("wrapped selector did not publish selected features")
        self.selected_features_ = list(selected)
        self.result_ = getattr(self.selector, "result_", None)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return select_feature_frame(
            X,
            self.selected_features_,
            selector_name=self.selector_cls.__name__,
        )

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self) -> list[str]:
        if self.selected_features_ is None:
            raise ValueError("OriginalFeatureSelectorAdapter must be fitted first")
        return list(self.selected_features_)

    def __getattr__(self, name: str) -> Any:
        # Preserve legacy importance/trace attributes used by artifact helpers.
        if name == "selector":
            raise AttributeError(name)
        selector = self.__dict__.get("selector")
        if selector is None:
            raise AttributeError(name)
        return getattr(selector, name)


__all__ = ["OriginalFeatureSelectorAdapter"]
