"""Standalone CatBoost-backed recursive feature elimination.

Relationship to the legacy path. `credit_risk_fs.selectors.rfe.RFESelector` is
CatBoost-backed, as the roadmap records, but it delegates to
`sklearn.feature_selection.RFE` with an **integer** ``step=10`` (remove exactly ten
features per iteration) and exposes only sklearn's final ``ranking_``. It records
neither the number of estimator fits nor the realized removals per iteration.

This module is therefore a separate implementation, not a wrapper: it runs the
elimination loop explicitly so it can record a **fractional** removal step, the
exact fit count, and per-iteration elimination history. The legacy class is
untouched and keeps its own registry route.

Both use the same estimator profile (CatBoost, 500 iterations, depth 6, learning
rate 0.05, CPU, no file writing, silent) so the two remain scientifically
comparable, and the difference in step semantics is recorded in the descriptor and
in every result's configuration.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from credit_risk_fs.selectors.heavy._support import (
    available_ram_bytes,
    estimator_config_hash,
    heavy_stage,
    process_rss_bytes,
)
from credit_risk_fs.selectors.lightweight.contract import (
    ControlledSelectorFailure,
    LightweightSelector,
)

logger = logging.getLogger(__name__)

#: Fraction of the *currently surviving* features removed per iteration. 0.20 is
#: the production default requested by the roadmap for the standalone method; the
#: legacy sklearn-backed path keeps its frozen integer step of 10.
DEFAULT_STEP_FRACTION = 0.20

#: Estimator profile copied from `RFESelector._estimator_config` so the standalone
#: method and the legacy method fit the same model.
DEFAULT_CATBOOST_PARAMS: dict[str, Any] = {
    "iterations": 500,
    "depth": 6,
    "learning_rate": 0.05,
    "verbose": False,
    "allow_writing_files": False,
    "task_type": "CPU",
}


class CatBoostRFESelector(LightweightSelector):
    """Eliminate features recursively using CatBoost importance.

    RFE is a budgeted wrapper method, not an all-relevant method: it has no natural
    stopping point, so ``k`` is required and ``natural_selected`` stays absent
    rather than being fabricated from the surviving set.
    """

    method_id: ClassVar[str] = "rfe_catboost"
    display_label: ClassVar[str] = "RFE (CatBoost)"
    implementation_id: ClassVar[str] = "rfe_catboost_fractional_step_v1"
    supervised: ClassVar[bool] = True
    supports_ranking: ClassVar[bool] = True
    supports_natural_support: ClassVar[bool] = False
    supports_fixed_budget: ClassVar[bool] = True
    default_selection_mode: ClassVar[str] = "matched_budget"
    #: The published order is elimination order. Per-iteration CatBoost importances
    #: are not comparable across refits, so a rank-derived score is published and
    #: the raw per-step importances are retained separately in the history.
    score_orientation: ClassVar[str] = "rank_1_is_best"

    def __init__(
        self,
        *,
        k: int | None = None,
        step_fraction: float = DEFAULT_STEP_FRACTION,
        catboost_params: dict[str, Any] | None = None,
        random_state: int = 42,
        thread_count: int = 1,
        excluded_columns: Sequence[str] | None = None,
        fit_scope: str = "dev_fold_training_only",
    ) -> None:
        super().__init__(
            k=k,
            random_state=random_state,
            excluded_columns=excluded_columns,
            fit_scope=fit_scope,
        )
        if not 0.0 < float(step_fraction) < 1.0:
            raise ValueError(
                "step_fraction must be a fraction in (0, 1); the legacy integer-step "
                "path is reachable through the 'rfe' registry id"
            )
        if int(thread_count) <= 0:
            raise ValueError("thread_count must be positive")
        self.step_fraction = float(step_fraction)
        self.thread_count = int(thread_count)
        self.catboost_params = dict(DEFAULT_CATBOOST_PARAMS)
        if catboost_params:
            self.catboost_params.update(catboost_params)
        self.elimination_history_: pd.DataFrame | None = None
        self.estimator_fit_count_: int = 0
        self.final_importances_: dict[str, float] | None = None
        self._resource: dict[str, Any] = {}

    # -- configuration -----------------------------------------------------

    def _effective_catboost_params(self) -> dict[str, Any]:
        params = dict(self.catboost_params)
        params.update(
            {
                "random_seed": self.random_state,
                "thread_count": self.thread_count,
                # Non-negotiable: CatBoost must not litter the working tree or
                # flood the terminal from inside a selector.
                "allow_writing_files": False,
                "verbose": False,
            }
        )
        return params

    def describe_configuration(self) -> dict[str, Any]:
        configuration = super().describe_configuration()
        configuration.update(
            {
                "estimator": "catboost.CatBoostClassifier",
                "estimator_params": self._effective_catboost_params(),
                "step_kind": "fraction_of_surviving_features",
                "step_fraction": self.step_fraction,
                "thread_count": self.thread_count,
                "importance_type": "catboost_default_prediction_values_change",
                "natural_support": "unsupported_wrapper_method",
                "tie_rule_detail": (
                    "least important first; exact importance ties eliminate the "
                    "candidate appearing later in the authenticated order"
                ),
                "legacy_counterpart": {
                    "registry_id": "rfe",
                    "implementation": "sklearn.feature_selection.RFE",
                    "step_kind": "integer_features_per_iteration",
                    "step": 10,
                },
            }
        )
        return configuration

    def _estimator_config_sha256(self) -> str | None:
        return estimator_config_hash(
            {
                **self._effective_catboost_params(),
                "step_fraction": self.step_fraction,
                "requested_k": self.k,
            }
        )

    def _heavy_metadata(self) -> dict[str, Any]:
        history = (
            []
            if self.elimination_history_ is None
            else self.elimination_history_.to_dict("records")
        )
        return {
            "cost_class": "heavy",
            "estimator_fit_count": int(self.estimator_fit_count_),
            "step_fraction_configured": self.step_fraction,
            "elimination_history": history,
            "final_estimator_importances": dict(self.final_importances_ or {}),
            "thread_count": self.thread_count,
            **self._resource,
        }

    # -- validation --------------------------------------------------------

    def _validate_configuration(self, *, eligible_count: int) -> None:
        if self.k is None:
            raise ControlledSelectorFailure(
                method_id=self.method_id,
                stage="budget_validation",
                cause=(
                    "recursive elimination has no natural stopping point, so a fixed "
                    "feature budget k is required; k=None is unsupported and no "
                    "natural support is fabricated"
                ),
                configuration=self.describe_configuration(),
            )
        if int(self.k) <= 0:
            raise ControlledSelectorFailure(
                method_id=self.method_id,
                stage="budget_validation",
                cause=f"feature budget must be positive; received k={self.k}",
                configuration=self.describe_configuration(),
            )

    # -- elimination -------------------------------------------------------

    def _fit_importances(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        features: Sequence[str],
    ) -> dict[str, float]:
        from catboost import CatBoostClassifier

        model = CatBoostClassifier(**self._effective_catboost_params())
        try:
            model.fit(X.loc[:, list(features)], y)
        except Exception as error:
            # A training failure is reported as-is. Changing parameters, rows,
            # step, or estimator to get past it would silently alter the method.
            raise ControlledSelectorFailure(
                method_id=self.method_id,
                stage="estimator_fit",
                cause=(
                    f"CatBoost training failed after {self.estimator_fit_count_} fit(s) "
                    f"on {len(features)} feature(s): {type(error).__name__}: {error}"
                ),
                configuration=self.describe_configuration(),
            ) from error
        self.estimator_fit_count_ += 1
        importances = np.asarray(model.get_feature_importance(), dtype="float64")
        if importances.shape[0] != len(features):
            raise ControlledSelectorFailure(
                method_id=self.method_id,
                stage="importance_extraction",
                cause=(
                    f"CatBoost returned {importances.shape[0]} importances for "
                    f"{len(features)} features"
                ),
                configuration=self.describe_configuration(),
            )
        return {str(name): float(value) for name, value in zip(features, importances, strict=True)}

    def _compute(
        self,
        X: pd.DataFrame,
        y: pd.Series | None,
        *,
        candidate_order: Sequence[str],
    ) -> tuple[list[str], dict[str, float] | None, list[str] | None]:
        if y is None:  # pragma: no cover - guarded by LightweightSelector.fit
            raise ControlledSelectorFailure(
                method_id=self.method_id,
                stage="target_validation",
                cause="recursive elimination requires the binary default target",
                configuration=self.describe_configuration(),
            )

        order_index = {str(name): position for position, name in enumerate(candidate_order)}
        target = int(min(int(self.k or 0), len(candidate_order)))
        surviving = [str(name) for name in candidate_order]
        eliminated: list[tuple[str, int]] = []
        history: list[dict[str, Any]] = []
        self.estimator_fit_count_ = 0
        self.final_importances_ = None

        if target >= len(surviving):
            # Nothing to eliminate. Returning the universe without fitting keeps
            # the fit count honest at zero rather than doing pointless work.
            self.elimination_history_ = pd.DataFrame(
                columns=[
                    "iteration",
                    "surviving_before",
                    "requested_removals",
                    "realized_removals",
                    "removed_features",
                ]
            )
            self._resource = {
                "peak_process_rss_bytes": process_rss_bytes(),
                "minimum_available_ram_bytes": available_ram_bytes(),
                "elimination_skipped": True,
            }
            scores = {
                name: float(len(surviving) - order_index[name]) for name in surviving
            }
            return surviving, scores, None

        with heavy_stage(
            logger,
            method_id=self.method_id,
            stage="recursive_elimination",
            detail=(
                f"candidates={len(surviving)} k={target} "
                f"step_fraction={self.step_fraction}"
            ),
        ) as observations:
            iteration = 0
            while len(surviving) > target:
                iteration += 1
                importances = self._fit_importances(X, y, surviving)
                requested = max(1, int(self.step_fraction * len(surviving)))
                removals = min(requested, len(surviving) - target)
                # Least important first. An exact importance tie is broken by the
                # authenticated candidate order: the later candidate goes first,
                # so the surviving set never depends on container order.
                ranked = sorted(
                    surviving,
                    key=lambda name: (importances[name], -order_index[name]),
                )
                removed = ranked[:removals]
                for name in removed:
                    eliminated.append((name, iteration))
                    surviving.remove(name)
                history.append(
                    {
                        "iteration": iteration,
                        "surviving_before": len(surviving) + removals,
                        "requested_removals": requested,
                        "realized_removals": removals,
                        "removed_features": list(removed),
                    }
                )
            # One final fit on the surviving subset orders the retained features.
            self.final_importances_ = self._fit_importances(X, y, surviving)
            observations["fits"] = self.estimator_fit_count_
            observations["iterations"] = iteration
            observations["surviving"] = len(surviving)

        self.elimination_history_ = pd.DataFrame(history)
        self._resource = {
            "peak_process_rss_bytes": process_rss_bytes(),
            "minimum_available_ram_bytes": available_ram_bytes(),
            "elimination_skipped": False,
        }

        final = self.final_importances_
        retained = sorted(
            surviving,
            key=lambda name: (-final[name], order_index[name]),
        )
        # Eliminated features rank after every survivor, latest-eliminated first,
        # which reproduces the elimination order exactly.
        dropped = sorted(
            eliminated,
            key=lambda item: (-item[1], order_index[item[0]]),
        )
        ranking = retained + [name for name, _ in dropped]
        # A rank-derived score, because importances from different refits are not
        # a single comparable scale. The raw per-step evidence lives in the history.
        scores = {name: float(len(ranking) - position) for position, name in enumerate(ranking)}
        return ranking, scores, None

    def _collect_warnings(self, budget_status: str) -> list[str]:
        collected = super()._collect_warnings(budget_status)
        if self._resource.get("elimination_skipped"):
            collected.append(
                "requested budget covers the whole eligible universe; no elimination "
                "was necessary, so zero estimator fits were performed and the "
                "published order is the authenticated candidate order"
            )
        return collected


__all__ = ["DEFAULT_CATBOOST_PARAMS", "DEFAULT_STEP_FRACTION", "CatBoostRFESelector"]
