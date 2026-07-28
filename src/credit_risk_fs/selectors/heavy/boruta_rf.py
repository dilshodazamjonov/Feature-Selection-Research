"""Random-forest Boruta with confirmed / tentative / rejected preserved.

Relationship to the legacy path. `credit_risk_fs.selectors.boruta.BorutaSelector`
reads only ``BorutaPy.support_`` and ``ranking_``. It therefore keeps confirmed
features and **discards the tentative state entirely** -- ``support_weak_`` is
never read. It does not pad (it clamps a requested count down to the confirmed
count), but a caller cannot tell a tentative feature from a rejected one.

This module preserves all three states. The legacy class is untouched, keeps its
``boruta`` registry route, and remains the implementation the frozen voting
protocol resolves its ``boruta`` voter to. Nothing here is permitted in that
protocol.
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

CONFIRMED = "confirmed"
TENTATIVE = "tentative"
REJECTED = "rejected"

BORUTA_MODES = ("natural_confirmed", "confirmed_top_k", "confirmed_then_tentative")

#: Forest profile copied from the legacy ``BorutaSelector`` so the two remain
#: scientifically comparable. Prompt 8 also exposes a deliberately tiny profile for
#: synthetic tests; Prompt 9 measures real single-fold cost and freezes production.
DEFAULT_FOREST_PARAMS: dict[str, Any] = {
    "n_estimators": 500,
    "max_depth": 6,
    "class_weight": None,
}

DEFAULT_BORUTA_PARAMS: dict[str, Any] = {
    "n_estimators": "auto",
    "max_iter": 10,
    "perc": 100,
    "alpha": 0.05,
    "two_step": True,
    "verbose": 0,
}


class BorutaRandomForestSelector(LightweightSelector):
    """All-relevant selection preserving Boruta's three decision states.

    ``natural_selected`` holds the **confirmed** features only. Tentative features
    are reported separately and are never promoted silently; reaching a budget by
    including them requires the explicitly named ``confirmed_then_tentative`` mode,
    which the result labels as a matched-budget adaptation rather than natural
    Boruta support. Rejected features never pad a budget under any mode.
    """

    method_id: ClassVar[str] = "boruta_random_forest"
    display_label: ClassVar[str] = "Boruta (random forest)"
    implementation_id: ClassVar[str] = "boruta_random_forest_confirmed_tentative_v1"
    supervised: ClassVar[bool] = True
    supports_ranking: ClassVar[bool] = True
    supports_natural_support: ClassVar[bool] = True
    supports_fixed_budget: ClassVar[bool] = True
    default_selection_mode: ClassVar[str] = "natural_confirmed"
    score_orientation: ClassVar[str] = "rank_1_is_best"

    def __init__(
        self,
        *,
        k: int | None = None,
        selection_mode: str = "natural_confirmed",
        forest_params: dict[str, Any] | None = None,
        boruta_params: dict[str, Any] | None = None,
        random_state: int = 42,
        n_jobs: int = 1,
        engine_factory: Any = None,
        excluded_columns: Sequence[str] | None = None,
        fit_scope: str = "dev_fold_training_only",
    ) -> None:
        super().__init__(
            k=k,
            random_state=random_state,
            excluded_columns=excluded_columns,
            fit_scope=fit_scope,
        )
        if str(selection_mode) not in BORUTA_MODES:
            raise ValueError(
                f"selection_mode must be one of {list(BORUTA_MODES)}; got {selection_mode!r}"
            )
        if int(n_jobs) <= 0:
            raise ValueError("n_jobs must be positive")
        self.selection_mode = str(selection_mode)
        self.n_jobs = int(n_jobs)
        self.forest_params = dict(DEFAULT_FOREST_PARAMS)
        if forest_params:
            self.forest_params.update(forest_params)
        self.boruta_params = dict(DEFAULT_BORUTA_PARAMS)
        if boruta_params:
            self.boruta_params.update(boruta_params)
        #: Injection point for the deterministic stub used by the support-policy
        #: tests. Production leaves it None and the installed BorutaPy is used.
        self.engine_factory = engine_factory
        self.support_states_: dict[str, str] | None = None
        self.confirmed_: list[str] = []
        self.tentative_: list[str] = []
        self.rejected_: list[str] = []
        self.engine_ranking_: dict[str, int] | None = None
        self.stop_reason_: str | None = None
        self._budget_status_override: str | None = None
        self._resource: dict[str, Any] = {}

    # -- configuration -----------------------------------------------------

    def _effective_forest_params(self) -> dict[str, Any]:
        params = dict(self.forest_params)
        params.update({"random_state": self.random_state, "n_jobs": self.n_jobs})
        return params

    def _effective_boruta_params(self) -> dict[str, Any]:
        params = dict(self.boruta_params)
        params.update({"random_state": self.random_state})
        return params

    def describe_configuration(self) -> dict[str, Any]:
        configuration = super().describe_configuration()
        configuration.update(
            {
                "engine": "boruta.BorutaPy",
                "estimator": "sklearn.ensemble.RandomForestClassifier",
                "forest_params": self._effective_forest_params(),
                "boruta_params": self._effective_boruta_params(),
                "selection_mode": self.selection_mode,
                "n_jobs": self.n_jobs,
                "support_states": [CONFIRMED, TENTATIVE, REJECTED],
                "natural_support_definition": "confirmed_only",
                "tentative_promotion": (
                    "only through the explicitly named confirmed_then_tentative mode"
                ),
                "rejected_used_for_padding": False,
                "profile_note": (
                    "the tiny profile used by synthetic tests is not the frozen "
                    "research configuration; Prompt 9 freezes production"
                ),
                "legacy_counterpart": {
                    "registry_id": "boruta",
                    "implementation": "credit_risk_fs.selectors.boruta.BorutaSelector",
                    "exposes_tentative": False,
                },
            }
        )
        return configuration

    def _estimator_config_sha256(self) -> str | None:
        return estimator_config_hash(
            {
                "forest": self._effective_forest_params(),
                "boruta": self._effective_boruta_params(),
                "selection_mode": self.selection_mode,
                "requested_k": self.k,
            }
        )

    def _heavy_metadata(self) -> dict[str, Any]:
        return {
            "cost_class": "heavy",
            "support_states": dict(self.support_states_ or {}),
            "confirmed": list(self.confirmed_),
            "tentative": list(self.tentative_),
            "rejected": list(self.rejected_),
            "confirmed_count": len(self.confirmed_),
            "tentative_count": len(self.tentative_),
            "rejected_count": len(self.rejected_),
            "engine_ranking": dict(self.engine_ranking_ or {}),
            "selection_mode": self.selection_mode,
            "natural_support_definition": "confirmed_only",
            "stop_reason": self.stop_reason_,
            "n_jobs": self.n_jobs,
            **self._resource,
        }

    # -- validation --------------------------------------------------------

    def _validate_configuration(self, *, eligible_count: int) -> None:
        if self.selection_mode in {"confirmed_top_k", "confirmed_then_tentative"}:
            if self.k is None:
                raise ControlledSelectorFailure(
                    method_id=self.method_id,
                    stage="budget_validation",
                    cause=(
                        f"selection_mode={self.selection_mode!r} is a fixed-budget mode "
                        "and requires k; use natural_confirmed for Boruta's own answer"
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

    # -- engine ------------------------------------------------------------

    def _build_engine(self) -> Any:
        if self.engine_factory is not None:
            return self.engine_factory(
                forest_params=self._effective_forest_params(),
                boruta_params=self._effective_boruta_params(),
            )
        from boruta import BorutaPy
        from sklearn.ensemble import RandomForestClassifier

        estimator = RandomForestClassifier(**self._effective_forest_params())
        return BorutaPy(estimator=estimator, **self._effective_boruta_params())

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
                cause="Boruta requires the binary default target",
                configuration=self.describe_configuration(),
            )

        # BorutaPy delegates to a scikit-learn forest, which rejects non-finite
        # values. The legacy BorutaSelector shares this constraint -- it also hands
        # X.to_numpy() straight to the engine. Checking here turns an opaque
        # sklearn ValueError into an attributable selector failure that names the
        # offending columns. The selector deliberately does NOT impute: doing so
        # would silently change the method's preprocessing.
        frame = X.loc[:, list(candidate_order)]
        non_finite = [
            str(name)
            for name in candidate_order
            if not np.isfinite(
                pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype="float64")
            ).all()
        ]
        if non_finite:
            raise ControlledSelectorFailure(
                method_id=self.method_id,
                stage="design_matrix_validation",
                cause=(
                    f"the Boruta engine requires a finite numeric matrix, but "
                    f"{len(non_finite)} candidate feature(s) contain missing or "
                    f"non-finite values: {non_finite[:5]}; impute upstream rather "
                    "than relying on the selector to do it"
                ),
                configuration=self.describe_configuration(),
            )

        engine = self._build_engine()
        with heavy_stage(
            logger,
            method_id=self.method_id,
            stage="boruta_all_relevant_search",
            detail=(
                f"candidates={len(candidate_order)} "
                f"max_iter={self._effective_boruta_params().get('max_iter')}"
            ),
        ) as observations:
            try:
                engine.fit(frame.to_numpy(), np.asarray(y.to_numpy()))
            except Exception as error:
                raise ControlledSelectorFailure(
                    method_id=self.method_id,
                    stage="engine_fit",
                    cause=(
                        f"Boruta engine failed on {len(candidate_order)} feature(s): "
                        f"{type(error).__name__}: {error}"
                    ),
                    configuration=self.describe_configuration(),
                ) from error

            support = np.asarray(getattr(engine, "support_", None), dtype=bool)
            weak = getattr(engine, "support_weak_", None)
            weak = (
                np.zeros(len(candidate_order), dtype=bool)
                if weak is None
                else np.asarray(weak, dtype=bool)
            )
            ranking = np.asarray(getattr(engine, "ranking_", None))
            for name, array in (("support_", support), ("support_weak_", weak), ("ranking_", ranking)):
                if array.shape[0] != len(candidate_order):
                    raise ControlledSelectorFailure(
                        method_id=self.method_id,
                        stage="support_extraction",
                        cause=(
                            f"Boruta {name} has length {array.shape[0]} for "
                            f"{len(candidate_order)} candidate feature(s)"
                        ),
                        configuration=self.describe_configuration(),
                    )
            observations["confirmed"] = int(support.sum())
            observations["tentative"] = int((weak & ~support).sum())

        self.engine_ranking_ = {
            str(name): int(value)
            for name, value in zip(candidate_order, ranking, strict=True)
        }
        self.stop_reason_ = str(
            getattr(engine, "stop_reason_", None)
            or f"engine_completed_max_iter_{self._effective_boruta_params().get('max_iter')}"
        )

        states: dict[str, str] = {}
        for position, name in enumerate(candidate_order):
            key = str(name)
            if support[position]:
                states[key] = CONFIRMED
            elif weak[position]:
                # A feature can appear in both arrays; confirmed wins, so tentative
                # means "weak and not confirmed" and the states stay disjoint.
                states[key] = TENTATIVE
            else:
                states[key] = REJECTED
        self.support_states_ = states

        order_index = {str(name): position for position, name in enumerate(candidate_order)}

        def engine_order(name: str) -> tuple[int, int]:
            return (self.engine_ranking_[name], order_index[name])

        self.confirmed_ = sorted(
            (name for name, state in states.items() if state == CONFIRMED), key=engine_order
        )
        self.tentative_ = sorted(
            (name for name, state in states.items() if state == TENTATIVE), key=engine_order
        )
        self.rejected_ = sorted(
            (name for name, state in states.items() if state == REJECTED), key=engine_order
        )
        self._resource = {
            "peak_process_rss_bytes": process_rss_bytes(),
            "minimum_available_ram_bytes": available_ram_bytes(),
        }

        # Confirmed, then tentative, then rejected -- so the published ranking
        # carries the state ordering explicitly and no mode has to invent one.
        ranking_order = self.confirmed_ + self.tentative_ + self.rejected_
        scores = {
            name: float(len(ranking_order) - position)
            for position, name in enumerate(ranking_order)
        }
        return ranking_order, scores, list(self.confirmed_)

    # -- mode-specific budget resolution -----------------------------------

    def _finalize_selection(
        self,
        *,
        ranking: Sequence[str],
        natural: Sequence[str] | None,
        universe_size: int,
    ) -> tuple[list[str], int | None, str]:
        confirmed = list(natural or [])
        self._budget_status_override = None

        if self.selection_mode == "natural_confirmed":
            # k is ignored by design: this is Boruta's own all-relevant answer.
            return confirmed, None, "not_applicable"

        requested = int(self.k or 0)
        pool = confirmed if self.selection_mode == "confirmed_top_k" else (
            confirmed + list(self.tentative_)
        )
        if len(pool) >= requested:
            return pool[:requested], requested, "satisfied"
        # Short of the budget. Rejected features are never used to pad, and under
        # confirmed_top_k neither are tentative ones.
        return pool, requested, "infeasible_natural_support"

    def _effective_selection_mode(self, budget_status: str) -> str:
        return self.selection_mode

    def _collect_warnings(self, budget_status: str) -> list[str]:
        collected = super()._collect_warnings(budget_status)
        if self.selection_mode == "natural_confirmed" and self.k is not None:
            collected.append(
                f"natural_confirmed ignores the requested budget {self.k}; Boruta's "
                "confirmed set is its own answer"
            )
        if not self.confirmed_ and self.support_states_ is not None:
            collected.append(
                "Boruta confirmed no feature; this is a valid natural outcome and no "
                "tentative or rejected feature was promoted to fill it"
            )
        if budget_status == "infeasible_natural_support":
            if self.selection_mode == "confirmed_top_k":
                collected.append(
                    f"confirmed set holds {len(self.confirmed_)} feature(s), fewer than "
                    f"the requested {self.k}; tentative features were not used as "
                    "filler because confirmed_top_k forbids it"
                )
            else:
                collected.append(
                    f"confirmed + tentative holds "
                    f"{len(self.confirmed_) + len(self.tentative_)} feature(s), fewer "
                    f"than the requested {self.k}; rejected features were not used as "
                    "filler"
                )
        if self.selection_mode == "confirmed_then_tentative":
            collected.append(
                "confirmed_then_tentative is a matched-budget adaptation, not natural "
                "Boruta support; the natural_selected field remains confirmed-only"
            )
        return collected


__all__ = [
    "BORUTA_MODES",
    "CONFIRMED",
    "DEFAULT_BORUTA_PARAMS",
    "DEFAULT_FOREST_PARAMS",
    "REJECTED",
    "TENTATIVE",
    "BorutaRandomForestSelector",
]
