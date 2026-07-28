"""CatBoost native SHAP ranking.

This is a new method: the repository had no SHAP path at all before Prompt 8.
`CatBoostModel.get_feature_importance()` is called with no arguments, which is
CatBoost's default PredictionValuesChange -- a different quantity entirely.

Definition implemented here, all of it recorded in the implementation id:

    feature importance type : EFstrType.ShapValues   (native CatBoost)
    SHAP calculation type   : Regular
    model output scale      : native raw model output
    aggregation             : mean absolute SHAP over explanation rows
    expected-value column   : the trailing column, excluded from the ranking

There is no fallback. If native SHAP is unavailable or returns something
unusable, the selector fails explicitly rather than quietly substituting
PredictionValuesChange, impurity importance, permutation importance, or a generic
``shap`` package result -- each of which would be a different method wearing this
method's name. Any Exact/Approximate, interventional, reference-data,
probability-output, or multiclass variant requires a new implementation id.
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
    ordered_rows_hash,
    process_rss_bytes,
)
from credit_risk_fs.selectors.lightweight.contract import (
    ControlledSelectorFailure,
    LightweightSelector,
)

logger = logging.getLogger(__name__)

SHAP_IMPORTANCE_TYPE = "ShapValues"
SHAP_CALC_TYPE = "Regular"
SHAP_AGGREGATION = "mean_absolute_shap_over_explanation_rows"
SHAP_MODEL_OUTPUT = "native_raw_model_output"

DEFAULT_CATBOOST_PARAMS: dict[str, Any] = {
    "iterations": 500,
    "depth": 6,
    "learning_rate": 0.05,
    "verbose": False,
    "allow_writing_files": False,
    "task_type": "CPU",
}


class CatBoostShapSelector(LightweightSelector):
    """Rank features by mean absolute native CatBoost SHAP value.

    Both the model fit and the explanation sample come from the rows handed to
    :meth:`fit`. There is no defensible natural threshold on a SHAP magnitude in
    this project, so ``k`` is required and ``natural_selected`` stays absent.
    """

    method_id: ClassVar[str] = "catboost_shap"
    display_label: ClassVar[str] = "CatBoost-SHAP"
    implementation_id: ClassVar[str] = (
        "catboost_native_shap_regular_mean_abs_train_sample_v1"
    )
    supervised: ClassVar[bool] = True
    supports_ranking: ClassVar[bool] = True
    supports_natural_support: ClassVar[bool] = False
    supports_fixed_budget: ClassVar[bool] = True
    default_selection_mode: ClassVar[str] = "matched_budget"
    score_orientation: ClassVar[str] = "higher_is_better"

    def __init__(
        self,
        *,
        k: int | None = None,
        explanation_sample_size: int | None = 10_000,
        catboost_params: dict[str, Any] | None = None,
        random_state: int = 42,
        explanation_sample_seed: int | None = None,
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
        if explanation_sample_size is not None and int(explanation_sample_size) <= 0:
            raise ValueError("explanation_sample_size must be positive or None")
        if int(thread_count) <= 0:
            raise ValueError("thread_count must be positive")
        self.explanation_sample_size = (
            None if explanation_sample_size is None else int(explanation_sample_size)
        )
        self.explanation_sample_seed = (
            int(random_state) if explanation_sample_seed is None
            else int(explanation_sample_seed)
        )
        self.thread_count = int(thread_count)
        self.catboost_params = dict(DEFAULT_CATBOOST_PARAMS)
        if catboost_params:
            self.catboost_params.update(catboost_params)
        self.shap_scores_: dict[str, float] | None = None
        self.explanation_sample_: dict[str, Any] | None = None
        self._resource: dict[str, Any] = {}

    # -- configuration -----------------------------------------------------

    def _effective_catboost_params(self) -> dict[str, Any]:
        params = dict(self.catboost_params)
        params.update(
            {
                "random_seed": self.random_state,
                "thread_count": self.thread_count,
                "allow_writing_files": False,
                "verbose": False,
            }
        )
        params.setdefault("task_type", "CPU")
        return params

    def describe_configuration(self) -> dict[str, Any]:
        configuration = super().describe_configuration()
        configuration.update(
            {
                "estimator": "catboost.CatBoostClassifier",
                "estimator_params": self._effective_catboost_params(),
                "feature_importance_type": SHAP_IMPORTANCE_TYPE,
                "shap_calc_type": SHAP_CALC_TYPE,
                "model_output": SHAP_MODEL_OUTPUT,
                "aggregation": SHAP_AGGREGATION,
                "expected_value_column": "trailing_column_excluded_from_ranking",
                "explanation_sample_size": self.explanation_sample_size,
                "explanation_sample_seed": self.explanation_sample_seed,
                "explanation_sample_scope": "selector_training_partition_only",
                "explanation_sample_rule": (
                    "deterministic_stratified_without_replacement_local_rng"
                ),
                "thread_count": self.thread_count,
                "natural_support": "unsupported_ranking_method",
                "fallback_importance": "none_permitted",
            }
        )
        return configuration

    def _estimator_config_sha256(self) -> str | None:
        return estimator_config_hash(
            {
                **self._effective_catboost_params(),
                "shap_calc_type": SHAP_CALC_TYPE,
                "feature_importance_type": SHAP_IMPORTANCE_TYPE,
                "explanation_sample_size": self.explanation_sample_size,
                "explanation_sample_seed": self.explanation_sample_seed,
                "requested_k": self.k,
            }
        )

    def _heavy_metadata(self) -> dict[str, Any]:
        return {
            "cost_class": "heavy",
            "feature_importance_type": SHAP_IMPORTANCE_TYPE,
            "shap_calc_type": SHAP_CALC_TYPE,
            "model_output": SHAP_MODEL_OUTPUT,
            "aggregation": SHAP_AGGREGATION,
            "explanation_sample": dict(self.explanation_sample_ or {}),
            "shap_scores": dict(self.shap_scores_ or {}),
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
                    "SHAP magnitude has no defensible natural selection threshold in "
                    "this project, so a fixed feature budget k is required; k=None is "
                    "unsupported and no natural support is fabricated"
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

    # -- explanation sample ------------------------------------------------

    def _draw_explanation_sample(self, y: pd.Series) -> np.ndarray:
        """Positions of the explanation rows, drawn from training rows only.

        Stratified on the target so the sample keeps the training prevalence, and
        seeded from a local generator so the draw is reproducible and cannot be
        perturbed by unrelated global RNG use.
        """

        total = len(y)
        requested = self.explanation_sample_size
        if requested is None or requested >= total:
            return np.arange(total)

        generator = np.random.default_rng(self.explanation_sample_seed)
        labels = np.asarray(y.to_numpy())
        positive = np.flatnonzero(labels == 1)
        negative = np.flatnonzero(labels != 1)
        if positive.size == 0 or negative.size == 0:
            return np.sort(generator.choice(total, size=requested, replace=False))

        # Allocate proportionally, then guarantee at least one row per class so a
        # rare-event sample can never collapse to a single class.
        positive_target = int(round(requested * positive.size / total))
        positive_target = max(1, min(positive_target, positive.size, requested - 1))
        negative_target = requested - positive_target
        if negative_target > negative.size:
            negative_target = negative.size
            positive_target = min(requested - negative_target, positive.size)

        drawn = np.concatenate(
            [
                generator.choice(positive, size=positive_target, replace=False),
                generator.choice(negative, size=negative_target, replace=False),
            ]
        )
        return np.sort(drawn)

    # -- computation -------------------------------------------------------

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
                cause="CatBoost-SHAP requires the binary default target",
                configuration=self.describe_configuration(),
            )

        frame = X.loc[:, list(candidate_order)]
        target = y.reset_index(drop=True)
        model = self._fit_model(frame, target)
        positions = self._draw_explanation_sample(target)

        sample_frame = frame.iloc[positions]
        sample_target = target.iloc[positions]
        identity = [str(value) for value in X.index.to_numpy()[positions]]
        self.explanation_sample_ = {
            "requested_size": self.explanation_sample_size,
            "realized_size": int(len(positions)),
            "training_row_count": int(len(frame)),
            "used_all_training_rows": bool(len(positions) == len(frame)),
            "positive_count": int((sample_target == 1).sum()),
            "negative_count": int((sample_target != 1).sum()),
            "training_positive_rate": round(float((target == 1).mean()), 9),
            "sample_positive_rate": round(float((sample_target == 1).mean()), 9),
            "seed": self.explanation_sample_seed,
            "row_identity_sha256": ordered_rows_hash(identity),
            "scope": "selector_training_partition_only",
        }

        scores = self._shap_scores(model, sample_frame, sample_target, candidate_order)
        self.shap_scores_ = dict(scores)
        self._resource = {
            "peak_process_rss_bytes": process_rss_bytes(),
            "minimum_available_ram_bytes": available_ram_bytes(),
        }

        order_index = {str(name): position for position, name in enumerate(candidate_order)}
        ranking = sorted(
            candidate_order, key=lambda name: (-scores[str(name)], order_index[str(name)])
        )
        return [str(name) for name in ranking], scores, None

    def _fit_model(self, frame: pd.DataFrame, target: pd.Series) -> Any:
        from catboost import CatBoostClassifier

        with heavy_stage(
            logger,
            method_id=self.method_id,
            stage="catboost_fit",
            detail=f"rows={len(frame)} features={frame.shape[1]}",
        ) as observations:
            model = CatBoostClassifier(**self._effective_catboost_params())
            try:
                model.fit(frame, target)
            except Exception as error:
                raise ControlledSelectorFailure(
                    method_id=self.method_id,
                    stage="estimator_fit",
                    cause=(
                        f"CatBoost training failed on {frame.shape[1]} feature(s): "
                        f"{type(error).__name__}: {error}"
                    ),
                    configuration=self.describe_configuration(),
                ) from error
            observations["tree_count"] = int(getattr(model, "tree_count_", 0) or 0)
        return model

    def _shap_scores(
        self,
        model: Any,
        sample_frame: pd.DataFrame,
        sample_target: pd.Series,
        candidate_order: Sequence[str],
    ) -> dict[str, float]:
        from catboost import EFstrType, Pool

        expected_columns = len(candidate_order) + 1
        with heavy_stage(
            logger,
            method_id=self.method_id,
            stage="native_shap_values",
            detail=f"rows={len(sample_frame)} calc_type={SHAP_CALC_TYPE}",
        ) as observations:
            try:
                raw = model.get_feature_importance(
                    Pool(sample_frame, sample_target),
                    type=EFstrType.ShapValues,
                    shap_calc_type=SHAP_CALC_TYPE,
                    thread_count=self.thread_count,
                )
            except Exception as error:
                raise ControlledSelectorFailure(
                    method_id=self.method_id,
                    stage="shap_calculation",
                    cause=(
                        f"native CatBoost {SHAP_IMPORTANCE_TYPE} "
                        f"({SHAP_CALC_TYPE}) failed: {type(error).__name__}: {error}; "
                        "no substitute importance is permitted"
                    ),
                    configuration=self.describe_configuration(),
                ) from error

            values = np.asarray(raw, dtype="float64")
            if values.ndim != 2 or values.shape != (len(sample_frame), expected_columns):
                raise ControlledSelectorFailure(
                    method_id=self.method_id,
                    stage="shap_shape_validation",
                    cause=(
                        f"expected SHAP array of shape "
                        f"{(len(sample_frame), expected_columns)} "
                        f"(rows, features + expected-value column); got {values.shape}"
                    ),
                    configuration=self.describe_configuration(),
                )
            if not np.isfinite(values).all():
                raise ControlledSelectorFailure(
                    method_id=self.method_id,
                    stage="shap_value_validation",
                    cause=(
                        f"native SHAP returned "
                        f"{int((~np.isfinite(values)).sum())} non-finite value(s)"
                    ),
                    configuration=self.describe_configuration(),
                )

            # Drop the trailing expected-value (base) column before aggregating.
            contributions = values[:, :-1]
            aggregate = np.abs(contributions).mean(axis=0)
            observations["features"] = int(aggregate.shape[0])

        scores = {
            str(name): float(value)
            for name, value in zip(candidate_order, aggregate, strict=True)
        }
        return scores

    def _collect_warnings(self, budget_status: str) -> list[str]:
        collected = super()._collect_warnings(budget_status)
        scores = self.shap_scores_ or {}
        if scores and all(value == 0.0 for value in scores.values()):
            collected.append(
                "all_scores_zero: every mean absolute SHAP value is zero, so the "
                "ranking is the authenticated candidate order and carries no "
                "evidence of informativeness"
            )
        sample = self.explanation_sample_ or {}
        if sample.get("requested_size") and sample.get("used_all_training_rows"):
            collected.append(
                f"requested explanation sample {sample['requested_size']} exceeds the "
                f"{sample['training_row_count']} available training rows; every "
                "training row was used"
            )
        return collected


__all__ = [
    "DEFAULT_CATBOOST_PARAMS",
    "SHAP_AGGREGATION",
    "SHAP_CALC_TYPE",
    "SHAP_IMPORTANCE_TYPE",
    "SHAP_MODEL_OUTPUT",
    "CatBoostShapSelector",
]
