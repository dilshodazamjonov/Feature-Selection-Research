"""LASSO for a binary default target: L1-penalized logistic regression.

In this project "LASSO" means L1-penalized *logistic* regression, not
least-squares Lasso. The registry identity ``lasso_l1_logistic`` says so
explicitly, because a squared-error Lasso applied to a 0/1 default flag would be
a different model with a different link and different coefficients.

The natural/matched-budget distinction is the point of this module. An L1 fit
chooses its own support size; a fixed feature budget is an external constraint.
Those two subsets are recorded separately and a budget-matched subset is never
allowed to masquerade as a natural support.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from credit_risk_fs.selectors.lightweight.contract import (
    ControlledSelectorFailure,
    LightweightSelector,
)


def _sklearn_minor_version() -> tuple[int, int]:
    try:
        major, minor = (int(part) for part in sklearn.__version__.split(".")[:2])
    except (TypeError, ValueError):  # pragma: no cover - unparsable dev version
        return (0, 0)
    return (major, minor)


#: scikit-learn 1.8 deprecated ``penalty`` in favour of ``l1_ratio`` and warns
#: that ``penalty='l1'`` alongside the default ``l1_ratio=0.0`` is inconsistent;
#: ``penalty`` is scheduled for removal in 1.10. The two spellings were verified
#: to produce bit-identical coefficients on this solver, so this selects the
#: non-deprecated spelling where it exists without changing the estimator. Older
#: releases only accept ``l1_ratio`` for elastic-net, hence the version split.
_PREFER_L1_RATIO = _sklearn_minor_version() >= (1, 8)


def _l1_penalty_kwargs() -> tuple[dict[str, Any], str]:
    if _PREFER_L1_RATIO:
        return {"l1_ratio": 1.0}, "l1_ratio=1.0"
    return {"penalty": "l1"}, "penalty='l1'"


#: ``n_jobs`` has had no effect on LogisticRegression since scikit-learn 1.8 and
#: is scheduled for removal in 1.10. The configured value is still recorded --
#: thread counts are part of the declared run contract -- but it is not forwarded
#: to an estimator that would only warn about it.
_LOGISTIC_ACCEPTS_N_JOBS = _sklearn_minor_version() < (1, 8)

#: ``liblinear`` is the default because its L1 coordinate descent is
#: deterministic given the data. ``saga`` is permitted but is stochastic and
#: therefore depends on ``random_state``.
L1_SOLVERS = frozenset({"liblinear", "saga"})

IMPUTATION_STRATEGIES = frozenset({"median", "mean", "constant"})


class L1LogisticSelector(LightweightSelector):
    """Select features by the support of an L1-penalized logistic regression.

    Imputation, scaling, and the penalized fit are all estimated on the rows
    handed to :meth:`fit` and on nothing else, so the preprocessing statistics
    cannot leak across a fold boundary.
    """

    method_id: ClassVar[str] = "lasso_l1_logistic"
    display_label: ClassVar[str] = "LASSO (L1-penalized logistic regression)"
    implementation_id: ClassVar[str] = "lasso_l1_logistic_v1"
    supervised: ClassVar[bool] = True
    supports_ranking: ClassVar[bool] = True
    supports_natural_support: ClassVar[bool] = True
    supports_fixed_budget: ClassVar[bool] = True
    default_selection_mode: ClassVar[str] = "matched_budget"
    score_orientation: ClassVar[str] = "higher_is_better"

    def __init__(
        self,
        *,
        k: int | None = None,
        C: float = 0.05,
        solver: str = "liblinear",
        max_iter: int = 2_000,
        tol: float = 1e-4,
        class_weight: str | None = None,
        coefficient_tolerance: float = 1e-10,
        imputation_strategy: str = "median",
        allow_zero_coefficient_fill: bool = False,
        random_state: int = 42,
        n_jobs: int = 1,
        excluded_columns: Sequence[str] | None = None,
        fit_scope: str = "dev_fold_training_only",
    ) -> None:
        super().__init__(
            k=k,
            random_state=random_state,
            excluded_columns=excluded_columns,
            fit_scope=fit_scope,
        )
        if str(solver) not in L1_SOLVERS:
            raise ValueError(f"solver must support an L1 penalty; got {solver!r}")
        if str(imputation_strategy) not in IMPUTATION_STRATEGIES:
            raise ValueError(
                f"imputation_strategy must be one of {sorted(IMPUTATION_STRATEGIES)}"
            )
        if float(C) <= 0.0:
            raise ValueError("C must be positive")
        if int(n_jobs) <= 0:
            raise ValueError("n_jobs must be positive")
        self.C = float(C)
        self.solver = str(solver)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.class_weight = class_weight
        self.coefficient_tolerance = float(coefficient_tolerance)
        self.imputation_strategy = str(imputation_strategy)
        self.allow_zero_coefficient_fill = bool(allow_zero_coefficient_fill)
        self.n_jobs = int(n_jobs)
        self.coefficients_: pd.DataFrame | None = None
        self.penalty_api_: str | None = None
        self.converged_: bool | None = None
        self.n_iter_: int | None = None
        self.convergence_warnings_: tuple[str, ...] = ()
        self._natural_support: list[str] = []
        self._used_zero_coefficient_fill = False

    def describe_configuration(self) -> dict[str, Any]:
        configuration = super().describe_configuration()
        configuration.update(
            {
                "penalty": "l1",
                "penalty_api": _l1_penalty_kwargs()[1],
                "sklearn_version": sklearn.__version__,
                "estimator": "sklearn.linear_model.LogisticRegression",
                "C": self.C,
                "solver": self.solver,
                "max_iter": self.max_iter,
                "tol": self.tol,
                "class_weight": self.class_weight,
                "coefficient_tolerance": self.coefficient_tolerance,
                "imputation_strategy": self.imputation_strategy,
                "scaling": "sklearn.preprocessing.StandardScaler",
                "allow_zero_coefficient_fill": self.allow_zero_coefficient_fill,
                "n_jobs": self.n_jobs,
                "n_jobs_forwarded_to_estimator": bool(
                    self.solver == "saga" and _LOGISTIC_ACCEPTS_N_JOBS
                ),
                "preprocessing_fit_scope": "supplied_training_rows_only",
            }
        )
        return configuration

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
                cause="L1 logistic regression requires the binary default target",
                configuration=self.describe_configuration(),
            )

        frame = X.loc[:, list(candidate_order)]
        non_numeric = [
            name
            for name in candidate_order
            if not pd.api.types.is_numeric_dtype(frame[name].dtype)
            and not pd.api.types.is_bool_dtype(frame[name].dtype)
        ]
        if non_numeric:
            # Silently one-hot encoding here would change the candidate-universe
            # identity and therefore the universe hash, so the caller must supply
            # an already-encoded matrix.
            raise ControlledSelectorFailure(
                method_id=self.method_id,
                stage="design_matrix_validation",
                cause=(
                    f"{len(non_numeric)} candidate feature(s) are not numeric and this "
                    f"selector does not encode them implicitly: {non_numeric[:5]}"
                ),
                configuration=self.describe_configuration(),
            )

        values = frame.to_numpy(dtype="float64", copy=True)
        values[~np.isfinite(values)] = np.nan
        all_missing = [
            name for name, column in zip(candidate_order, values.T, strict=True)
            if not np.isfinite(column).any()
        ]

        imputer = SimpleImputer(
            strategy=self.imputation_strategy,
            fill_value=0.0 if self.imputation_strategy == "constant" else None,
            keep_empty_features=True,
        )
        scaler = StandardScaler()
        design = scaler.fit_transform(imputer.fit_transform(values))
        design = np.nan_to_num(design, nan=0.0, posinf=0.0, neginf=0.0)

        penalty_kwargs, penalty_spelling = _l1_penalty_kwargs()
        self.penalty_api_ = penalty_spelling
        estimator = LogisticRegression(
            C=self.C,
            solver=self.solver,
            max_iter=self.max_iter,
            tol=self.tol,
            class_weight=self.class_weight,
            random_state=self.random_state,
            **penalty_kwargs,
            **(
                {"n_jobs": self.n_jobs}
                if self.solver == "saga" and _LOGISTIC_ACCEPTS_N_JOBS
                else {}
            ),
        )
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            estimator.fit(design, y.to_numpy())
        self.convergence_warnings_ = tuple(
            str(item.message) for item in captured
            if issubclass(item.category, ConvergenceWarning)
        )
        # Anything that is not a convergence warning is re-raised to the caller.
        # Swallowing it here would hide genuine signals -- an estimator
        # deprecation, a dtype problem, a numerical instability -- behind a
        # context manager that exists only to capture convergence state.
        for item in captured:
            if not issubclass(item.category, ConvergenceWarning):
                warnings.warn_explicit(
                    item.message,
                    item.category,
                    item.filename,
                    item.lineno,
                )

        iterations = np.asarray(estimator.n_iter_).ravel()
        self.n_iter_ = int(iterations.max()) if iterations.size else None
        self.converged_ = not self.convergence_warnings_
        # A convergence failure is reported, never repaired by quietly changing
        # the solver, tolerance, penalty, sample, or feature set.

        coefficients = np.asarray(estimator.coef_).ravel()
        signed = {name: float(value) for name, value in zip(candidate_order, coefficients, strict=True)}
        absolute = {name: abs(value) for name, value in signed.items()}
        self.coefficients_ = pd.DataFrame(
            {
                "feature": list(candidate_order),
                "coefficient": [signed[name] for name in candidate_order],
                "absolute_coefficient": [absolute[name] for name in candidate_order],
                "non_zero": [
                    absolute[name] > self.coefficient_tolerance for name in candidate_order
                ],
                "all_missing_in_training": [name in set(all_missing) for name in candidate_order],
            }
        )

        # Full coefficient ranking over the universe; the natural support is the
        # prefix of it whose magnitude clears the documented tolerance.
        ranking = sorted(candidate_order, key=lambda name: (-absolute[name], name))
        self._natural_support = [
            name for name in ranking if absolute[name] > self.coefficient_tolerance
        ]
        return ranking, absolute, list(self._natural_support)

    def _finalize_selection(
        self,
        *,
        ranking: Sequence[str],
        natural: Sequence[str] | None,
        universe_size: int,
    ) -> tuple[list[str], int | None, str]:
        natural_support = list(natural or [])
        self._used_zero_coefficient_fill = False

        if self.k is None:
            return natural_support, None, "not_applicable"

        requested = int(self.k)
        effective = min(requested, universe_size)
        clipped = requested > universe_size

        if len(natural_support) >= effective:
            selected = natural_support[:effective]
        elif self.allow_zero_coefficient_fill:
            # Explicitly predeclared mode: extend past the natural support using
            # the absolute-coefficient ranking. Recorded as a distinct selection
            # mode so it can never be read as an L1 support.
            self._used_zero_coefficient_fill = True
            selected = list(ranking)[:effective]
        else:
            return natural_support, requested, "infeasible_natural_support"

        return selected, requested, "clipped_to_universe" if clipped else "satisfied"

    def _effective_selection_mode(self, budget_status: str) -> str:
        if self.k is None:
            return "natural"
        if self._used_zero_coefficient_fill:
            return "coefficient_ranking"
        return "matched_budget"

    def _collect_warnings(self, budget_status: str) -> list[str]:
        collected = super()._collect_warnings(budget_status)
        if budget_status == "infeasible_natural_support":
            collected.append(
                f"L1 natural support holds {len(self._natural_support)} feature(s), "
                f"fewer than the requested {self.k}; zero-coefficient features were "
                "not used as filler because allow_zero_coefficient_fill is False"
            )
        if not self._natural_support:
            collected.append(
                "L1 penalty drove every coefficient to zero; this is a valid "
                "outcome and no substitute selector was applied"
            )
        if self.convergence_warnings_:
            collected.append(
                f"solver did not converge in max_iter={self.max_iter}: "
                f"{self.convergence_warnings_[0]}"
            )
        return collected


__all__ = ["IMPUTATION_STRATEGIES", "L1_SOLVERS", "L1LogisticSelector"]
