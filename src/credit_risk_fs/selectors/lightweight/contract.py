"""Shared contract for the Prompt 7 lightweight selector controls.

This module does not introduce a second selector framework. Every selector here
still implements the repository's existing :class:`FeatureSelector` protocol
(``fit`` / ``transform`` / ``fit_transform`` plus ``selected_features_``), so the
fold runner in :mod:`credit_risk_fs.models._fold` consumes them unchanged. What
this module adds is the *evidence record* that the existing protocol has no room
for: accurate method identity, a complete ranking with score orientation, the
distinction between a natural subset and a budget-matched subset, explicit
budget feasibility, the fit boundary that produced the result, and deterministic
tie handling.

The field vocabulary deliberately mirrors the frozen voting ranking schema in
:mod:`credit_risk_fs.experiments.rank_voting` (``candidate_universe_sha256``,
``fit_scope``, ``score_direction``, ``seed``) so that Prompt 7 artifacts read as
the same repository rather than as a near-duplicate parallel schema. It does not
reuse that builder: ``build_long_voter_ranking_frame`` is hard-wired to the two
frozen voters and validates an exact ``2 * universe_size`` row count, and it is
part of a frozen protocol that Prompt 7 must not touch.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from credit_risk_fs.selectors.base import (
    SelectedFeaturesMixin,
    select_feature_frame,
    validate_feature_frame,
)
from credit_risk_fs.utils.hashing import sha256_text

CONTRACT_VERSION = "lightweight_selector_contract_v1"

#: Fit boundary shared with the frozen voting protocol's ``REQUIRED_FIT_SCOPE``.
DEV_FOLD_TRAINING_ONLY = "dev_fold_training_only"

#: ``natural``            algorithm decides its own subset size
#: ``matched_budget``     top-k of the algorithm's own ranking
#: ``full_control``       every eligible candidate feature, no selection
#: ``random_control``     chance-performance control, target never inspected
#: ``coefficient_ranking`` explicitly predeclared opt-in that allows a
#:                        budget-matched subset to extend past a model's natural
#:                        support. Named separately so a padded subset can never
#:                        be mistaken for a natural support.
SELECTION_MODES = frozenset(
    {
        "natural",
        "matched_budget",
        "full_control",
        "random_control",
        "coefficient_ranking",
    }
)

#: ``satisfied``                   actual count equals the request
#: ``clipped_to_universe``         request exceeded the eligible universe, so the
#:                                whole universe was returned. This preserves the
#:                                pre-existing repository policy in
#:                                ``selectors.base.resolve_feature_budget``
#:                                (``min(requested, available)``) instead of
#:                                inventing a new one, and records the clip.
#: ``not_applicable``              mode ignores a budget (``full_control``)
#: ``infeasible_natural_support``  a model's natural support is smaller than the
#:                                requested budget and padding was not
#:                                predeclared, so the budget was NOT met
#: ``empty_universe``              nothing was eligible
BUDGET_STATUSES = frozenset(
    {
        "satisfied",
        "clipped_to_universe",
        "not_applicable",
        "infeasible_natural_support",
        "empty_universe",
    }
)

SCORE_ORIENTATIONS = frozenset(
    {"higher_is_better", "lower_is_better", "rank_1_is_best", "not_applicable"}
)

#: One documented tie rule for every ranking selector in this package. It is a
#: pure function of the score and the feature name, so it cannot inherit
#: incidental DataFrame column order or dictionary insertion order.
TIE_RULE = "descending_score_then_ascending_feature_name"

#: Tie rule for the controls that produce no comparable score.
TIE_RULE_UNIVERSE_ORDER = "candidate_universe_order_preserved"

LONG_FRAME_COLUMNS = (
    "method_id",
    "implementation_id",
    "display_label",
    "selection_mode",
    "supervised",
    "fit_scope",
    "feature",
    "rank",
    "raw_score",
    "score_orientation",
    "natural_selected",
    "matched_budget_selected",
    "requested_budget",
    "actual_selected_count",
    "budget_status",
    "seed",
    "tie_rule",
    "candidate_universe_sha256",
    "candidate_universe_count",
    "training_identity_sha256",
    "contract_version",
)


class SelectorContractError(ValueError):
    """Raised when a selector result violates a frozen contract invariant."""


class ControlledSelectorFailure(RuntimeError):
    """Explicit, attributable selector failure.

    Carries the method, stage, cause, and configuration so that a failure is
    always reportable evidence. There is deliberately no fallback path: a
    selector that cannot run must say so rather than quietly degrade into a
    different algorithm.
    """

    def __init__(
        self,
        *,
        method_id: str,
        stage: str,
        cause: str,
        configuration: Mapping[str, Any] | None = None,
    ) -> None:
        self.method_id = str(method_id)
        self.stage = str(stage)
        self.cause = str(cause)
        self.configuration = dict(configuration or {})
        super().__init__(
            f"{self.method_id} failed at stage '{self.stage}': {self.cause} "
            f"(configuration={json.dumps(self.configuration, sort_keys=True, default=str)})"
        )


def ordered_name_hash(values: Iterable[str]) -> str:
    """Hash an ordered feature-name sequence.

    Byte-identical to ``rank_voting._ordered_name_hash`` so a Prompt 7 universe
    hash is directly comparable with a frozen voting universe hash.
    """

    return sha256_text(
        json.dumps([str(value) for value in values], ensure_ascii=False, separators=(",", ":"))
    )


def training_identity_hash(X: pd.DataFrame, y: pd.Series | None) -> str:
    """Hash the exact training rows, and target, a selector was allowed to see.

    This is the evidence that a supervised fit stayed inside its fold boundary.
    Two selectors handed different row sets cannot produce the same hash, so a
    leaked validation or OOT row is detectable after the fact.
    """

    index_payload = [str(value) for value in X.index.to_list()]
    if y is None:
        target_payload: list[str] = []
    else:
        target_payload = [str(value) for value in np.asarray(y).tolist()]
    return sha256_text(
        json.dumps(
            {"rows": index_payload, "target": target_payload},
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


def rank_by_score(
    scores: Mapping[str, float],
    *,
    candidate_order: Sequence[str],
) -> list[str]:
    """Order features by descending score, breaking ties on the feature name.

    ``candidate_order`` supplies the eligible universe; anything absent from
    ``scores`` is treated as unscored and excluded. Sorting on the name rather
    than on position is what makes the result independent of the caller's column
    order: reordering the input DataFrame cannot reorder the output.
    """

    eligible = [str(name) for name in candidate_order if str(name) in scores]
    return sorted(
        eligible,
        key=lambda name: (-float(scores[name]), name),
    )


@dataclass(frozen=True)
class SelectionResult:
    """Complete, serializable evidence for one selector fit.

    Invariants are enforced on construction rather than at write time, so an
    invalid result cannot reach an artifact.
    """

    method_id: str
    display_label: str
    implementation_id: str
    selection_mode: str
    supervised: bool
    selected_features: tuple[str, ...]
    candidate_universe: tuple[str, ...]
    requested_budget: int | None
    budget_status: str
    score_orientation: str
    tie_rule: str
    fit_scope: str = DEV_FOLD_TRAINING_ONLY
    seed: int | None = None
    configuration: Mapping[str, Any] = field(default_factory=dict)
    ranking: tuple[str, ...] | None = None
    raw_scores: Mapping[str, float] | None = None
    natural_selected: tuple[str, ...] | None = None
    training_row_count: int | None = None
    training_identity_sha256: str | None = None
    fit_seconds: float = 0.0
    warnings: tuple[str, ...] = ()
    failure_reason: str | None = None
    contract_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.selection_mode not in SELECTION_MODES:
            raise SelectorContractError(
                f"unknown selection mode {self.selection_mode!r}; "
                f"expected one of {sorted(SELECTION_MODES)}"
            )
        if self.budget_status not in BUDGET_STATUSES:
            raise SelectorContractError(
                f"unknown budget status {self.budget_status!r}; "
                f"expected one of {sorted(BUDGET_STATUSES)}"
            )
        if self.score_orientation not in SCORE_ORIENTATIONS:
            raise SelectorContractError(
                f"unknown score orientation {self.score_orientation!r}"
            )

        universe = list(self.candidate_universe)
        if len(universe) != len(set(universe)):
            raise SelectorContractError("candidate universe contains duplicate names")
        universe_set = set(universe)

        selected = list(self.selected_features)
        if len(selected) != len(set(selected)):
            raise SelectorContractError(
                f"{self.method_id} produced duplicate selected features"
            )
        outside = [name for name in selected if name not in universe_set]
        if outside:
            raise SelectorContractError(
                f"{self.method_id} selected {len(outside)} feature(s) outside the "
                f"authenticated candidate universe: {outside[:5]}"
            )

        if self.ranking is not None:
            ranking = list(self.ranking)
            if len(ranking) != len(set(ranking)):
                raise SelectorContractError(f"{self.method_id} ranking contains duplicates")
            ranked_outside = [name for name in ranking if name not in universe_set]
            if ranked_outside:
                raise SelectorContractError(
                    f"{self.method_id} ranked {len(ranked_outside)} feature(s) outside "
                    f"the candidate universe: {ranked_outside[:5]}"
                )
            # A selected feature that is absent from the ranking would make the
            # published rank column silently incomplete.
            unranked = [name for name in selected if name not in set(ranking)]
            if unranked:
                raise SelectorContractError(
                    f"{self.method_id} selected {len(unranked)} feature(s) that carry no "
                    f"rank: {unranked[:5]}"
                )

        if self.natural_selected is not None:
            natural_outside = [
                name for name in self.natural_selected if name not in universe_set
            ]
            if natural_outside:
                raise SelectorContractError(
                    f"{self.method_id} natural support falls outside the candidate "
                    f"universe: {natural_outside[:5]}"
                )

        if self.requested_budget is not None and int(self.requested_budget) < 0:
            raise SelectorContractError("requested budget must be non-negative")

        if self.budget_status == "satisfied" and self.requested_budget is not None:
            if len(selected) != int(self.requested_budget):
                raise SelectorContractError(
                    f"{self.method_id} reports budget_status='satisfied' but selected "
                    f"{len(selected)} of {self.requested_budget} requested features"
                )
        if self.budget_status == "not_applicable" and self.requested_budget is not None:
            raise SelectorContractError(
                "budget_status='not_applicable' requires requested_budget=None"
            )

    # -- derived -----------------------------------------------------------

    @property
    def actual_selected_count(self) -> int:
        return len(self.selected_features)

    @property
    def candidate_universe_count(self) -> int:
        return len(self.candidate_universe)

    @property
    def candidate_universe_sha256(self) -> str:
        return ordered_name_hash(self.candidate_universe)

    @property
    def natural_selected_count(self) -> int | None:
        return None if self.natural_selected is None else len(self.natural_selected)

    # -- serialization -----------------------------------------------------

    def to_long_frame(self) -> pd.DataFrame:
        """One row per ranked (or selected) feature, in the common schema."""

        ordered = list(self.ranking) if self.ranking is not None else list(self.selected_features)
        selected_set = set(self.selected_features)
        natural_set = None if self.natural_selected is None else set(self.natural_selected)
        scores = dict(self.raw_scores or {})
        rows = [
            {
                "method_id": self.method_id,
                "implementation_id": self.implementation_id,
                "display_label": self.display_label,
                "selection_mode": self.selection_mode,
                "supervised": self.supervised,
                "fit_scope": self.fit_scope,
                "feature": name,
                "rank": position,
                "raw_score": scores.get(name),
                "score_orientation": self.score_orientation,
                "natural_selected": (
                    None if natural_set is None else name in natural_set
                ),
                "matched_budget_selected": name in selected_set,
                "requested_budget": self.requested_budget,
                "actual_selected_count": self.actual_selected_count,
                "budget_status": self.budget_status,
                "seed": self.seed,
                "tie_rule": self.tie_rule,
                "candidate_universe_sha256": self.candidate_universe_sha256,
                "candidate_universe_count": self.candidate_universe_count,
                "training_identity_sha256": self.training_identity_sha256,
                "contract_version": self.contract_version,
            }
            for position, name in enumerate(ordered, start=1)
        ]
        return pd.DataFrame(rows, columns=list(LONG_FRAME_COLUMNS))

    def to_dict(self) -> dict[str, Any]:
        """Round-trippable payload preserving identity, order, scores, hashes."""

        return {
            "contract_version": self.contract_version,
            "method_id": self.method_id,
            "display_label": self.display_label,
            "implementation_id": self.implementation_id,
            "selection_mode": self.selection_mode,
            "supervised": bool(self.supervised),
            "fit_scope": self.fit_scope,
            "seed": self.seed,
            "configuration": dict(self.configuration),
            "candidate_universe": list(self.candidate_universe),
            "candidate_universe_count": self.candidate_universe_count,
            "candidate_universe_sha256": self.candidate_universe_sha256,
            "requested_budget": self.requested_budget,
            "selected_features": list(self.selected_features),
            "actual_selected_count": self.actual_selected_count,
            "budget_status": self.budget_status,
            "ranking": None if self.ranking is None else list(self.ranking),
            "raw_scores": None if self.raw_scores is None else dict(self.raw_scores),
            "score_orientation": self.score_orientation,
            "natural_selected": (
                None if self.natural_selected is None else list(self.natural_selected)
            ),
            "natural_selected_count": self.natural_selected_count,
            "tie_rule": self.tie_rule,
            "training_row_count": self.training_row_count,
            "training_identity_sha256": self.training_identity_sha256,
            "fit_seconds": float(self.fit_seconds),
            "warnings": list(self.warnings),
            "failure_reason": self.failure_reason,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SelectionResult:
        version = str(payload.get("contract_version", ""))
        if version != CONTRACT_VERSION:
            raise SelectorContractError(
                f"unsupported selector contract version {version!r}; "
                f"expected {CONTRACT_VERSION!r}"
            )
        ranking = payload.get("ranking")
        scores = payload.get("raw_scores")
        natural = payload.get("natural_selected")
        return cls(
            method_id=str(payload["method_id"]),
            display_label=str(payload["display_label"]),
            implementation_id=str(payload["implementation_id"]),
            selection_mode=str(payload["selection_mode"]),
            supervised=bool(payload["supervised"]),
            selected_features=tuple(str(name) for name in payload["selected_features"]),
            candidate_universe=tuple(str(name) for name in payload["candidate_universe"]),
            requested_budget=(
                None if payload.get("requested_budget") is None
                else int(payload["requested_budget"])
            ),
            budget_status=str(payload["budget_status"]),
            score_orientation=str(payload["score_orientation"]),
            tie_rule=str(payload["tie_rule"]),
            fit_scope=str(payload.get("fit_scope", DEV_FOLD_TRAINING_ONLY)),
            seed=None if payload.get("seed") is None else int(payload["seed"]),
            configuration=dict(payload.get("configuration") or {}),
            ranking=None if ranking is None else tuple(str(name) for name in ranking),
            raw_scores=(
                None if scores is None
                else {str(name): float(value) for name, value in scores.items()}
            ),
            natural_selected=(
                None if natural is None else tuple(str(name) for name in natural)
            ),
            training_row_count=(
                None if payload.get("training_row_count") is None
                else int(payload["training_row_count"])
            ),
            training_identity_sha256=payload.get("training_identity_sha256"),
            fit_seconds=float(payload.get("fit_seconds", 0.0)),
            warnings=tuple(str(item) for item in payload.get("warnings") or ()),
            failure_reason=payload.get("failure_reason"),
        )

    @classmethod
    def from_json(cls, text: str) -> SelectionResult:
        return cls.from_dict(json.loads(text))


class LightweightSelector(SelectedFeaturesMixin):
    """Base class implementing the shared fit/validate/record cycle.

    Subclasses implement :meth:`_compute`, which returns the ordered ranking,
    optional per-feature scores, and an optional natural support. All budget
    resolution, invariant checking, timing, hashing, and leakage guarding
    happens here so no individual selector can quietly diverge from the
    contract.
    """

    method_id: ClassVar[str]
    display_label: ClassVar[str]
    implementation_id: ClassVar[str]
    supervised: ClassVar[bool] = True
    supports_ranking: ClassVar[bool] = True
    supports_natural_support: ClassVar[bool] = False
    supports_fixed_budget: ClassVar[bool] = True
    default_selection_mode: ClassVar[str] = "matched_budget"
    score_orientation: ClassVar[str] = "higher_is_better"
    tie_rule: ClassVar[str] = TIE_RULE

    def __init__(
        self,
        *,
        k: int | None = None,
        random_state: int = 42,
        excluded_columns: Sequence[str] | None = None,
        fit_scope: str = DEV_FOLD_TRAINING_ONLY,
    ) -> None:
        self.k = None if k is None else int(k)
        self.random_state = int(random_state)
        self.excluded_columns = tuple(str(name) for name in (excluded_columns or ()))
        self.fit_scope = str(fit_scope)
        self.selected_features_ = None
        self.result_: SelectionResult | None = None

    # -- subclass hook -----------------------------------------------------

    def _compute(
        self,
        X: pd.DataFrame,
        y: pd.Series | None,
        *,
        candidate_order: Sequence[str],
    ) -> tuple[list[str], dict[str, float] | None, list[str] | None]:
        """Return ``(ranking, raw_scores, natural_support)``."""

        raise NotImplementedError

    def describe_configuration(self) -> dict[str, Any]:
        """Configuration recorded verbatim in the result and in artifacts."""

        return {
            "k": self.k,
            "random_state": self.random_state,
            "fit_scope": self.fit_scope,
            "excluded_columns": list(self.excluded_columns),
        }

    # -- shared machinery --------------------------------------------------

    def _guard_excluded(self, names: Sequence[str]) -> None:
        present = sorted(set(names) & set(self.excluded_columns))
        if present:
            raise ControlledSelectorFailure(
                method_id=self.method_id,
                stage="candidate_universe_validation",
                cause=(
                    "identity, target, split, time, or leakage-excluded columns were "
                    f"offered as candidate features: {present[:10]}"
                ),
                configuration=self.describe_configuration(),
            )

    def _require_target(self, y: pd.Series | None) -> pd.Series:
        if y is None:
            raise ControlledSelectorFailure(
                method_id=self.method_id,
                stage="target_validation",
                cause="supervised selector received no target labels",
                configuration=self.describe_configuration(),
            )
        target = pd.Series(np.asarray(y)).reset_index(drop=True)
        finite = target.dropna()
        if len(finite) != len(target):
            raise ControlledSelectorFailure(
                method_id=self.method_id,
                stage="target_validation",
                cause="target contains missing values",
                configuration=self.describe_configuration(),
            )
        observed = set(pd.unique(finite))
        if not observed.issubset({0, 1, 0.0, 1.0, True, False}):
            raise ControlledSelectorFailure(
                method_id=self.method_id,
                stage="target_validation",
                cause=(
                    "binary default target must be encoded as 0/1 with 1 = default; "
                    f"observed {sorted(map(str, observed))[:6]}"
                ),
                configuration=self.describe_configuration(),
            )
        if len(observed) < 2:
            raise ControlledSelectorFailure(
                method_id=self.method_id,
                stage="target_validation",
                cause="target has a single class in the supplied training partition",
                configuration=self.describe_configuration(),
            )
        return target.astype(int)

    def _resolve_budget(
        self,
        *,
        ranking: Sequence[str],
        universe_size: int,
    ) -> tuple[list[str], int | None, str]:
        """Apply the requested budget under the frozen clamp policy."""

        if universe_size == 0:
            return [], self.k, "empty_universe"
        if self.k is None:
            return list(ranking), None, "not_applicable"

        requested = int(self.k)
        if requested > universe_size:
            # Preserve the pre-existing repository clamp
            # (selectors.base.resolve_feature_budget) rather than padding,
            # duplicating, or redefining the budget, and record the clip.
            return list(ranking), requested, "clipped_to_universe"
        selected = list(ranking)[:requested]
        status = "satisfied" if len(selected) == requested else "infeasible_natural_support"
        return selected, requested, status

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> LightweightSelector:
        started = time.perf_counter()
        names = validate_feature_frame(X)
        self._guard_excluded(names)

        target = self._require_target(y) if self.supervised else None
        if not self.supervised and y is not None:
            # A control that must not see labels never receives them, even if a
            # caller passes them positionally through the shared fold runner.
            target = None

        if not names:
            self.result_ = SelectionResult(
                method_id=self.method_id,
                display_label=self.display_label,
                implementation_id=self.implementation_id,
                selection_mode=self.default_selection_mode,
                supervised=self.supervised,
                selected_features=(),
                candidate_universe=(),
                requested_budget=self.k,
                budget_status="empty_universe",
                score_orientation=self.score_orientation,
                tie_rule=self.tie_rule,
                fit_scope=self.fit_scope,
                seed=self.random_state,
                configuration=self.describe_configuration(),
                ranking=(),
                raw_scores={},
                training_row_count=int(len(X)),
                training_identity_sha256=training_identity_hash(X, target),
                fit_seconds=time.perf_counter() - started,
                warnings=("no eligible candidate features were supplied",),
            )
            self.selected_features_ = []
            return self

        ranking, scores, natural = self._compute(X, target, candidate_order=names)
        selected, requested, status = self._finalize_selection(
            ranking=ranking,
            natural=natural,
            universe_size=len(names),
        )

        self.result_ = SelectionResult(
            method_id=self.method_id,
            display_label=self.display_label,
            implementation_id=self.implementation_id,
            selection_mode=self._effective_selection_mode(status),
            supervised=self.supervised,
            selected_features=tuple(selected),
            candidate_universe=tuple(names),
            requested_budget=requested,
            budget_status=status,
            score_orientation=self.score_orientation,
            tie_rule=self.tie_rule,
            fit_scope=self.fit_scope,
            seed=self.random_state,
            configuration=self.describe_configuration(),
            ranking=tuple(ranking),
            raw_scores=None if scores is None else dict(scores),
            natural_selected=None if natural is None else tuple(natural),
            training_row_count=int(len(X)),
            training_identity_sha256=training_identity_hash(X, target),
            fit_seconds=time.perf_counter() - started,
            warnings=tuple(self._collect_warnings(status)),
        )
        self.selected_features_ = list(selected)
        return self

    def _finalize_selection(
        self,
        *,
        ranking: Sequence[str],
        natural: Sequence[str] | None,
        universe_size: int,
    ) -> tuple[list[str], int | None, str]:
        return self._resolve_budget(ranking=ranking, universe_size=universe_size)

    def _effective_selection_mode(self, budget_status: str) -> str:
        if budget_status == "not_applicable":
            return "natural" if self.supports_natural_support else self.default_selection_mode
        return self.default_selection_mode

    def _collect_warnings(self, budget_status: str) -> list[str]:
        warnings: list[str] = []
        if budget_status == "clipped_to_universe":
            warnings.append(
                f"requested budget {self.k} exceeds the eligible candidate universe; "
                "returned every eligible feature and recorded the clip"
            )
        if budget_status == "infeasible_natural_support":
            warnings.append(
                f"requested budget {self.k} exceeds the algorithm's natural support; "
                "the budget was not padded"
            )
        return warnings

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return select_feature_frame(
            X,
            self.selected_features_,
            selector_name=self.__class__.__name__,
        )

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    @property
    def result(self) -> SelectionResult:
        if self.result_ is None:
            raise ValueError(f"{self.__class__.__name__} must be fitted first.")
        return self.result_


__all__ = [
    "BUDGET_STATUSES",
    "CONTRACT_VERSION",
    "ControlledSelectorFailure",
    "DEV_FOLD_TRAINING_ONLY",
    "LONG_FRAME_COLUMNS",
    "LightweightSelector",
    "SCORE_ORIENTATIONS",
    "SELECTION_MODES",
    "SelectionResult",
    "SelectorContractError",
    "TIE_RULE",
    "TIE_RULE_UNIVERSE_ORDER",
    "ordered_name_hash",
    "rank_by_score",
    "training_identity_hash",
]
