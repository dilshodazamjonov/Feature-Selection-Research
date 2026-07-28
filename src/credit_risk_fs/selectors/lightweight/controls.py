"""Scientific controls: deterministic random-k and the full-candidate baseline.

Neither control is a competitive method. ``random_k`` establishes what a feature
budget buys by chance; ``full_features`` establishes what selection costs
relative to using every eligible candidate. Both are unsupervised by
construction -- they never receive the target, which is what makes them valid
reference points rather than weak selectors.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from credit_risk_fs.selectors.lightweight.contract import (
    TIE_RULE_UNIVERSE_ORDER,
    LightweightSelector,
)


class RandomKSelector(LightweightSelector):
    """Draw a reproducible random subset of the eligible candidate universe.

    Randomness comes from a local :class:`numpy.random.Generator` seeded by
    ``random_state``; global NumPy RNG state is never read or written, so an
    unrelated caller cannot perturb the draw. The published ranking is the full
    random priority order, which is what makes the selected subset independently
    reproducible from the artifact alone.
    """

    method_id: ClassVar[str] = "random_k"
    display_label: ClassVar[str] = "Random-k control"
    implementation_id: ClassVar[str] = "random_k_local_generator_v1"
    supervised: ClassVar[bool] = False
    supports_ranking: ClassVar[bool] = True
    supports_natural_support: ClassVar[bool] = False
    supports_fixed_budget: ClassVar[bool] = True
    default_selection_mode: ClassVar[str] = "random_control"
    score_orientation: ClassVar[str] = "rank_1_is_best"

    def describe_configuration(self) -> dict[str, Any]:
        configuration = super().describe_configuration()
        configuration.update(
            {
                "generator": "numpy.random.default_rng",
                "seed_source": "random_state",
                "uses_global_rng_state": False,
                "inspects_target": False,
                "priority_order": "permutation_of_authenticated_candidate_order",
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
        generator = np.random.default_rng(self.random_state)
        positions = generator.permutation(len(candidate_order))
        ranking = [str(candidate_order[index]) for index in positions]
        # The score is the reciprocal priority, recorded only so the artifact can
        # be re-sorted without consulting the rank column.
        scores = {name: float(len(ranking) - order) for order, name in enumerate(ranking)}
        return ranking, scores, None


class FullCandidateFeaturesSelector(LightweightSelector):
    """Explicit no-selection control over the eligible candidate universe.

    "Full features" means every candidate that survived the frozen leakage and
    metadata exclusions -- not every raw dataset column. The result is
    deliberately identical to the historical ``none`` route, which continues to
    resolve to no selector object at all; this class exists so the control can be
    named, ranked, hashed, and audited like any other method.
    """

    method_id: ClassVar[str] = "full_features"
    display_label: ClassVar[str] = "Full candidate features"
    implementation_id: ClassVar[str] = "full_candidate_features_v1"
    supervised: ClassVar[bool] = False
    supports_ranking: ClassVar[bool] = True
    supports_natural_support: ClassVar[bool] = False
    supports_fixed_budget: ClassVar[bool] = False
    default_selection_mode: ClassVar[str] = "full_control"
    score_orientation: ClassVar[str] = "not_applicable"
    tie_rule: ClassVar[str] = TIE_RULE_UNIVERSE_ORDER

    def __init__(
        self,
        *,
        k: int | None = None,
        random_state: int = 42,
        excluded_columns: Sequence[str] | None = None,
        fit_scope: str = "dev_fold_training_only",
    ) -> None:
        # A fixed-k request is ignored rather than honoured, and the fact that it
        # was ignored is recorded instead of being silently dropped.
        super().__init__(
            k=None,
            random_state=random_state,
            excluded_columns=excluded_columns,
            fit_scope=fit_scope,
        )
        self.ignored_budget_request = None if k is None else int(k)

    def describe_configuration(self) -> dict[str, Any]:
        configuration = super().describe_configuration()
        configuration.update(
            {
                "selection": "none",
                "candidate_scope": "eligible_candidates_after_frozen_exclusions",
                "inspects_target": False,
                "ignored_budget_request": self.ignored_budget_request,
                "equivalent_legacy_registry_id": "none",
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
        return [str(name) for name in candidate_order], None, None

    def _collect_warnings(self, budget_status: str) -> list[str]:
        collected = super()._collect_warnings(budget_status)
        if self.ignored_budget_request is not None:
            collected.append(
                f"full_features ignores the fixed budget {self.ignored_budget_request} "
                "by design; every eligible candidate feature is retained"
            )
        return collected


__all__ = ["FullCandidateFeaturesSelector", "RandomKSelector"]
