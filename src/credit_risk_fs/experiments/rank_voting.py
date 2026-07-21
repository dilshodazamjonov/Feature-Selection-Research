"""The explicit prospective cross-dataset rank-voting v1 aggregation rule."""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Iterable, Mapping

import pandas as pd


PROTOCOL_NAME = "cross_dataset_rank_voting_v1"
ELIGIBLE_VOTERS = ("rf_corr_mrmr", "boruta")
VOTER_ALIASES = {
    "rf_corr_mrmr": "rf_corr_mrmr",
    "randomforestrelevancemrmrselector": "rf_corr_mrmr",
    "boruta": "boruta",
    "borutaselector": "boruta",
}
REQUIRED_FIT_SCOPE = "dev_fold_training_only"
DEFAULT_FORBIDDEN_FEATURES = {
    "target",
    "sk_id_curr",
    "sk_id_bureau",
    "sk_id_prev",
    "loan_id",
    "id",
    "member_id",
    "recent_decision",
    "issue_d",
    "loan_status",
}


def _canonical_name(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("feature and voter names must be strings")
    name = unicodedata.normalize("NFC", value.strip())
    if not name:
        raise ValueError("feature and voter names must not be empty")
    return name.casefold()


def _canonical_voter(value: str) -> str:
    compact = _canonical_name(value).replace("_", "").replace("-", "")
    aliases = {
        alias.replace("_", "").replace("-", ""): canonical
        for alias, canonical in VOTER_ALIASES.items()
    }
    if compact not in aliases:
        raise ValueError(f"unknown voter for {PROTOCOL_NAME}: {value!r}")
    return aliases[compact]


def aggregate_cross_dataset_rank_voting(
    *,
    eligible_features: Iterable[str],
    rankings: Mapping[str, Iterable[str]],
    fit_scopes: Mapping[str, str],
    candidate_cap: int | None = None,
    forbidden_features: Iterable[str] = DEFAULT_FORBIDDEN_FEATURES,
) -> pd.DataFrame:
    """Aggregate the two registered fold-local rankings under the frozen rule."""

    universe = list(eligible_features)
    canonical_features = [_canonical_name(feature) for feature in universe]
    if len(set(canonical_features)) != len(canonical_features):
        raise ValueError("eligible feature universe contains duplicates or canonical-name collisions")
    forbidden = {_canonical_name(feature) for feature in forbidden_features}
    contamination = sorted(set(canonical_features) & forbidden)
    if contamination:
        raise ValueError(f"eligible feature universe contains leakage/identity fields: {contamination}")
    if not universe:
        raise ValueError("eligible feature universe must not be empty")
    feature_by_canonical = dict(zip(canonical_features, universe, strict=True))

    canonical_rankings: dict[str, list[str]] = {}
    for supplied_voter, supplied_features in rankings.items():
        voter = _canonical_voter(supplied_voter)
        if voter in canonical_rankings:
            raise ValueError(f"voter alias duplicates an existing vote: {voter}")
        ranking = list(supplied_features)
        normalized = [_canonical_name(feature) for feature in ranking]
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"{voter} ranking contains duplicate features")
        unknown = sorted(set(normalized) - set(canonical_features))
        if unknown:
            raise ValueError(f"{voter} ranking contains unknown features: {unknown}")
        if set(normalized) & forbidden:
            raise ValueError(f"{voter} ranking contains leakage/identity fields")
        canonical_rankings[voter] = normalized
    if set(canonical_rankings) != set(ELIGIBLE_VOTERS):
        raise ValueError(
            f"{PROTOCOL_NAME} requires exactly {list(ELIGIBLE_VOTERS)}; "
            f"received {sorted(canonical_rankings)}"
        )

    canonical_scopes: dict[str, str] = {}
    for supplied_voter, scope in fit_scopes.items():
        voter = _canonical_voter(supplied_voter)
        if voter in canonical_scopes:
            raise ValueError(f"voter alias duplicates a fitting boundary: {voter}")
        canonical_scopes[voter] = str(scope)
    if set(canonical_scopes) != set(ELIGIBLE_VOTERS) or any(
        scope != REQUIRED_FIT_SCOPE for scope in canonical_scopes.values()
    ):
        raise ValueError(
            f"every voter must use fitting boundary {REQUIRED_FIT_SCOPE!r}"
        )

    universe_size = len(universe)
    if candidate_cap is not None and (
        isinstance(candidate_cap, bool)
        or not isinstance(candidate_cap, int)
        or candidate_cap < 1
        or candidate_cap > universe_size
    ):
        raise ValueError("candidate_cap must be an integer within the eligible universe")

    rank_lookup = {
        voter: {feature: rank for rank, feature in enumerate(ranking, start=1)}
        for voter, ranking in canonical_rankings.items()
    }
    rows = []
    denominator = max(universe_size - 1, 1)
    for canonical_feature in canonical_features:
        row: dict[str, object] = {
            "feature": feature_by_canonical[canonical_feature],
            "normalized_feature_name": canonical_feature,
        }
        scores: list[float] = []
        ranks: list[int] = []
        presence_count = 0
        for voter in ELIGIBLE_VOTERS:
            rank = rank_lookup[voter].get(canonical_feature)
            present = rank is not None
            score = 1.0 - (rank - 1) / denominator if present else 0.0
            if not math.isfinite(score):
                raise ValueError(f"{voter} produced a non-finite normalized score")
            row[f"{voter}_raw_rank"] = rank
            row[f"{voter}_normalized_score"] = score
            row[f"{voter}_present"] = present
            scores.append(score)
            if present:
                presence_count += 1
                ranks.append(int(rank))
        row["aggregate_score"] = sum(scores) / len(ELIGIBLE_VOTERS)
        row["voter_presence_count"] = presence_count
        row["best_individual_rank"] = min(ranks) if ranks else universe_size + 1
        rows.append(row)

    result = pd.DataFrame(rows).sort_values(
        [
            "aggregate_score",
            "voter_presence_count",
            "best_individual_rank",
            "normalized_feature_name",
        ],
        ascending=[False, False, True, True],
        kind="mergesort",
    )
    result = result.reset_index(drop=True)
    result.insert(0, "aggregate_rank", range(1, len(result) + 1))
    return result.head(candidate_cap).copy() if candidate_cap is not None else result


__all__ = [
    "ELIGIBLE_VOTERS",
    "PROTOCOL_NAME",
    "REQUIRED_FIT_SCOPE",
    "aggregate_cross_dataset_rank_voting",
]
