from __future__ import annotations

import pandas as pd


def jaccard(left: set[str] | list[str], right: set[str] | list[str]) -> float:
    a = set(left)
    b = set(right)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def semantic_pool_diagnostics(pool: pd.DataFrame, *, feature_column: str = "feature_name", group_column: str = "semantic_group") -> dict[str, object]:
    if feature_column not in pool.columns:
        raise ValueError(f"missing feature column: {feature_column}")
    groups = pool[group_column].fillna("missing") if group_column in pool.columns else pd.Series(["missing"] * len(pool))
    shares = groups.value_counts(normalize=True)
    families = pool[feature_column].astype(str).map(_base_family)
    family_counts = families.value_counts()
    repeated = int((family_counts > 1).sum())
    return {
        "pool_size": int(len(pool)),
        "semantic_group_count": int(groups.nunique()),
        "largest_group_share": float(shares.max()) if len(shares) else 0.0,
        "repeated_family_share": float(repeated / max(1, len(family_counts))),
        "feature_family_coverage": int(family_counts.size),
    }


def overlap_summary(candidate_pool: list[str], clip_pool: list[str], full_mrmr_selected: list[str]) -> dict[str, float]:
    return {
        "jaccard_with_clip_v2_pool": jaccard(candidate_pool, clip_pool),
        "overlap_with_full_mrmr_selected": jaccard(candidate_pool, full_mrmr_selected),
    }


def _base_family(feature: str) -> str:
    parts = feature.lower().split("_")
    suffixes = {"mean", "max", "min", "var", "sum", "std", "median", "flag", "rate", "ratio"}
    while parts and parts[-1] in suffixes:
        parts.pop()
    return "_".join(parts) if parts else feature.lower()

