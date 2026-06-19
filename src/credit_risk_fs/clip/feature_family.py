from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping

import pandas as pd


AGGREGATE_SUFFIXES = ("MEAN", "MAX", "MIN", "SUM", "VAR", "AVG", "MEDI", "MODE")
FLAG_SUFFIXES = ("is_zero", "missing_flag", "flag", "ratio", "share")


@dataclass(frozen=True)
class FeatureFamilyResolution:
    canonical_feature_family: str
    family_resolution_source: str
    family_resolution_rule: str


def derive_feature_family(feature_name: str) -> str:
    """Backward-compatible conservative feature-name family heuristic."""
    name = str(feature_name)
    for token in [f"_{suffix}" for suffix in AGGREGATE_SUFFIXES]:
        if name.endswith(token):
            return name[: -len(token)]
    name = re.sub(r"(_is_zero|_missing_flag|_flag|_ratio|_share)$", "", name)
    parts = name.split("_")
    if len(parts) > 3:
        return "_".join(parts[:3])
    return name


def derive_canonical_feature_family(
    feature_name: str,
    semantic_group: str | None = None,
    source_formula: str | None = None,
    source_table: str | None = None,
    *,
    aliases: Mapping[str, str] | None = None,
) -> FeatureFamilyResolution:
    """Resolve an auditable, deterministic base family for feature-level splitting.

    The resolver is intentionally conservative: explicit aliases win, formula
    lineage is used only when it names one source variable, and name heuristics
    are limited to suffixes that encode common engineered variants.
    """
    name = _clean(feature_name)
    if not name:
        return FeatureFamilyResolution("", "feature_name_fallback", "empty_feature_name")

    alias_map = {_clean(key): _clean(value) for key, value in (aliases or {}).items()}
    if name in alias_map and alias_map[name]:
        return FeatureFamilyResolution(alias_map[name], "configured_alias", f"{name}->{alias_map[name]}")

    formula_family = _formula_family(source_formula)
    if formula_family:
        return FeatureFamilyResolution(formula_family, "formula_lineage", "single_source_variable")

    for suffix in ("_W_CITY", "_WITH_CITY"):
        if name.endswith(suffix):
            base = name[: -len(suffix)]
            if base:
                return FeatureFamilyResolution(base, "feature_name_heuristic", f"suffix:{suffix}")

    aggregate_family = _aggregate_suffix_family(name)
    if aggregate_family:
        return FeatureFamilyResolution(aggregate_family, "feature_name_heuristic", "aggregate_suffix")

    lowered = name.lower()
    for suffix in FLAG_SUFFIXES:
        token = f"_{suffix}"
        if lowered.endswith(token):
            return FeatureFamilyResolution(name[: -len(token)], "feature_name_heuristic", f"suffix:{token}")

    return FeatureFamilyResolution(name, "feature_name_fallback", "exact_feature_name")


def build_feature_family_audit(frame: pd.DataFrame, aliases: Mapping[str, str] | None = None) -> tuple[pd.DataFrame, dict[str, object]]:
    records = []
    for row in frame.to_dict("records"):
        feature = str(row.get("feature_name") or row.get("feature") or "")
        resolution = derive_canonical_feature_family(
            feature,
            semantic_group=_clean(row.get("semantic_group")),
            source_formula=_clean(row.get("source_formula") or row.get("source_formula_or_table")),
            source_table=_clean(row.get("source_table") or row.get("source_table_or_formula")),
            aliases=aliases,
        )
        records.append(
            {
                "dataset": str(row.get("dataset", "")),
                "feature_name": feature,
                "canonical_feature_family": resolution.canonical_feature_family,
                "family_resolution_source": resolution.family_resolution_source,
                "family_resolution_rule": resolution.family_resolution_rule,
                "semantic_group": _clean(row.get("semantic_group")),
                "source_table": _clean(row.get("source_table") or row.get("source_table_or_formula")),
            }
        )
    audit = pd.DataFrame(records).sort_values(["dataset", "feature_name"], kind="mergesort").reset_index(drop=True)
    if audit.empty:
        summary = {
            "canonical_family_count": 0,
            "multi_feature_family_count": 0,
            "largest_families": [],
            "ambiguous_family_assignments": [],
            "alias_rules": dict(aliases or {}),
            "alias_rule_usage_count": 0,
            "fallback_assignment_count": 0,
        }
        return audit, summary

    member_counts = audit.groupby("canonical_feature_family")["feature_name"].transform("size")
    audit["family_member_count"] = member_counts.astype(int)
    family_sizes = (
        audit.groupby("canonical_feature_family")["feature_name"]
        .size()
        .sort_values(ascending=False, kind="mergesort")
        .reset_index(name="member_count")
    )
    ambiguous = audit[
        audit["family_resolution_source"].eq("feature_name_heuristic")
        & audit["family_member_count"].eq(1)
    ]["feature_name"].astype(str).tolist()
    summary = {
        "canonical_family_count": int(audit["canonical_feature_family"].nunique()),
        "multi_feature_family_count": int((family_sizes["member_count"] > 1).sum()),
        "largest_families": family_sizes.head(20).to_dict("records"),
        "ambiguous_family_assignments": ambiguous,
        "alias_rules": dict(aliases or {}),
        "alias_rule_usage_count": int(audit["family_resolution_source"].eq("configured_alias").sum()),
        "fallback_assignment_count": int(audit["family_resolution_source"].eq("feature_name_fallback").sum()),
    }
    return audit, summary


def _aggregate_suffix_family(name: str) -> str:
    parts = name.split("_")
    if len(parts) >= 3 and parts[-1] in AGGREGATE_SUFFIXES:
        return "_".join(parts[:-1])
    return ""


def _formula_family(source_formula: str | None) -> str:
    text = _clean(source_formula)
    if not text or text.lower() == "nan":
        return ""
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", text):
        return text
    match = re.fullmatch(r"(?:mean|max|min|sum|avg|median|mode|var)\(([A-Za-z_][A-Za-z0-9_]*)\)", text, flags=re.IGNORECASE)
    return match.group(1) if match else ""


def _clean(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()
