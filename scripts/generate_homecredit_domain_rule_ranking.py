"""Generate a deterministic Home Credit ranking from explicit domain rules.

This module makes no LLM or network call. It reads feature metadata only and
applies the exact, family, aggregation, and fallback rules of
``credit_risk_fs.selectors.domain_rules`` (the same rules the domain-rules
selector uses).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from credit_risk_fs.selectors.domain_rules import (  # noqa: E402
    AGG_SUFFIXES,
    EXACT_RULES,
    FALLBACKS,
    PROPERTY_TERMS,
    agg_adjustment,
    agg_priority,
    classify,
    leakage_status,
    rule,
    sort_bonus,
    strip_agg,
)

INPUT_PATH = Path(
    "results/corrected_homecredit_clip/feature_universe/"
    "feature_universe_reconciliation.csv"
)
OUTPUT_DIR = Path("results/domain_rule_rankings")
OUTPUT_PATH = OUTPUT_DIR / "homecredit_full_529_domain_rule_ranking.csv"
SUMMARY_PATH = OUTPUT_DIR / "homecredit_full_529_domain_rule_ranking_validation.txt"

REQUIRED_INPUT_COLUMNS = [
    "feature_id",
    "feature_name",
    "source_table",
    "semantic_group",
]
OUTPUT_COLUMNS = [
    "feature_id",
    "feature_name",
    "source_table",
    "semantic_group",
    "feature_family",
    "credit_relevance_score",
    "risk_concept",
    "reason_short",
    "rank",
    "normalized_rank",
    "leakage_status",
]

FORBIDDEN_COLUMN_MARKERS = (
    "target",
    "auc",
    "ks",
    "psi",
    "gini",
    "iv",
    "mrmr",
    "selected",
    "fold",
    "performance",
    "bad_rate",
    "label",
)


def normalized_source(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def validate_input(frame: pd.DataFrame) -> list[str]:
    errors = []
    missing = [column for column in REQUIRED_INPUT_COLUMNS if column not in frame]
    if missing:
        errors.append(f"missing required input columns: {missing}")
    if len(frame) != 529:
        errors.append(f"input feature count is {len(frame)}, expected exactly 529")
    if "feature_id" in frame and frame["feature_id"].nunique(dropna=False) != 529:
        errors.append("feature_id values are not unique")
    if "feature_name" in frame and frame["feature_name"].nunique(dropna=False) != 529:
        errors.append("feature_name values are not unique")
    return errors


def validate_output(frame: pd.DataFrame) -> list[str]:
    errors = []
    if len(frame) != 529:
        errors.append(f"output rows={len(frame)}")
    if frame["feature_id"].nunique(dropna=False) != 529:
        errors.append("duplicate feature_id in output")
    if frame["feature_name"].nunique(dropna=False) != 529:
        errors.append("duplicate feature_name in output")
    if frame["rank"].tolist() != list(range(1, 530)):
        errors.append("rank values are not exactly 1..529")
    if frame["rank"].duplicated().any():
        errors.append("duplicate ranks")
    expected_leakage = {
        "LEAKAGE_SAFE_LABEL_FREE",
        "POTENTIAL_LEAKAGE_NAME_REVIEW_NEEDED",
    }
    if set(frame["leakage_status"]) - expected_leakage:
        errors.append("invalid leakage_status values")
    if (frame["leakage_status"].astype(str).str.len() == 0).any():
        errors.append("blank leakage_status values")
    for rank, normalized_rank in zip(frame["rank"], frame["normalized_rank"]):
        expected = f"{(int(rank) - 1) / 528:.9f}"
        if normalized_rank != expected:
            errors.append("normalized_rank mismatch")
            break
    return errors


def build_ranking(input_frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in input_frame.itertuples(index=False):
        name = str(getattr(row, "feature_name"))
        source = normalized_source(getattr(row, "source_table"))
        group = str(getattr(row, "semantic_group"))
        _, agg = strip_agg(name)
        base_score, family, concept, reason = classify(name, group)
        adjusted_score = base_score + agg_adjustment(family, agg)
        score = max(1, min(100, int(round(adjusted_score))))
        status = leakage_status(name)
        rows.append(
            {
                "feature_id": str(getattr(row, "feature_id")),
                "feature_name": name,
                "source_table": source,
                "semantic_group": group,
                "feature_family": family,
                "credit_relevance_score": score,
                "risk_concept": concept,
                "reason_short": reason,
                "_flagged": status != "LEAKAGE_SAFE_LABEL_FREE",
                "_fine_score": adjusted_score + sort_bonus(name, family, concept, agg),
                "leakage_status": status,
            }
        )
    ranked = pd.DataFrame(rows).sort_values(
        by=["_flagged", "_fine_score", "credit_relevance_score", "feature_name"],
        ascending=[True, False, False, True],
        kind="mergesort",
    )
    ranked = ranked.reset_index(drop=True)
    ranked["rank"] = range(1, len(ranked) + 1)
    ranked["normalized_rank"] = ranked["rank"].map(
        lambda rank: f"{(rank - 1) / 528:.9f}"
    )
    return ranked[OUTPUT_COLUMNS]


def main() -> int:
    input_frame = pd.read_csv(INPUT_PATH, dtype=str).fillna("")
    input_errors = validate_input(input_frame)
    if input_errors:
        raise SystemExit("validation failed: " + "; ".join(input_errors))

    ignored_columns = [
        column
        for column in input_frame.columns
        if any(marker in column.lower() for marker in FORBIDDEN_COLUMN_MARKERS)
    ]
    output_frame = build_ranking(input_frame)
    output_errors = validate_output(output_frame)
    if output_errors:
        raise SystemExit("validation failed: " + "; ".join(output_errors))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_frame.to_csv(OUTPUT_PATH, index=False)
    leakage_count = int(
        (
            output_frame["leakage_status"]
            == "POTENTIAL_LEAKAGE_NAME_REVIEW_NEEDED"
        ).sum()
    )
    summary = [
        "ranking_method: deterministic_domain_rules",
        "uses_llm: false",
        f"input_feature_count: {len(input_frame)}",
        f"ranked_feature_count: {len(output_frame)}",
        f"duplicate_feature_ids: {int(input_frame['feature_id'].duplicated().sum())}",
        f"duplicate_ranks: {int(output_frame['rank'].duplicated().sum())}",
        f"leakage_flagged_features: {leakage_count}",
        "target_performance_columns_ignored: "
        + (
            "yes (" + ", ".join(ignored_columns) + ")"
            if ignored_columns
            else "no forbidden target/performance columns present"
        ),
        f"output_csv: {OUTPUT_PATH.as_posix()}",
    ]
    SUMMARY_PATH.write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))
    return 0


__all__ = [
    "AGG_SUFFIXES",
    "EXACT_RULES",
    "FALLBACKS",
    "PROPERTY_TERMS",
    "agg_adjustment",
    "agg_priority",
    "build_ranking",
    "classify",
    "leakage_status",
    "rule",
    "sort_bonus",
    "strip_agg",
]


if __name__ == "__main__":
    raise SystemExit(main())
