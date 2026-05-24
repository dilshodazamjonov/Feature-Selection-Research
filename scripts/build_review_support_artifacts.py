from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in [PROJECT_ROOT, SRC_ROOT]:
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from credit_risk_fs.preprocessing.lendingclub import (  # noqa: E402
    LENDINGCLUB_DIRECT_LEAKAGE_COLUMNS,
    LENDINGCLUB_IDENTIFIER_OR_TEXT_COLUMNS,
    LENDINGCLUB_POLICY_OR_POST_APPROVAL_COLUMNS,
    LENDINGCLUB_POST_OUTCOME_LEAKAGE_COLUMNS,
    LENDINGCLUB_TEXT_OR_LOW_SIGNAL_COLUMNS,
    LENDINGCLUB_UNDERWRITING_POLICY_COLUMNS,
    lendingclub_model_blacklist,
)


def _frame_to_text(df: pd.DataFrame) -> str:
    return df.to_string(index=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build lightweight review-support artifacts for final research reporting.",
    )
    parser.add_argument(
        "--dataset",
        choices=["homecredit", "lendingclub", "all"],
        default="all",
        help="Dataset to process. Default builds both artifact sets.",
    )
    return parser


def _write_homecredit_temporal_review() -> list[Path]:
    metadata_dir = PROJECT_ROOT / "data" / "homecredit" / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "table_name": "application_train",
            "temporal_field": "row anchor",
            "used_for": "application-level modeling row",
            "current_repo_behavior": "serves as the application-level anchor after feature merges",
            "automated_verification_status": "anchored_by_primary_application_row",
            "notes": "No auxiliary as-of assumption needed for the application row itself.",
        },
        {
            "table_name": "previous_application",
            "temporal_field": "DAYS_DECISION",
            "used_for": "direct recency signal and current split proxy fallback",
            "current_repo_behavior": "feeds PREV_recent_decision_MAX and recent_decision",
            "automated_verification_status": "manual_review_required",
            "notes": "Relative-day semantics look historical, but file-only checks cannot prove strict as-of alignment.",
        },
        {
            "table_name": "bureau",
            "temporal_field": "DAYS_CREDIT",
            "used_for": "application-time proxy support and bureau-history features",
            "current_repo_behavior": "used in application_time_proxy and bureau aggregates",
            "automated_verification_status": "manual_review_required",
            "notes": "Should represent historical bureau exposures, but semantic confirmation remains manual.",
        },
        {
            "table_name": "installments_payments",
            "temporal_field": "DAYS_INSTALMENT / DAYS_ENTRY_PAYMENT",
            "used_for": "application-time proxy support and installment behavior features",
            "current_repo_behavior": "used in application_time_proxy and repayment aggregates",
            "automated_verification_status": "manual_review_required",
            "notes": "Contains historical payment-event timing, but as-of treatment must be defended manually.",
        },
        {
            "table_name": "POS_CASH_balance",
            "temporal_field": "MONTHS_BALANCE",
            "used_for": "application-time proxy support and POS behavior features",
            "current_repo_behavior": "MONTHS_BALANCE is converted to relative days for proxy support",
            "automated_verification_status": "manual_review_required",
            "notes": "Relative-month balance history is plausible for historical use, not auto-verifiable from files alone.",
        },
        {
            "table_name": "credit_card_balance",
            "temporal_field": "MONTHS_BALANCE",
            "used_for": "application-time proxy support and credit-card utilization features",
            "current_repo_behavior": "MONTHS_BALANCE is converted to relative days for proxy support",
            "automated_verification_status": "manual_review_required",
            "notes": "Historical balance interpretation should be confirmed manually for write-up rigor.",
        },
        {
            "table_name": "derived_proxy",
            "temporal_field": "application_time_proxy / recent_decision",
            "used_for": "DEV/OOT split construction",
            "current_repo_behavior": "takes the most recent event across historical sources",
            "automated_verification_status": "depends_on_source_table_review",
            "notes": "The proxy is only as defensible as the source-table as-of semantics above.",
        },
    ]
    review_df = pd.DataFrame(rows)
    csv_path = metadata_dir / "temporal_asof_review.csv"
    md_path = metadata_dir / "temporal_asof_review.md"
    review_df.to_csv(csv_path, index=False)

    manual_count = int((review_df["automated_verification_status"] == "manual_review_required").sum())
    md_lines = [
        "# Home Credit Temporal As-Of Review",
        "",
        "This artifact is intentionally conservative.",
        "",
        "It does not claim that auxiliary Home Credit tables are fully verified as-of the application date.",
        "Instead, it records which tables contribute temporal signals or proxy support and marks the current manual-review burden explicitly.",
        "",
        f"- Rows requiring explicit manual confirmation: `{manual_count}`",
        "- Automated verification is limited to repository structure and column usage.",
        "- Source-table as-of semantics still require manual review for professor-facing reporting.",
        "",
        "## Review Table",
        "",
        _frame_to_text(review_df),
        "",
        "## Practical Conclusion",
        "",
        "Use the current Home Credit temporal split as a reasonable research proxy, but keep the auxiliary-table caveat in the final report until source semantics are manually confirmed.",
    ]
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return [csv_path, md_path]


def _load_raw_schema_columns(path: Path) -> set[str]:
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        files = payload.get("files", [])
        if files:
            return set(files[0].get("columns", []))
    raw_dir = PROJECT_ROOT / "data" / "lendingclub" / "raw"
    candidates = sorted(raw_dir.glob("*.csv"))
    if not candidates:
        return set()
    return set(pd.read_csv(candidates[0], nrows=0).columns.tolist())


def _write_lendingclub_leakage_review() -> list[Path]:
    metadata_dir = PROJECT_ROOT / "data" / "lendingclub" / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    schema_path = metadata_dir / "raw_schema_snapshot.json"
    raw_columns = _load_raw_schema_columns(schema_path)
    strict_blacklist = set(lendingclub_model_blacklist())

    category_map = {
        "direct_target_leakage": set(LENDINGCLUB_DIRECT_LEAKAGE_COLUMNS),
        "post_outcome_repayment": set(LENDINGCLUB_POST_OUTCOME_LEAKAGE_COLUMNS),
        "post_approval_operational": set(LENDINGCLUB_POLICY_OR_POST_APPROVAL_COLUMNS),
        "underwriting_policy": set(LENDINGCLUB_UNDERWRITING_POLICY_COLUMNS),
        "identifier_or_text": set(LENDINGCLUB_IDENTIFIER_OR_TEXT_COLUMNS) | set(LENDINGCLUB_TEXT_OR_LOW_SIGNAL_COLUMNS),
    }

    rows: list[dict[str, object]] = []
    for category, columns in category_map.items():
        for column in sorted(columns):
            rows.append(
                {
                    "column_name": column,
                    "category": category,
                    "present_in_raw_schema": column in raw_columns,
                    "covered_by_strict_blacklist": column in strict_blacklist,
                    "guardrail_status": (
                        "covered"
                        if column not in raw_columns or column in strict_blacklist
                        else "missing_from_blacklist"
                    ),
                }
            )

    review_df = pd.DataFrame(rows).sort_values(["category", "column_name"]).reset_index(drop=True)
    csv_path = metadata_dir / "raw_leakage_blacklist_review.csv"
    md_path = metadata_dir / "raw_leakage_blacklist_review.md"
    review_df.to_csv(csv_path, index=False)

    missing_df = review_df[
        review_df["present_in_raw_schema"].astype(bool) & ~review_df["covered_by_strict_blacklist"].astype(bool)
    ]
    if not missing_df.empty:
        raise ValueError(
            "LendingClub raw leakage review found raw columns missing from the strict blacklist: "
            + ", ".join(sorted(missing_df["column_name"].astype(str).unique().tolist()))
        )

    present_and_covered = review_df[
        review_df["present_in_raw_schema"].astype(bool) & review_df["covered_by_strict_blacklist"].astype(bool)
    ]
    md_lines = [
        "# LendingClub Raw Leakage Blacklist Review",
        "",
        "This artifact checks that known risky raw LendingClub columns remain covered by the strict preprocessing blacklist.",
        "",
        f"- Raw schema columns reviewed: `{len(raw_columns):,}`",
        f"- Known risky raw columns present and covered: `{len(present_and_covered):,}`",
        "- Result: `strict blacklist coverage confirmed for the known raw-risk set`",
        "",
        "## Review Table",
        "",
        _frame_to_text(review_df),
        "",
        "## Practical Conclusion",
        "",
        "Raw direct use should still be treated as blocked or tightly audited, but the current centralized blacklist covers the known high-risk raw fields used by repository preparation and review tooling.",
    ]
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return [csv_path, md_path]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    written: list[Path] = []

    if args.dataset in {"homecredit", "all"}:
        written.extend(_write_homecredit_temporal_review())
    if args.dataset in {"lendingclub", "all"}:
        written.extend(_write_lendingclub_leakage_review())

    for path in written:
        print(path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
