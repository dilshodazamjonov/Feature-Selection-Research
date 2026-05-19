from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in [PROJECT_ROOT, SRC_ROOT]:
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from credit_risk_fs.preprocessing.labeling import (  # noqa: E402
    LENDINGCLUB_AMBIGUOUS_OR_ONGOING_STATUSES,
    LENDINGCLUB_FINAL_BAD_STATUSES,
    LENDINGCLUB_FINAL_GOOD_STATUSES,
    build_lendingclub_target,
)
from credit_risk_fs.preprocessing.lendingclub import (  # noqa: E402
    LENDINGCLUB_DIRECT_LEAKAGE_COLUMNS,
    LENDINGCLUB_POLICY_OR_POST_APPROVAL_COLUMNS,
    LENDINGCLUB_POST_OUTCOME_LEAKAGE_COLUMNS,
    LENDINGCLUB_TEXT_OR_LOW_SIGNAL_COLUMNS,
    prepare_lendingclub_application_frame,
)

POST_OUTCOME_COLUMNS = [
    "out_prncp",
    "out_prncp_inv",
    "total_pymnt",
    "total_pymnt_inv",
    "total_rec_prncp",
    "total_rec_int",
    "total_rec_late_fee",
    "recoveries",
    "collection_recovery_fee",
    "last_pymnt_d",
    "last_pymnt_amnt",
    "next_pymnt_d",
    "last_credit_pull_d",
    "last_fico_range_low",
    "last_fico_range_high",
]
POLICY_LEAKAGE_COLUMNS = [
    "grade",
    "sub_grade",
    "int_rate",
    "installment",
    "funded_amnt",
    "funded_amnt_inv",
]
IDENTIFIER_OR_TEXT_COLUMNS = ["id", "member_id", "url", "desc", "emp_title", "zip_code"]
OUTCOME_DATE_COLUMNS = [
    "last_pymnt_d",
    "next_pymnt_d",
    "last_credit_pull_d",
    "debt_settlement_flag_date",
    "settlement_date",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare LendingClub into the research single-table format.")
    parser.add_argument("--raw-file", default=None, help="Path to the raw accepted-loans CSV.")
    parser.add_argument("--output-dir", default="data/lendingclub/processed")
    parser.add_argument("--metadata-dir", default="data/lendingclub/metadata")
    return parser


def _resolve_raw_file(raw_file: str | None) -> Path:
    if raw_file:
        return Path(raw_file)
    candidates = sorted((PROJECT_ROOT / "data" / "lendingclub" / "raw").glob("*.csv"))
    if not candidates:
        raise FileNotFoundError("No LendingClub raw CSV found under data/lendingclub/raw.")
    return candidates[0]


def _write_schema_snapshot(raw_file: Path, metadata_dir: Path) -> None:
    try:
        with raw_file.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, [])
    except UnicodeDecodeError:
        with raw_file.open("r", encoding="latin1", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, [])
    payload = {"dataset": "lendingclub", "files": [{"name": raw_file.name, "columns": header}]}
    (metadata_dir / "raw_schema_snapshot.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _parse_monthly_dates(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, format="%b-%Y", errors="coerce")
    if parsed.notna().any():
        return parsed
    return pd.to_datetime(series, errors="coerce")


def _extract_term_months(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype("string").str.extract(r"(\d+)")[0], errors="coerce")


def _resolve_observation_end_date(df: pd.DataFrame) -> pd.Timestamp | None:
    observed_maxima: list[pd.Timestamp] = []
    for column in OUTCOME_DATE_COLUMNS:
        if column not in df.columns:
            continue
        parsed = _parse_monthly_dates(df[column])
        max_value = parsed.max()
        if pd.notna(max_value):
            observed_maxima.append(max_value)
    if not observed_maxima:
        return None
    return max(observed_maxima)


def _build_audit_frame(df: pd.DataFrame) -> pd.DataFrame:
    audit = df.copy()
    audit["issue_d"] = _parse_monthly_dates(audit["issue_d"])
    audit["term_months"] = _extract_term_months(audit["term"])
    final_statuses = LENDINGCLUB_FINAL_BAD_STATUSES | LENDINGCLUB_FINAL_GOOD_STATUSES
    audit = audit[audit["loan_status"].isin(final_statuses)].copy()
    audit["TARGET"] = build_lendingclub_target(audit)
    return audit


def _write_target_definition(
    metadata_dir: Path,
    *,
    raw_df: pd.DataFrame,
    audit_df: pd.DataFrame,
    observation_end_date: pd.Timestamp | None,
) -> Path:
    raw_count = int(len(raw_df))
    final_count = int(len(audit_df))
    excluded_count = raw_count - final_count
    term_summary = (
        audit_df.groupby("term_months")["TARGET"]
        .agg(row_count="size", bad_rate="mean")
        .reset_index()
        .sort_values("term_months")
    )
    term_lines = [
        f"- `{int(row.term_months)}` months: `{int(row.row_count):,}` rows, bad rate `{row.bad_rate:.2%}`"
        for row in term_summary.itertuples(index=False)
        if pd.notna(row.term_months)
    ]
    excluded_present = sorted(set(raw_df["loan_status"].dropna().astype(str)) & LENDINGCLUB_AMBIGUOUS_OR_ONGOING_STATUSES)
    observation_text = observation_end_date.strftime("%Y-%m-%d") if observation_end_date is not None else "unknown"
    md = "\n".join(
        [
            "# LendingClub Target Definition",
            "",
            "## Final Label Rule",
            "",
            "Only final resolved statuses are used for `TARGET`.",
            "",
            "Good statuses:",
            *[f"- `{status}`" for status in sorted(LENDINGCLUB_FINAL_GOOD_STATUSES)],
            "",
            "Bad statuses:",
            *[f"- `{status}`" for status in sorted(LENDINGCLUB_FINAL_BAD_STATUSES)],
            "",
            "Excluded ongoing or ambiguous statuses:",
            *[f"- `{status}`" for status in excluded_present],
            "",
            "## Counts",
            "",
            f"- Raw rows read: `{raw_count:,}`",
            f"- Rows retained after final-status filter: `{final_count:,}`",
            f"- Rows removed as ongoing or ambiguous: `{excluded_count:,}`",
            "",
            "## Term Censoring Handling",
            "",
            f"- Estimated observation end date from outcome-related columns: `{observation_text}`",
            "- The label set keeps only resolved final statuses, not `Current`, `Issued`, `In Grace Period`, or `Late` statuses.",
            "- 36-month and 60-month term distributions are audited separately in `issue_date_target_distribution.csv`.",
            "- Main experiments use the 2014-2016 window to avoid relying on late-vintage LendingClub outcomes as the primary evaluation regime.",
            "",
            "Retained rows by term:",
            *term_lines,
            "",
            "## Saved Audit Files",
            "",
            "- `target_definition.md`",
            "- `leakage_columns.yaml`",
            "- `label_distribution.csv`",
            "- `issue_date_target_distribution.csv`",
        ]
    )
    output_path = metadata_dir / "target_definition.md"
    output_path.write_text(md, encoding="utf-8")
    return output_path


def _write_label_distribution(metadata_dir: Path, audit_df: pd.DataFrame) -> Path:
    label_distribution = (
        audit_df.assign(target_label=audit_df["TARGET"].map({0: "good", 1: "bad"}))
        .groupby(["TARGET", "target_label"])
        .size()
        .reset_index(name="row_count")
        .sort_values("TARGET")
    )
    label_distribution["row_share"] = label_distribution["row_count"] / label_distribution["row_count"].sum()
    output_path = metadata_dir / "label_distribution.csv"
    label_distribution.to_csv(output_path, index=False)
    return output_path


def _write_issue_date_target_distribution(metadata_dir: Path, audit_df: pd.DataFrame) -> Path:
    monthly = (
        audit_df.groupby([pd.Grouper(key="issue_d", freq="MS"), "term_months"])["TARGET"]
        .agg(observation_count="size", bad_count="sum", bad_rate="mean")
        .reset_index()
        .sort_values(["issue_d", "term_months"])
    )
    monthly["good_count"] = monthly["observation_count"] - monthly["bad_count"]
    monthly["issue_month"] = monthly["issue_d"].dt.strftime("%Y-%m")
    monthly = monthly[
        [
            "issue_d",
            "issue_month",
            "term_months",
            "observation_count",
            "good_count",
            "bad_count",
            "bad_rate",
        ]
    ]
    output_path = metadata_dir / "issue_date_target_distribution.csv"
    monthly.to_csv(output_path, index=False)
    return output_path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    raw_file = _resolve_raw_file(args.raw_file)
    output_dir = PROJECT_ROOT / args.output_dir
    metadata_dir = PROJECT_ROOT / args.metadata_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(raw_file, low_memory=False)
    audit_df = _build_audit_frame(raw_df)
    observation_end_date = _resolve_observation_end_date(raw_df)
    target_definition_path = _write_target_definition(
        metadata_dir,
        raw_df=raw_df,
        audit_df=audit_df,
        observation_end_date=observation_end_date,
    )
    label_distribution_path = _write_label_distribution(metadata_dir, audit_df)
    issue_date_distribution_path = _write_issue_date_target_distribution(metadata_dir, audit_df)

    df = raw_df.copy()
    hardship_cols = [column for column in df.columns if column.startswith("hardship_")]
    settlement_cols = [column for column in df.columns if column.startswith("settlement_")]
    drop_cols = POST_OUTCOME_COLUMNS + POLICY_LEAKAGE_COLUMNS + IDENTIFIER_OR_TEXT_COLUMNS + hardship_cols + settlement_cols
    df = df.drop(columns=[column for column in drop_cols if column in df.columns], errors="ignore")
    df = prepare_lendingclub_application_frame(df)

    output_path = output_dir / "application_train.csv"
    df.to_csv(output_path, index=False)

    description_path = metadata_dir / "columns_description.csv"
    if not description_path.exists():
        pd.DataFrame(
            {
                "row": df.columns.tolist(),
                "description": [""] * len(df.columns),
                "table": ["application_train"] * len(df.columns),
            }
        ).to_csv(description_path, index=False)

    leakage_path = metadata_dir / "leakage_columns.yaml"
    leakage_path.write_text(
        "label_definition:\n"
        + "  final_good_statuses:\n"
        + "".join(f"    - {column}\n" for column in sorted(LENDINGCLUB_FINAL_GOOD_STATUSES))
        + "  final_bad_statuses:\n"
        + "".join(f"    - {column}\n" for column in sorted(LENDINGCLUB_FINAL_BAD_STATUSES))
        + "  excluded_ongoing_or_ambiguous_statuses:\n"
        + "".join(f"    - {column}\n" for column in sorted(LENDINGCLUB_AMBIGUOUS_OR_ONGOING_STATUSES))
        + "feature_leakage:\n"
        + "  raw_post_outcome_columns:\n"
        + "".join(f"    - {column}\n" for column in POST_OUTCOME_COLUMNS)
        + "  direct_target_leakage_columns:\n"
        + "".join(f"    - {column}\n" for column in LENDINGCLUB_DIRECT_LEAKAGE_COLUMNS)
        + "  underwriting_policy_columns:\n"
        + "".join(f"    - {column}\n" for column in POLICY_LEAKAGE_COLUMNS)
        + "  post_approval_operational_columns:\n"
        + "".join(f"    - {column}\n" for column in LENDINGCLUB_POLICY_OR_POST_APPROVAL_COLUMNS)
        + "  identifier_or_text_columns:\n"
        + "".join(f"    - {column}\n" for column in IDENTIFIER_OR_TEXT_COLUMNS)
        + "".join(f"    - {column}\n" for column in LENDINGCLUB_TEXT_OR_LOW_SIGNAL_COLUMNS)
        + "  additional_pipeline_drops:\n"
        + "".join(f"    - {column}\n" for column in LENDINGCLUB_POST_OUTCOME_LEAKAGE_COLUMNS)
        + "".join(f"    - {column}\n" for column in hardship_cols)
        + "".join(f"    - {column}\n" for column in settlement_cols),
        encoding="utf-8",
    )
    _write_schema_snapshot(raw_file, metadata_dir)

    print(f"Prepared LendingClub file: {output_path}")
    print(f"Rows: {len(df):,} | Columns: {df.shape[1]:,}")
    print(f"Target definition: {target_definition_path}")
    print(f"Label distribution: {label_distribution_path}")
    print(f"Issue-date target distribution: {issue_date_distribution_path}")
    print(f"Description file: {description_path}")
    print(f"Leakage policy: {leakage_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
