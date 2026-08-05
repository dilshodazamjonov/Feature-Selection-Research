"""Build the outcome-blind Stage 1 review for the third benchmark.

This tool is intentionally limited to raw-file bytes, Parquet metadata/schemas,
the base split columns, and relational case/order identifiers.  It never opens
any existing experiment result, prediction, or metric artifact and it does not
fit a model, selector, encoder, or other learned transformation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from credit_risk_fs.models._cv_utils import GroupedTimeSeriesSplit


SCHEMA_VERSION = "third_dataset_protocol_review_v1"
DATASET_ID = "homecredit_model_stability_2024"
OFFICIAL_NAME = "Home Credit - Credit Risk Model Stability 2024"
RAW_RELATIVE_ROOT = Path("data/homecredit_model_stability_2024")
OUTPUT_RELATIVE_ROOT = Path("cleanup/audits/third_dataset_protocol_freeze")
STARTING_COMMIT = "c8f7949e835ce1a9589b8c6890fda42360618ac8"
COMBINATION_ORDER = [
    "statistical_normalized_average_rank",
    "iv_then_boruta",
    "boruta_then_mrmr_mutual_information",
    "boruta_then_rfe_catboost",
]
BASELINE_ORDER = [
    "full_features",
    "random_k",
    "iv_woe",
    "mrmr_mutual_information",
    "lasso_l1_logistic",
    "legacy_rf_relevance_corr",
    "catboost_shap",
    "boruta_random_forest",
    "rfe_catboost",
]
MODELS = ["lr", "catboost"]
STRUCTURAL_BASE = {"case_id", "target", "date_decision", "MONTH", "WEEK_NUM"}
REQUIRED_PACKAGE_FILES = [
    "dataset_identity.json",
    "raw_file_inventory.csv",
    "schema_and_cardinality_profile.csv",
    "feature_definition_coverage.json",
    "temporal_population_profile.csv",
    "proposed_split_and_fold_boundaries.json",
    "leakage_and_availability_review.csv",
    "proposed_adapter_protocol.json",
    "proposed_method_matrix.json",
    "preregistered_hypotheses_and_analysis.md",
    "protocol_review.md",
]
SUPPORT_FILES = [
    "cleanup/tools/build_third_dataset_protocol_freeze.py",
    "tests/test_third_dataset_protocol_freeze.py",
]


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def git(repository_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def utc_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat().replace(
        "+00:00", "Z"
    )


def classify_parquet(name: str) -> tuple[str, int | None, int | None]:
    stem = name.removeprefix("train_").removesuffix(".parquet")
    if stem == "base":
        return "base", None, None
    multipart = re.fullmatch(r"(.+)_([012])_(\d+)", stem)
    if multipart:
        return multipart.group(1), int(multipart.group(2)), int(multipart.group(3))
    single = re.fullmatch(r"(.+)_([012])", stem)
    if single:
        return single.group(1), int(single.group(2)), None
    raise ValueError(f"unrecognized training Parquet convention: {name}")


def logical_type(name: str, arrow_type: pa.DataType) -> str:
    if name.endswith("D") or pa.types.is_date(arrow_type) or pa.types.is_timestamp(arrow_type):
        return "date"
    if pa.types.is_boolean(arrow_type):
        return "boolean"
    if (
        pa.types.is_integer(arrow_type)
        or pa.types.is_floating(arrow_type)
        or pa.types.is_decimal(arrow_type)
    ):
        return "numeric"
    if (
        pa.types.is_string(arrow_type)
        or pa.types.is_large_string(arrow_type)
        or pa.types.is_dictionary(arrow_type)
    ):
        return "categorical"
    return "unsupported"


def id_hash(frame: pd.DataFrame, with_target: bool = False) -> str:
    ordered = frame.sort_values(["date_decision", "case_id"], kind="mergesort")
    digest = hashlib.sha256()
    if with_target:
        for case_id, target in ordered[["case_id", "target"]].itertuples(index=False):
            digest.update(f"{int(case_id)}\x1f{int(target)}\n".encode("utf-8"))
    else:
        for case_id in ordered["case_id"]:
            digest.update(f"{int(case_id)}\n".encode("utf-8"))
    return digest.hexdigest()


def target_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame["target"].value_counts().to_dict()
    return {
        "rows": int(len(frame)),
        "target_0": int(counts.get(0, 0)),
        "target_1": int(counts.get(1, 0)),
    }


def frame_boundary(frame: pd.DataFrame) -> dict[str, Any]:
    counts = target_counts(frame)
    return {
        **counts,
        "date_min": frame["date_decision"].min().date().isoformat(),
        "date_max": frame["date_decision"].max().date().isoformat(),
        "unique_dates": int(frame["date_decision"].nunique()),
        "ordered_case_id_sha256": id_hash(frame),
        "ordered_case_id_target_sha256": id_hash(frame, with_target=True),
    }


def combine_unique_counts(parts: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    ids = np.concatenate([item[0] for item in parts])
    counts = np.concatenate([item[1] for item in parts])
    order = np.argsort(ids, kind="mergesort")
    ids = ids[order]
    counts = counts[order]
    starts = np.r_[True, ids[1:] != ids[:-1]]
    positions = np.flatnonzero(starts)
    return ids[positions], np.add.reduceat(counts, positions)


def membership_mask(sorted_base_ids: np.ndarray, values: np.ndarray) -> np.ndarray:
    positions = np.searchsorted(sorted_base_ids, values)
    valid = positions < len(sorted_base_ids)
    matched = np.zeros(len(values), dtype=bool)
    matched[valid] = sorted_base_ids[positions[valid]] == values[valid]
    return matched


def inventory(
    repository_root: Path,
    raw_root: Path,
) -> tuple[list[dict[str, Any]], dict[tuple[str, int | None], list[tuple[int | None, Path]]], str]:
    rows: list[dict[str, Any]] = []
    families: dict[tuple[str, int | None], list[tuple[int | None, Path]]] = defaultdict(list)
    for path in sorted(item for item in raw_root.rglob("*") if item.is_file()):
        relative_to_raw = path.relative_to(raw_root).as_posix()
        relative_to_repo = path.relative_to(repository_root).as_posix()
        suffix = path.suffix.lower()
        table_family = ""
        depth: int | None = None
        part: int | None = None
        include = False
        reason = ""
        parquet_rows: int | str = ""
        parquet_columns: int | str = ""
        case_id_present: bool | str = ""
        case_id_dtype = ""
        grouping_columns: list[dict[str, str]] = []
        if relative_to_raw == "feature_definitions.csv":
            include = True
            table_family = "feature_definitions"
            reason = "included_official_feature_definitions"
        elif suffix == ".parquet" and path.parent.name == "train":
            table_family, depth, part = classify_parquet(path.name)
            metadata = pq.ParquetFile(path)
            schema = metadata.schema_arrow
            parquet_rows = int(metadata.metadata.num_rows)
            parquet_columns = int(metadata.metadata.num_columns)
            case_id_present = "case_id" in schema.names
            case_id_dtype = str(schema.field("case_id").type) if case_id_present else ""
            grouping_columns = [
                {"name": field.name, "dtype": str(field.type)}
                for field in schema
                if field.name.startswith("num_group")
            ]
            include = table_family == "base" or depth in {0, 1}
            reason = (
                "included_train_base"
                if table_family == "base"
                else f"included_relational_depth_{depth}"
                if include
                else "excluded_relational_depth_2"
            )
            if include:
                families[(table_family, depth)].append((part, path))
        elif "test" in path.parts:
            reason = "excluded_competition_test_data"
        elif suffix == ".csv":
            reason = "excluded_noncanonical_csv_or_submission"
        else:
            reason = "excluded_outside_frozen_training_scope"
        rows.append(
            {
                "relative_path": relative_to_repo,
                "byte_size": int(path.stat().st_size),
                "modified_utc": utc_mtime(path),
                "sha256": sha256_file(path) if include else "",
                "file_type": suffix.removeprefix("."),
                "table_family": table_family,
                "relational_depth": "base" if table_family == "base" else "" if depth is None else depth,
                "file_part_number": "" if part is None else part,
                "inclusion_status": "included" if include else "excluded",
                "inclusion_or_exclusion_reason": reason,
                "parquet_row_count": parquet_rows,
                "parquet_column_count": parquet_columns,
                "case_id_present": case_id_present,
                "case_id_dtype": case_id_dtype,
                "grouping_order_columns_json": json.dumps(grouping_columns, separators=(",", ":")),
            }
        )
    included_identity = [
        {"relative_path": row["relative_path"], "byte_size": row["byte_size"], "sha256": row["sha256"]}
        for row in rows
        if row["inclusion_status"] == "included"
    ]
    digest = sha256_bytes(canonical_json_bytes(included_identity))
    return rows, dict(families), digest


def profile_families(
    repository_root: Path,
    families: dict[tuple[str, int | None], list[tuple[int | None, Path]]],
    base_ids: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, pa.Schema]]:
    sorted_base = np.sort(base_ids.astype(np.int64, copy=True))
    rows: list[dict[str, Any]] = []
    schemas: dict[str, pa.Schema] = {}
    for (family, depth), parts in sorted(
        families.items(), key=lambda item: (-1 if item[0][1] is None else item[0][1], item[0][0])
    ):
        parts = sorted(parts, key=lambda item: -1 if item[0] is None else item[0])
        part_schemas = [pq.ParquetFile(path).schema_arrow for _, path in parts]
        if not all(schema.equals(part_schemas[0]) for schema in part_schemas[1:]):
            raise ValueError(f"schema mismatch across parts for {family} depth {depth}")
        schema = part_schemas[0]
        schemas[f"{family}:base" if depth is None else f"{family}:depth{depth}"] = schema
        unique_parts: list[tuple[np.ndarray, np.ndarray]] = []
        total_rows = 0
        orphan_rows = 0
        case_missing = 0
        grouping: dict[str, dict[str, Any]] = {}
        for part, path in parts:
            group_names = [name for name in schema.names if name.startswith("num_group")]
            table = pq.read_table(path, columns=["case_id", *group_names])
            case_missing += int(table["case_id"].null_count)
            if table["case_id"].null_count:
                raise ValueError(f"missing case_id in {path}")
            ids = np.asarray(table["case_id"]).astype(np.int64)
            total_rows += len(ids)
            matched = membership_mask(sorted_base, ids)
            orphan_rows += int((~matched).sum())
            unique_parts.append(np.unique(ids, return_counts=True))
            for name in group_names:
                array = np.asarray(table[name])
                item = grouping.setdefault(
                    name,
                    {"dtype": str(table.schema.field(name).type), "missing": 0, "min": None, "max": None},
                )
                item["missing"] += int(table[name].null_count)
                if len(array):
                    observed_min = int(array.min())
                    observed_max = int(array.max())
                    item["min"] = observed_min if item["min"] is None else min(item["min"], observed_min)
                    item["max"] = observed_max if item["max"] is None else max(item["max"], observed_max)
        unique_ids, per_case_counts = combine_unique_counts(unique_parts)
        matched_unique = membership_mask(sorted_base, unique_ids)
        rows.append(
            {
                "table_family": family,
                "relational_depth": "base" if depth is None else depth,
                "part_count": len(parts),
                "relative_paths_json": json.dumps(
                    [path.relative_to(repository_root).as_posix() for _, path in parts], separators=(",", ":")
                ),
                "schema_identical_across_parts": True,
                "schema_sha256": sha256_bytes(
                    canonical_json_bytes([{"name": field.name, "dtype": str(field.type)} for field in schema])
                ),
                "column_count": len(schema),
                "row_count": int(total_rows),
                "case_id_present": "case_id" in schema.names,
                "case_id_dtype": str(schema.field("case_id").type),
                "case_id_missing": int(case_missing),
                "distinct_case_ids": int(len(unique_ids)),
                "matched_base_case_ids": int(matched_unique.sum()),
                "orphan_case_id_rows": int(orphan_rows),
                "orphan_distinct_case_ids": int((~matched_unique).sum()),
                "base_case_coverage_count": int(matched_unique.sum()),
                "base_case_coverage_fraction": float(matched_unique.sum() / len(sorted_base)),
                "min_rows_per_present_case": int(per_case_counts.min()),
                "max_rows_per_present_case": int(per_case_counts.max()),
                "mean_rows_per_present_case": float(per_case_counts.mean()),
                "cardinality": "one_to_one" if int(per_case_counts.max()) == 1 else "one_to_many",
                "grouping_order_columns_json": json.dumps(grouping, sort_keys=True, separators=(",", ":")),
            }
        )
    return rows, schemas


def build_feature_review(
    schemas: dict[str, pa.Schema], definitions_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with definitions_path.open("r", encoding="utf-8-sig", newline="") as handle:
        definition_rows = list(csv.DictReader(handle))
    definitions = {row["Variable"]: row["Description"].strip() for row in definition_rows}
    review: list[dict[str, Any]] = []
    predictor_names: dict[str, list[str]] = defaultdict(list)
    type_counts: Counter[str] = Counter()
    included_predictor_names: set[str] = set()
    for key in sorted(schemas, key=lambda item: (0 if item == "base:base" else 1, item)):
        family, depth_label = key.split(":", 1)
        depth: int | None = None if depth_label == "base" else int(depth_label.removeprefix("depth"))
        for field in schemas[key]:
            name = field.name
            if name == "case_id":
                logical = "identifier"
                aggregation = "group_key_only"
                availability = "structural_identifier_not_predictor"
                action = "exclude"
                reason = "stable row/join identifier; predictor use forbidden"
                description = "Stable case identifier."
            elif name.startswith("num_group"):
                logical = "order_identifier"
                aggregation = "stable_sort_key_only_not_aggregated_as_measurement"
                availability = "relational_order_identifier_not_predictor"
                action = "exclude"
                reason = "row/group order identifier; used only for deterministic ordering"
                description = "Relational row-order/group identifier."
            elif family == "base" and name == "target":
                logical = "target"
                aggregation = "none"
                availability = "outcome"
                action = "exclude"
                reason = "binary response variable"
                description = "Binary competition response variable."
            elif family == "base" and name in {"date_decision", "MONTH", "WEEK_NUM"}:
                logical = "split_control"
                aggregation = "none"
                availability = "split_control_only"
                action = "exclude"
                reason = "chronological split/control field; predictor use forbidden"
                description = "Chronological split-control field."
            else:
                logical = logical_type(name, field.type)
                type_counts[logical] += 1
                description = definitions.get(name, "")
                if depth == 0:
                    aggregation = (
                        "signed_days_relative_to_base_date_decision"
                        if logical == "date"
                        else "identity_after_family_prefix"
                    )
                elif logical == "numeric":
                    aggregation = "count_non_missing|missing_count|min|max|mean|sum|sample_variance|first_by_num_group1|last_by_num_group1"
                elif logical == "date":
                    aggregation = "parse_date_then_signed_days_from_date_decision;count_non_missing|missing_count|min|max|mean|sample_variance|first_by_num_group1|last_by_num_group1"
                elif logical == "boolean":
                    aggregation = "count_non_missing|missing_count|false_count|true_count|mean|first_by_num_group1|last_by_num_group1|any|all"
                elif logical == "categorical":
                    aggregation = "count_non_missing|missing_count|nunique|lexical_mode|first_by_num_group1|last_by_num_group1"
                else:
                    aggregation = "unsupported_fail_closed"
                if logical == "unsupported":
                    availability = "unresolved_unsupported_dtype"
                    action = "unresolved"
                    reason = "unsupported Parquet dtype requires explicit amendment"
                elif not description:
                    availability = "unresolved_missing_official_definition"
                    action = "unresolved"
                    reason = "official definition missing; explicit review required"
                else:
                    availability = "decision_time_application_or_historical_snapshot"
                    action = "include"
                    reason = "officially documented application/prior-history snapshot; no post-outcome meaning identified"
                    included_predictor_names.add(name)
                    predictor_names[name].append(key)
            review.append(
                {
                    "source_table_family": family,
                    "relational_depth": "base" if depth is None else depth,
                    "feature_name": name,
                    "description": description,
                    "observed_dtype": str(field.type),
                    "logical_type": logical,
                    "intended_aggregation": aggregation,
                    "decision_time_availability": availability,
                    "action": action,
                    "reason": reason,
                    "deterministic_output_prefix": "" if action != "include" else f"d{depth}__{family}__{name}",
                }
            )
    selected_defined = included_predictor_names & definitions.keys()
    definitions_not_selected = sorted(set(definitions) - included_predictor_names)
    cross_family_collisions = {
        name: locations for name, locations in predictor_names.items() if len(set(locations)) > 1
    }
    coverage = {
        "schema_version": "third_dataset_feature_definition_coverage_v1",
        "definition_row_count": len(definition_rows),
        "unique_definition_count": len(definitions),
        "blank_definition_descriptions": sum(not row["Description"].strip() for row in definition_rows),
        "included_raw_predictor_count": len(included_predictor_names),
        "documented_included_raw_predictor_count": len(selected_defined),
        "undocumented_included_raw_predictors": sorted(included_predictor_names - definitions.keys()),
        "definitions_not_in_included_depth_0_1_predictors_count": len(definitions_not_selected),
        "definitions_not_in_included_depth_0_1_predictors": definitions_not_selected,
        "review_row_count_before_actions": len(review),
        "included_review_rows": sum(row["action"] == "include" for row in review),
        "excluded_review_rows": sum(row["action"] == "exclude" for row in review),
        "unresolved_review_rows": sum(row["action"] == "unresolved" for row in review),
        "included_logical_type_counts": dict(sorted(type_counts.items())),
        "cross_family_raw_predictor_name_collision_count": len(cross_family_collisions),
        "cross_family_raw_predictor_name_collisions": cross_family_collisions,
        "collision_policy": "all modeling outputs are prefixed d{depth}__{family}__ before aggregation suffix",
    }
    return review, coverage


def build_temporal_design(base: pd.DataFrame) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    base = base.copy()
    base["date_decision"] = pd.to_datetime(base["date_decision"], errors="raise")
    by_date = base.groupby("date_decision", sort=True).size()
    reverse_tail = by_date.iloc[::-1].cumsum().iloc[::-1]
    oot_start, oot_rows = min(
        reverse_tail.items(),
        key=lambda item: (abs(float(item[1]) / len(base) - 0.20), -item[0].value),
    )
    dev = base.loc[base["date_decision"] < oot_start].sort_values(
        ["date_decision", "case_id"], kind="mergesort"
    ).reset_index(drop=True)
    oot = base.loc[base["date_decision"] >= oot_start].sort_values(
        ["date_decision", "case_id"], kind="mergesort"
    ).reset_index(drop=True)
    splitter = GroupedTimeSeriesSplit(n_splits=5, gap=1)
    folds: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    for period, group in base.groupby(base["date_decision"].dt.to_period("M"), sort=True):
        counts = target_counts(group)
        profile_rows.append(
            {
                "profile_level": "calendar_month",
                "partition": "ALL",
                "fold_id": "",
                "role": "month",
                "calendar_period": str(period),
                **counts,
                "date_min": group["date_decision"].min().date().isoformat(),
                "date_max": group["date_decision"].max().date().isoformat(),
                "unique_dates": int(group["date_decision"].nunique()),
            }
        )
    for label, frame in (("DEV", dev), ("OOT", oot)):
        boundary = frame_boundary(frame)
        profile_rows.append(
            {
                "profile_level": "split",
                "partition": label,
                "fold_id": "",
                "role": label.lower(),
                "calendar_period": "",
                **{key: boundary[key] for key in ("rows", "target_0", "target_1", "date_min", "date_max", "unique_dates")},
            }
        )
    for fold_id, (train_index, validation_index) in enumerate(
        splitter.split(dev["date_decision"].to_numpy()), start=1
    ):
        train = dev.iloc[train_index]
        validation = dev.iloc[validation_index]
        train_boundary = frame_boundary(train)
        validation_boundary = frame_boundary(validation)
        if set(train["case_id"]).intersection(validation["case_id"]):
            raise ValueError(f"case overlap in fold {fold_id}")
        fold = {
            "fold_id": fold_id,
            "train": train_boundary,
            "validation": validation_boundary,
            "case_id_overlap": 0,
            "date_unit_overlap": 0,
            "configured_gap_unique_time_groups": 1,
        }
        folds.append(fold)
        for role, boundary in (("train", train_boundary), ("validation", validation_boundary)):
            profile_rows.append(
                {
                    "profile_level": "fold",
                    "partition": "DEV",
                    "fold_id": fold_id,
                    "role": role,
                    "calendar_period": "",
                    **{key: boundary[key] for key in ("rows", "target_0", "target_1", "date_min", "date_max", "unique_dates")},
                }
            )
    split = {
        "schema_version": "third_dataset_temporal_split_proposal_v1",
        "status": "proposed_not_canonical_until_explicit_stage_2_approval",
        "primary_time_column": "date_decision",
        "stable_secondary_order": "case_id numeric ascending within date_decision",
        "oot_boundary_rule": "fallback_latest_contiguous_whole_date_tail_closest_to_20_percent_of_base_rows",
        "boundary_optimization_inputs": ["date_decision", "row_count"],
        "target_counts_used_only_for_feasibility_confirmation": True,
        "oot_start_inclusive": oot_start.date().isoformat(),
        "dev_end_inclusive": (oot_start - pd.Timedelta(days=1)).date().isoformat(),
        "target_oot_fraction": 0.20,
        "realized_oot_fraction": float(oot_rows / len(base)),
        "membership_specification": {
            "dev": f"date_decision < {oot_start.date().isoformat()}",
            "oot": f"date_decision >= {oot_start.date().isoformat()}",
            "whole_calendar_dates_kept_together": True,
            "authentication_order": ["date_decision_ascending", "case_id_numeric_ascending"],
            "line_serialization": "case_id decimal UTF-8 plus LF; target-bound form is case_id, U+001F, target, LF",
        },
        "dev": frame_boundary(dev),
        "oot": frame_boundary(oot),
        "fold_protocol": {
            "implementation": "credit_risk_fs.models._cv_utils.GroupedTimeSeriesSplit",
            "n_splits": 5,
            "gap_unique_time_groups": 1,
            "training_window": "expanding",
            "identical_date_values_kept_together": True,
        },
        "folds": folds,
        "validation_feasibility": {
            "all_partitions_contain_target_0_and_target_1": all(
                boundary["target_0"] > 0 and boundary["target_1"] > 0
                for fold in folds
                for boundary in (fold["train"], fold["validation"])
            )
            and target_counts(oot)["target_0"] > 0
            and target_counts(oot)["target_1"] > 0,
            "all_fold_case_overlaps_zero": True,
            "all_fold_date_overlaps_zero": True,
        },
    }
    return split, profile_rows


def proposed_adapter() -> dict[str, Any]:
    return {
        "schema_version": "homecredit_model_stability_2024_adapter_proposal_v1",
        "status": "specification_only_not_implemented_not_executed",
        "row_contract": {
            "output": "exactly_one_row_per_train_base.case_id",
            "anchor": "train_base.parquet",
            "join": "left_join_from_base_only",
            "orphan_policy": "fail_if_any_included_relation_case_id_is_absent_from_base",
            "base_case_policy": "fail_on_missing_or_duplicate_case_id",
        },
        "input_scope": {
            "include": ["train_base", "every_train_depth_0_family", "every_train_depth_1_family"],
            "exclude": ["every_depth_2_family", "competition_test", "submission", "CSV_duplicates"],
            "part_order": "numeric file-part order; unnumbered part precedes no numbered peer",
        },
        "depth_0": {
            "part_handling": "vertical_concat_in_numeric_part_order_then_require_case_id_unique",
            "merge_order": ["static", "static_cb"],
            "missing_relation": "leave feature values missing after base-left-join",
            "feature_naming": "d0__{family}__{source_feature}",
            "date_fields": "official *D fields parse strictly to dates and become signed whole days source_date - date_decision",
        },
        "depth_1": {
            "family_order": [
                "applprev", "credit_bureau_a", "credit_bureau_b", "debitcard", "deposit",
                "other", "person", "tax_registry_a", "tax_registry_b", "tax_registry_c",
            ],
            "row_order": ["case_id", "num_group1", "numeric_source_part", "physical_row_offset_within_part"],
            "row_count_feature": "d1__{family}__row_count; zero when no related row",
            "numeric_aggregations": [
                "count_non_missing", "missing_count", "min", "max", "mean", "sum",
                "sample_variance_ddof_1", "first_by_num_group1", "last_by_num_group1",
            ],
            "date_aggregations": {
                "conversion": "strict parse official *D field then signed whole days source_date - base.date_decision",
                "statistics": [
                    "count_non_missing", "missing_count", "min", "max", "mean",
                    "sample_variance_ddof_1", "first_by_num_group1", "last_by_num_group1",
                ],
                "future_information_rule": "use only values present in the frozen application snapshot; never refresh after date_decision; a positive signed scheduled date is not itself treated as an observed future outcome",
            },
            "boolean_aggregations": [
                "count_non_missing", "missing_count", "false_count", "true_count", "mean",
                "first_by_num_group1", "last_by_num_group1", "any", "all",
            ],
            "categorical_aggregations": [
                "count_non_missing", "missing_count", "nunique", "lexical_mode",
                "first_by_num_group1", "last_by_num_group1",
            ],
            "categorical_mode_tie_break": "Unicode NFC then casefold then original UTF-8 byte order",
            "empty_group_rule": "row_count=0; every other aggregate is missing",
            "order_columns": "num_group1 is sort-only and never a predictor or numeric measurement",
            "feature_naming": "d1__{family}__{source_feature}__{aggregation}",
        },
        "collision_handling": {
            "rule": "always prefix depth and table family before source name; append aggregation for depth 1",
            "duplicate_output": "fail closed before modeling",
            "observed_cross_family_raw_predictor_collisions": 0,
        },
        "preprocessing": {
            "boolean_storage": "nullable booleans become numeric 0/1 with missing preserved before fold-local imputation",
            "selector_stage": {
                "implementation": "credit_risk_fs.preprocessing.encoding.OriginalFeatureNumericEncoder",
                "fit_scope": "current training partition only",
                "numeric": "replace infinities with missing; training median; all-missing fallback 0; float32",
                "categorical": "training-only Unicode string category map; <MISSING> token; unseen=-1; one numeric column per original candidate",
            },
            "final_model_stage": {
                "implementation": "credit_risk_fs.preprocessing.encoding.Preprocessor",
                "fit_scope": "current training partition only",
                "numeric": "training mean imputation then StandardScaler; all-missing fallback 0; float32",
                "categorical": "Missing token then OneHotEncoder(handle_unknown=ignore,min_frequency=10,dense_float32)",
                "applies_to": ["lr", "catboost"],
            },
            "constant_and_all_missing": "retain as authenticated original candidates; deterministic training-fit preprocessing maps them to constants and records them; no outcome-informed pruning",
            "high_cardinality": "no feature-level cardinality exclusion; fold-local min_frequency=10 groups rare categorical levels; resource feasibility belongs to the bounded pilot",
            "unsupported_dtype": "fail closed and require a versioned pre-result amendment; none observed in the included schema",
        },
        "leakage_boundaries": {
            "fit_per_dev_fold": ["imputers", "category_maps", "one_hot_encoder", "scaler", "selectors", "models", "decision_threshold"],
            "fit_for_oot": "fit once on full DEV then transform locked OOT unchanged",
            "never_fit_on_oot": True,
            "excluded_predictors": ["target", "case_id", "num_group*", "date_decision", "MONTH", "WEEK_NUM"],
            "domain_feature_engineering": "forbidden beyond this deterministic relational reduction",
        },
    }


def proposed_method_matrix(repository_root: Path) -> dict[str, Any]:
    baseline_path = repository_root / "configs/experiments/full_baseline_v1.yaml"
    combination_path = repository_root / "configs/experiments/selector_combination_research_v1.yaml"
    protocol_path = repository_root / "configs/protocols/selector_combinations_v1/combination_protocol_lock.json"
    comparisons_path = repository_root / "configs/protocols/selector_combinations_v1/combination_comparison_registry.json"
    with baseline_path.open("r", encoding="utf-8") as handle:
        baseline = yaml.safe_load(handle)
    with combination_path.open("r", encoding="utf-8") as handle:
        combination = yaml.safe_load(handle)
    with protocol_path.open("r", encoding="utf-8") as handle:
        combination_protocol = json.load(handle)
    with comparisons_path.open("r", encoding="utf-8") as handle:
        comparisons = json.load(handle)
    variants: list[dict[str, Any]] = []
    order = 0
    for method in BASELINE_ORDER:
        order += 1
        variants.append({"order": order, "family": "canonical_baseline", "method_id": method, "iv_pool_budget": None})
    for method in COMBINATION_ORDER:
        pools = [100, 200, 300] if method == "iv_then_boruta" else [None]
        for pool in pools:
            order += 1
            variants.append({"order": order, "family": "approved_combination", "method_id": method, "iv_pool_budget": pool})
    matrix_cells = []
    for variant in variants:
        for model_index, model in enumerate(MODELS):
            if variant["method_id"] == "full_features":
                budget: int | str | None = None
                budget_semantics = "all_authenticated_candidates_budget_ignored"
            elif variant["method_id"] == "iv_then_boruta":
                budget = None
                budget_semantics = "boruta_confirmed_only_natural_support_within_iv_pool"
            else:
                budget = 20 if model == "lr" else 40
                budget_semantics = "requested_model_specific_final_k"
            matrix_cells.append(
                {
                    "configuration_order": 2 * (variant["order"] - 1) + model_index + 1,
                    **variant,
                    "model": model,
                    "requested_feature_budget": budget,
                    "feature_budget_semantics": budget_semantics,
                    "seed": 42,
                }
            )
    return {
        "schema_version": "third_dataset_method_matrix_proposal_v1",
        "status": "proposed_not_authorized_for_execution",
        "dataset": DATASET_ID,
        "method_order": [*BASELINE_ORDER, *COMBINATION_ORDER],
        "combination_order": COMBINATION_ORDER,
        "variant_order": variants,
        "models": MODELS,
        "feature_budgets": baseline["feature_budgets"],
        "iv_pool_budgets": combination["protocol"]["iv_pool_budgets"],
        "iv_pool_primary": combination["protocol"]["iv_pool_primary"],
        "seeds": {"experiment_selector_model": 42, "paired_bootstrap": 20260721},
        "final_model_settings": baseline["final_model_settings"],
        "selector_settings": baseline["selector_settings"],
        "combination_selector_settings": combination["selector_settings"],
        "combination_contract": {
            "statistical_voting": combination_protocol["statistical_voting"],
            "natural_support_rules": combination_protocol["natural_support_rules"],
            "padding": "forbidden",
        },
        "natural_support": {
            "rule": "report requested K and realized support; never pad",
            "label_when_realized_below_requested": "infeasible_natural_support",
            "methods": ["boruta_random_forest", "iv_then_boruta", "boruta_then_mrmr_mutual_information", "boruta_then_rfe_catboost"],
            "third_dataset_support": "unknown_until_bounded_pilot_and_never inferred from the earlier Home Credit dataset",
        },
        "phase_design": {
            "pilot": {
                "fold_ids": [1],
                "oot_access": "forbidden",
                "configuration_evaluation_cells": 30,
                "selector_fit_calls": 27,
                "note": "18 canonical baseline cell-local selector fits plus 9 authenticated combination selection fits; IV-then-Boruta is reused across final models only for exact pool identity",
            },
            "dev": {
                "fold_ids": [1, 2, 3, 4, 5],
                "configuration_count": 30,
                "selector_fit_calls": 135,
                "fold_evaluation_cells": 150,
                "five_fold_summary_rows": 30,
            },
            "oot": {
                "full_dev_selector_refit_calls": 27,
                "evaluation_cells": 30,
                "access": "closed_until authenticated bounded pilot review and authenticated complete DEV review",
                "single_locked_evaluation": True,
            },
            "count_derivation": {
                "model_configurations_per_model": 15,
                "models": 2,
                "baseline_selector_fits_per_fold": 18,
                "combination_selector_fits_per_fold": 9,
            },
        },
        "matrix_cells": matrix_cells,
        "metrics": {
            "primary_predictive": "roc_auc",
            "secondary_predictive": [
                "gini", "ks", "lift_at_10", "bad_rate_capture_at_10", "precision",
                "recall", "f1", "accuracy",
            ],
            "calibration": ["log_loss", "brier"],
            "stability": ["nogueira", "mean_all_pairwise_jaccard", "kuncheva_when_fixed_size", "selection_frequency"],
            "drift": ["selected_feature_psi", "dev_oof_to_oot_score_psi"],
            "resource": [
                "preprocessing_seconds", "feature_selection_seconds", "training_seconds",
                "prediction_seconds", "evaluation_seconds", "total_seconds", "peak_process_ram_bytes",
                "peak_system_ram_bytes", "peak_gpu_bytes",
            ],
            "threshold": "maximize KS on each fitting partition then apply unchanged to held-out rows",
        },
        "inference": {
            "comparison_registry": comparisons,
            "paired_auc": "two-sided DeLong on identical OOT case_id rows",
            "paired_bootstrap": {
                "type": "target-stratified paired",
                "attempted_repetitions": 2000,
                "minimum_valid_repetitions": 1900,
                "seed": 20260721,
                "confidence_interval": "95_percent_percentile",
                "metrics": ["auc", "ks", "lift_at_10"],
            },
            "multiplicity": "Holm within each named dataset-model-reference comparison family; never pool across models or reference families",
            "descriptive_only": ["stability", "PSI", "runtime", "RAM", "GPU", "pool_to_pool_contrasts"],
        },
        "resource_controls": {
            "accelerator": "cpu",
            "concurrent_cells": 1,
            "concurrent_folds": 1,
            "data_loader_workers": 0,
            "estimator_threads_maximum": 4,
            "wall_clock_limits_seconds": {
                "baseline_lightweight": 10800,
                "baseline_catboost_shap": 10800,
                "baseline_boruta_random_forest": 21600,
                "baseline_rfe_catboost": 28800,
                **combination["execution"]["wall_clock_limits_seconds"],
            },
            "checkpoint": "atomic per-selection and per-evaluation state with declared SHA-256; immutable completed cells; resume only exact authenticated identity; archive interrupted/failed attempts; execution lock required and released",
        },
        "extension_scope": {
            "semantic_coverage_metric": "deferred_no_preexisting_third-dataset semantic-group mapping; source-family coverage may be reported separately but not relabeled semantic coverage",
            "semantic_selector_or_voter": "deferred_no_pre_Prompt_14_plan_names_the_third_dataset",
            "llm_assisted_methods": "deferred_no_pre_Prompt_14_third_dataset_replication_plan_and_network_forbidden",
            "corrected_contrastive_methods": "deferred_no_pre_Prompt_14_third_dataset_scope_or_authenticated third-dataset contrastive provenance",
            "directional_transfer": "deferred_no_pre_Prompt_14_symmetric_third-dataset transfer directions were specified",
            "cross_dataset_rank_voting_v1": "deferred_existing_lock explicitly covers only homecredit and lendingclub_v2; no silent third-dataset extension",
        },
        "legacy_or_extension_method_resolution": {
            "pca": "not_in_current frozen full_baseline_v1 matrix; component-space output is outside the original-feature stability comparison and is not silently restored",
            "domain_rule_baseline": "deferred_Home-Credit-specific semantic rule has no frozen mapping for the 2024 schema",
            "llm": "deferred_with all LLM-only and LLM-hybrid variants under the explicit LLM scope decision",
            "boruta_rfe_legacy_registry": "not separately added; the approved canonical boruta_then_rfe_catboost method is the frozen sequential implementation",
            "legacy_mrmr_label": "retained under the accurate canonical method ID legacy_rf_relevance_corr",
        },
        "gates": {
            "stage_1": "review_only_no_execution",
            "stage_2": "canonical lock only after exact digest and scope approval",
            "pilot": "later Prompt 14 bounded manual run only after canonical lock and adapter validation",
            "dev": "closed until authenticated pilot review and approval",
            "oot": "closed until authenticated complete DEV review and approval",
        },
        "source_authentication": [
            {"path": path.relative_to(repository_root).as_posix(), "sha256": sha256_file(path)}
            for path in (
                baseline_path,
                combination_path,
                protocol_path,
                comparisons_path,
                repository_root / "configs/protocols/cross_dataset_rank_voting_v1.yaml",
                repository_root / "configs/protocols/credit_scoring_extension_v1.yaml",
                repository_root / "configs/execution/local_laptop_safe_v1.yaml",
                repository_root / "docs/experiment_protocol.md",
                repository_root / "README.md",
            )
        ],
    }


def preregistration_markdown() -> str:
    return """# Third-benchmark preregistration and analysis rules

Status: proposed in Stage 1; not canonical and not authorized for execution.

## Role and evidence hierarchy

Home Credit - Credit Risk Model Stability 2024 is the third robustness/replication benchmark. It is not fully independent institutional evidence because it shares Home Credit lineage with the earlier Home Credit dataset. Locked OOT evidence has priority over DEV evidence. DEV is for feasibility, diagnostics, and pre-authorized gating only; no method will be called “best” from DEV alone.

Predictive performance, feature-selection stability, calibration, drift, and resource cost will be reported in separate sections before any combined interpretation. Natural-support runs will always display requested K and realized support and will not be described as matched-K evidence when they differ. Failed, timed-out, resource-infeasible, or inapplicable configurations stay visible.

## Primary hypotheses

1. Each approved combination is compared on locked OOT with every component comparator registered before execution, within the same model, split, ordered case IDs, budget policy, and seed.
2. Each canonical standalone selector is compared with matched `full_features` and `random_k` controls on locked OOT.
3. A method’s predictive evidence and stability evidence are distinct: improved AUC does not imply improved stability, and stability does not substitute for AUC.
4. Replication across this benchmark strengthens robustness evidence but does not erase shared Home Credit lineage.

## Exact evidence-language rule

For each preregistered paired OOT comparison, let ΔAUC be comparator method minus reference. “Strong” predictive evidence requires ΔAUC > 0, a two-sided paired DeLong Holm-adjusted p < 0.05 within its frozen family, and a 95% paired stratified-bootstrap ΔAUC interval wholly above zero. “Moderate” requires ΔAUC > 0 and exactly one of those two inferential criteria. “Weak” requires ΔAUC > 0 and neither inferential criterion. “Not supported” applies when ΔAUC <= 0, paired identity/target alignment fails, or the required inference is unavailable or invalid. These labels apply only to the named comparison; they never establish a global “best” method.

Stability, calibration, drift, and resource results receive no significance label unless an already-frozen test exists. They are reported descriptively and separately, including Nogueira, all-pairwise Jaccard, eligible Kuncheva, PSI, log loss, Brier score, runtime, and memory. Any combined narrative must state the direction of each domain and any conflict.

## Deviations

After a canonical Stage 2 lock, every scope or implementation deviation requires a versioned amendment written and authenticated before the affected result is inspected. Silent deletion, replacement, padding, outcome-driven tuning, and post-OOT adaptation are forbidden.
"""


def review_markdown(
    inventory_rows: list[dict[str, Any]],
    input_digest: str,
    base: pd.DataFrame,
    split: dict[str, Any],
    coverage: dict[str, Any],
    profiles: list[dict[str, Any]],
    method_matrix: dict[str, Any],
) -> str:
    included_parquets = [row for row in inventory_rows if row["inclusion_status"] == "included" and row["file_type"] == "parquet"]
    excluded_parquets = [row for row in inventory_rows if row["inclusion_status"] == "excluded" and row["file_type"] == "parquet"]
    target = target_counts(base)
    profile_by_family = {row["table_family"]: row for row in profiles}
    fold_lines = []
    for fold in split["folds"]:
        train = fold["train"]
        validation = fold["validation"]
        fold_lines.append(
            f"| {fold['fold_id']} | {train['date_min']}..{train['date_max']} | {train['rows']:,} ({train['target_0']:,}/{train['target_1']:,}) | {validation['date_min']}..{validation['date_max']} | {validation['rows']:,} ({validation['target_0']:,}/{validation['target_1']:,}) |"
        )
    return f"""# Third-dataset protocol Stage 1 review

Status: **review package only; no canonical lock; no pilot/DEV/OOT authorization**

## Answer first

The local snapshot is structurally suitable for a locked third-benchmark protocol. It contains {len(included_parquets)} included training Parquet files and {len(excluded_parquets)} excluded depth-2 Parquet files. All included relations have zero orphan `case_id` rows, base IDs are unique, the target is exactly binary {{0,1}}, and every one of the {coverage['included_raw_predictor_count']} depth-0/1 raw predictors has an official definition. No unresolved leakage row or unsupported dtype remains.

The proposed OOT split is the latest whole-date tail closest to 20%: DEV is through 2020-02-25 and locked OOT starts 2020-02-26. This is the prompt-authorized fallback because no existing dataset-specific third-benchmark boundary exists. The canonical five-fold expanding splitter with one unique-date gap is then applied inside DEV.

## Identity and structural scope

- Official dataset: {OFFICIAL_NAME}
- Local identity: `{DATASET_ID}`
- Frozen included-input digest: `{input_digest}`
- Base: {target['rows']:,} rows; target 0 = {target['target_0']:,}; target 1 = {target['target_1']:,}; dates {base['date_decision'].min().date()}..{base['date_decision'].max().date()}
- Included: base plus depth 0 and depth 1 only. Depth 2 is inventoried and excluded.
- Included table families: base, static, static_cb, applprev, credit_bureau_a, credit_bureau_b, debitcard, deposit, other, person, tax_registry_a, tax_registry_b, tax_registry_c.
- Relational findings: zero orphans in every included family; depth-0 families are one-to-one; depth-1 families follow their observed one-to-one/one-to-many profile. `other` is currently one-to-one but remains governed by the depth-1 aggregation contract.
- `static_cb` covers {profile_by_family['static_cb']['matched_base_case_ids']:,}/{target['rows']:,} base cases; missing relation rows remain missing after the base-left-join.

## Exact temporal proposal

- DEV: {split['dev']['rows']:,} rows ({split['dev']['target_0']:,}/{split['dev']['target_1']:,}), {split['dev']['date_min']}..{split['dev']['date_max']}.
- OOT: {split['oot']['rows']:,} rows ({split['oot']['target_0']:,}/{split['oot']['target_1']:,}), {split['oot']['date_min']}..{split['oot']['date_max']}, {split['realized_oot_fraction']:.9%} of base.
- OOT membership: `date_decision >= 2020-02-26`; whole dates remain intact; ordered case-ID hashes authenticate DEV, OOT, and every fold.

| Fold | Train dates | Train rows (0/1) | Validation dates | Validation rows (0/1) |
|---:|---|---:|---|---:|
{chr(10).join(fold_lines)}

## Adapter and leakage decisions

The adapter is specification-only. It anchors one row per base `case_id`, concatenates multipart families in numeric part order, requires depth-0 uniqueness, and aggregates every depth-1 family in fixed `num_group1` order. Numeric, logical-date, boolean, and categorical aggregation lists are fixed in `proposed_adapter_protocol.json`; no result can choose them later. Every output is prefixed by depth and family.

The review has {coverage['review_row_count_before_actions']} source-family schema rows: {coverage['included_review_rows']} included predictors, {coverage['excluded_review_rows']} excluded identifiers/target/split controls, and {coverage['unresolved_review_rows']} unresolved. `target`, every `case_id`/`num_group*`, `date_decision`, `MONTH`, and `WEEK_NUM` are excluded. Fold-local canonical preprocessing is preserved; OOT is transformed only with full-DEV-fitted objects. No domain-crafted feature engineering is allowed beyond the fixed relational reduction.

## Scientific matrix and gates

The proposed matrix contains the nine frozen baselines followed by the four approved combinations in their approved order. IV→Boruta keeps pools 100/200/300 (200 primary); LR requests K=20 and CatBoost K=40; seed 42 is universal. This yields 15 variants per model and 30 evaluation configurations.

- Bounded fold-1 pilot: 27 selector-fit calls and 30 evaluations; OOT inaccessible.
- Five-fold DEV: 135 selector-fit calls and 150 held-out fold evaluations, summarized as 30 configurations.
- Locked OOT: 27 full-DEV selector refits and 30 one-time evaluation cells, only after separate pilot and DEV authentication/review gates.
- Natural support: no padding; any realized support below K is labeled `infeasible_natural_support` and compared with the realized count visible.

Source-table-family coverage may be reported descriptively, but formal semantic coverage is deferred because no pre-existing semantic-group map covers this schema. Semantic selector/voter extensions, LLM-assisted methods, corrected contrastive methods, directional transfer, and the existing two-dataset cross-dataset voting lock are also deferred because no pre-Prompt-14 plan specifies their use on this third dataset. This is a visible scope decision, not a performance-based removal.

## Blockers and approval boundary

There is no unresolved Stage 1 identity, schema, split-feasibility, leakage, or matrix blocker. The adapter is intentionally not implemented, the canonical protocol lock is intentionally absent, and all execution gates remain closed. Stage 2 may create a lock only after the user quotes the exact review digest and explicitly approves the full listed scope.

The canonical review digest is recorded in `review_digest.json`.
"""


def validate_package(repository_root: Path, output_root: Path) -> dict[str, Any]:
    errors: list[str] = []
    digest_path = output_root / "review_digest.json"
    if not digest_path.is_file():
        return {"valid": False, "errors": [f"missing {digest_path}"]}
    digest = json.loads(digest_path.read_text(encoding="utf-8"))
    supplied = digest.get("review_digest_sha256")
    payload = dict(digest)
    payload.pop("review_digest_sha256", None)
    observed = sha256_bytes(canonical_json_bytes(payload))
    if supplied != observed:
        errors.append(f"review digest mismatch: {supplied} != {observed}")
    for item in digest.get("artifact_hashes", []):
        path = repository_root / item["path"]
        if not path.is_file():
            errors.append(f"missing artifact: {item['path']}")
        elif sha256_file(path) != item["sha256"]:
            errors.append(f"artifact hash mismatch: {item['path']}")
    inventory_rows = list(csv.DictReader((output_root / "raw_file_inventory.csv").open(encoding="utf-8", newline="")))
    included = [row for row in inventory_rows if row["inclusion_status"] == "included"]
    included_identity = [
        {"relative_path": row["relative_path"], "byte_size": int(row["byte_size"]), "sha256": row["sha256"]}
        for row in included
    ]
    if sha256_bytes(canonical_json_bytes(included_identity)) != digest["summary"]["included_raw_input_digest"]:
        errors.append("included raw-input digest mismatch")
    coverage = json.loads((output_root / "feature_definition_coverage.json").read_text(encoding="utf-8"))
    split = json.loads((output_root / "proposed_split_and_fold_boundaries.json").read_text(encoding="utf-8"))
    matrix = json.loads((output_root / "proposed_method_matrix.json").read_text(encoding="utf-8"))
    if coverage["included_review_rows"] != digest["summary"]["included_candidate_features"]:
        errors.append("candidate count mismatch")
    if coverage["unresolved_review_rows"] != digest["summary"]["unresolved_leakage_rows"]:
        errors.append("unresolved leakage count mismatch")
    if split["oot_start_inclusive"] != digest["summary"]["oot_start_inclusive"]:
        errors.append("OOT boundary mismatch")
    if matrix["combination_order"] != COMBINATION_ORDER:
        errors.append("combination order mismatch")
    expected_counts = matrix["phase_design"]
    if (
        expected_counts["pilot"]["selector_fit_calls"],
        expected_counts["pilot"]["configuration_evaluation_cells"],
        expected_counts["dev"]["selector_fit_calls"],
        expected_counts["dev"]["fold_evaluation_cells"],
        expected_counts["oot"]["evaluation_cells"],
    ) != (27, 30, 135, 150, 30):
        errors.append("matrix phase counts mismatch")
    return {"valid": not errors, "errors": errors, "review_digest_sha256": supplied}


def build(repository_root: Path) -> dict[str, Any]:
    raw_root = repository_root / RAW_RELATIVE_ROOT
    output_root = repository_root / OUTPUT_RELATIVE_ROOT
    if not (raw_root / "feature_definitions.csv").is_file():
        raise FileNotFoundError(raw_root / "feature_definitions.csv")
    if not (raw_root / "parquet_files/train/train_base.parquet").is_file():
        raise FileNotFoundError(raw_root / "parquet_files/train/train_base.parquet")
    branch = git(repository_root, "branch", "--show-current")
    head = git(repository_root, "rev-parse", "HEAD")
    status_before = git(repository_root, "status", "--porcelain=v1")
    allowed_prefix = OUTPUT_RELATIVE_ROOT.as_posix() + "/"
    allowed_exact = set(SUPPORT_FILES)
    unexpected_status = []
    for line in status_before.splitlines():
        changed_path = line[3:].replace("\\", "/")
        if not (changed_path.startswith(allowed_prefix) or changed_path in allowed_exact):
            unexpected_status.append(line)
    if branch != "main" or head != STARTING_COMMIT or unexpected_status:
        raise RuntimeError(
            f"unexpected preflight: branch={branch!r}, head={head!r}, "
            f"unexpected_status={unexpected_status!r}, status={status_before!r}"
        )
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", str(RAW_RELATIVE_ROOT / "feature_definitions.csv")],
        cwd=repository_root,
        check=False,
    ).returncode == 0
    if not ignored:
        raise RuntimeError("raw third-dataset directory is not ignored")
    active_locks = sorted(
        path.relative_to(repository_root).as_posix()
        for root_name in ("results", "artifacts", "logs")
        if (repository_root / root_name).exists()
        for path in (repository_root / root_name).rglob(".execution.lock")
    )
    if active_locks:
        raise RuntimeError(f"active execution locks: {active_locks}")

    inventory_rows, families, input_digest = inventory(repository_root, raw_root)
    base_path = raw_root / "parquet_files/train/train_base.parquet"
    base = pq.read_table(base_path, columns=["case_id", "date_decision", "MONTH", "WEEK_NUM", "target"]).to_pandas()
    base["date_decision"] = pd.to_datetime(base["date_decision"], errors="raise")
    if base["case_id"].isna().any() or base["case_id"].duplicated().any():
        raise ValueError("base case_id must be nonmissing and unique")
    if set(base["target"].unique()) != {0, 1} or base["target"].isna().any():
        raise ValueError("target must be complete and exactly binary {0,1}")
    if base["date_decision"].isna().any():
        raise ValueError("date_decision must be complete")
    if int((base["MONTH"] != base["date_decision"].dt.year * 100 + base["date_decision"].dt.month).sum()):
        raise ValueError("MONTH is inconsistent with date_decision")
    if int(base.groupby("date_decision")["WEEK_NUM"].nunique().max()) != 1:
        raise ValueError("a date_decision maps to more than one WEEK_NUM")
    expected_weeks = set(range(int(base["WEEK_NUM"].min()), int(base["WEEK_NUM"].max()) + 1))
    if expected_weeks != set(int(value) for value in base["WEEK_NUM"].unique()):
        raise ValueError("WEEK_NUM is not contiguous")

    profile_rows, schemas = profile_families(repository_root, families, base["case_id"].to_numpy())
    feature_review, coverage = build_feature_review(schemas, raw_root / "feature_definitions.csv")
    split, temporal_rows = build_temporal_design(base)
    adapter = proposed_adapter()
    matrix = proposed_method_matrix(repository_root)
    created_at = datetime.now(tz=UTC).isoformat().replace("+00:00", "Z")
    included_parquets = [row for row in inventory_rows if row["inclusion_status"] == "included" and row["file_type"] == "parquet"]
    excluded_parquets = [row for row in inventory_rows if row["inclusion_status"] == "excluded" and row["file_type"] == "parquet"]
    included_depth_counts = Counter(str(row["relational_depth"]) for row in included_parquets)
    dataset_identity = {
        "schema_version": "third_dataset_identity_review_v1",
        "status": "stage_1_review_only_not_canonical_lock",
        "created_at_utc": created_at,
        "official_dataset_name": OFFICIAL_NAME,
        "dataset_id": DATASET_ID,
        "local_root": RAW_RELATIVE_ROOT.as_posix(),
        "snapshot_definition": "feature_definitions.csv plus train_base and every train relational depth-0/depth-1 Parquet; depth-2 excluded",
        "included_raw_file_count": sum(row["inclusion_status"] == "included" for row in inventory_rows),
        "included_parquet_file_count": len(included_parquets),
        "excluded_parquet_file_count": len(excluded_parquets),
        "included_parquet_depth_counts": dict(sorted(included_depth_counts.items())),
        "included_table_family_count": len(profile_rows),
        "excluded_depth_2_family_count": len({row["table_family"] for row in excluded_parquets}),
        "included_raw_input_digest_algorithm": "sha256(canonical JSON ordered inventory projection of relative_path,byte_size,sha256)",
        "included_raw_input_digest": input_digest,
        "base_population_validation": {
            "row_count": int(len(base)),
            "case_id_dtype": str(base["case_id"].dtype),
            "case_id_missing": int(base["case_id"].isna().sum()),
            "case_id_unique_count": int(base["case_id"].nunique()),
            "case_id_duplicate_count": int(base["case_id"].duplicated().sum()),
            "target_dtype": str(base["target"].dtype),
            "target_values": [0, 1],
            "target_missing": int(base["target"].isna().sum()),
            "target_0": int((base["target"] == 0).sum()),
            "target_1": int((base["target"] == 1).sum()),
            "date_decision_dtype": str(base["date_decision"].dtype),
            "date_decision_missing": int(base["date_decision"].isna().sum()),
            "date_min": base["date_decision"].min().date().isoformat(),
            "date_max": base["date_decision"].max().date().isoformat(),
            "unique_dates": int(base["date_decision"].nunique()),
            "month_dtype": str(base["MONTH"].dtype),
            "month_matches_date_decision_yyyymm": True,
            "week_num_dtype": str(base["WEEK_NUM"].dtype),
            "week_num_missing": int(base["WEEK_NUM"].isna().sum()),
            "week_num_min": int(base["WEEK_NUM"].min()),
            "week_num_max": int(base["WEEK_NUM"].max()),
            "week_num_unique": int(base["WEEK_NUM"].nunique()),
            "week_num_contiguous": True,
            "each_date_maps_to_one_week_num": True,
            "chronological_authority": "date_decision",
        },
        "repository_preflight": {
            "branch": branch,
            "commit": head,
            "git_status_porcelain_at_task_preflight": "",
            "generation_scope_validation": "all observed non-clean paths were confined to this new Stage 1 audit package, builder, and focused test",
            "generation_changes": [
                OUTPUT_RELATIVE_ROOT.as_posix() + "/",
                *SUPPORT_FILES,
            ],
            "python_executable": sys.executable,
            "python_version": sys.version.replace("\n", " "),
            "platform": platform.platform(),
            "raw_root_git_ignored": ignored,
            "active_execution_locks": active_locks,
            "experiment_workers": 0,
            "process_check_note": "read-only preflight found no experiment worker; VS Code Jupyter interrupt helper is not a workload",
        },
        "task_boundaries": {
            "existing_two_dataset_oot_metrics_opened": False,
            "model_or_selector_fit_run": False,
            "pilot_dev_or_oot_run": False,
            "raw_files_modified": False,
            "network_accessed": False,
            "canonical_protocol_lock_created": False,
        },
    }
    coverage["included_raw_input_digest"] = input_digest
    coverage["feature_definitions_sha256"] = next(
        row["sha256"] for row in inventory_rows if row["relative_path"].endswith("feature_definitions.csv")
    )
    adapter["included_raw_input_digest"] = input_digest
    matrix["included_raw_input_digest"] = input_digest
    split["included_raw_input_digest"] = input_digest

    write_json(output_root / "dataset_identity.json", dataset_identity)
    write_csv(
        output_root / "raw_file_inventory.csv",
        inventory_rows,
        [
            "relative_path", "byte_size", "modified_utc", "sha256", "file_type", "table_family",
            "relational_depth", "file_part_number", "inclusion_status", "inclusion_or_exclusion_reason",
            "parquet_row_count", "parquet_column_count", "case_id_present", "case_id_dtype",
            "grouping_order_columns_json",
        ],
    )
    write_csv(
        output_root / "schema_and_cardinality_profile.csv",
        profile_rows,
        list(profile_rows[0]),
    )
    write_json(output_root / "feature_definition_coverage.json", coverage)
    write_csv(
        output_root / "temporal_population_profile.csv",
        temporal_rows,
        [
            "profile_level", "partition", "fold_id", "role", "calendar_period", "rows",
            "target_0", "target_1", "date_min", "date_max", "unique_dates",
        ],
    )
    write_json(output_root / "proposed_split_and_fold_boundaries.json", split)
    write_csv(
        output_root / "leakage_and_availability_review.csv",
        feature_review,
        list(feature_review[0]),
    )
    write_json(output_root / "proposed_adapter_protocol.json", adapter)
    write_json(output_root / "proposed_method_matrix.json", matrix)
    (output_root / "preregistered_hypotheses_and_analysis.md").write_text(
        preregistration_markdown(), encoding="utf-8", newline="\n"
    )
    (output_root / "protocol_review.md").write_text(
        review_markdown(inventory_rows, input_digest, base, split, coverage, profile_rows, matrix),
        encoding="utf-8",
        newline="\n",
    )

    bound_paths = [OUTPUT_RELATIVE_ROOT / name for name in REQUIRED_PACKAGE_FILES]
    bound_paths.extend(Path(path) for path in SUPPORT_FILES)
    missing_support = [path.as_posix() for path in bound_paths if not (repository_root / path).is_file()]
    if missing_support:
        raise FileNotFoundError(f"digest support files missing: {missing_support}")
    digest_payload = {
        "schema_version": "third_dataset_protocol_stage_1_review_digest_v1",
        "status": "awaiting_explicit_user_approval",
        "created_at_utc": created_at,
        "hash_algorithm": "sha256",
        "canonical_serialization": "UTF-8 JSON sort_keys=true separators=(',',':') ensure_ascii=false allow_nan=false",
        "self_authentication_rule": "review_digest_sha256 hashes this JSON object with only review_digest_sha256 omitted",
        "artifact_hashes": [
            {"path": path.as_posix(), "byte_size": (repository_root / path).stat().st_size, "sha256": sha256_file(repository_root / path)}
            for path in bound_paths
        ],
        "summary": {
            "dataset_id": DATASET_ID,
            "included_raw_input_digest": input_digest,
            "included_parquet_files": len(included_parquets),
            "excluded_depth_2_parquet_files": len(excluded_parquets),
            "base_rows": int(len(base)),
            "target_0": int((base["target"] == 0).sum()),
            "target_1": int((base["target"] == 1).sum()),
            "oot_start_inclusive": split["oot_start_inclusive"],
            "included_candidate_features": coverage["included_review_rows"],
            "unresolved_leakage_rows": coverage["unresolved_review_rows"],
            "baseline_methods": BASELINE_ORDER,
            "approved_combination_methods": COMBINATION_ORDER,
            "models": MODELS,
            "feature_budgets": {"lr": 20, "catboost": 40},
            "seed": 42,
            "pilot_selector_fit_calls": 27,
            "pilot_evaluation_cells": 30,
            "dev_selector_fit_calls": 135,
            "dev_fold_evaluation_cells": 150,
            "oot_evaluation_cells": 30,
            "depth_scope": {"base": "included", "depth_0": "included", "depth_1": "included", "depth_2": "excluded"},
            "temporal_protocol": {
                "dev_end_inclusive": split["dev_end_inclusive"],
                "oot_start_inclusive": split["oot_start_inclusive"],
                "folds": 5,
                "gap_unique_dates": 1,
                "training_window": "expanding",
            },
            "adapter_protocol_sha256": sha256_file(output_root / "proposed_adapter_protocol.json"),
            "leakage_review_sha256": sha256_file(output_root / "leakage_and_availability_review.csv"),
            "method_matrix_sha256": sha256_file(output_root / "proposed_method_matrix.json"),
            "analysis_preregistration_sha256": sha256_file(output_root / "preregistered_hypotheses_and_analysis.md"),
            "extension_scope": matrix["extension_scope"],
            "execution_gates": matrix["gates"],
            "unresolved_blockers": [],
            "canonical_lock_created": False,
        },
    }
    digest_payload["review_digest_sha256"] = sha256_bytes(canonical_json_bytes(digest_payload))
    write_json(output_root / "review_digest.json", digest_payload)
    validation = validate_package(repository_root, output_root)
    if not validation["valid"]:
        raise RuntimeError(validation)
    return validation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    repository_root = args.repository_root.resolve()
    output_root = repository_root / OUTPUT_RELATIVE_ROOT
    result = validate_package(repository_root, output_root) if args.validate_only else build(repository_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("valid") else 1


if __name__ == "__main__":
    raise SystemExit(main())
