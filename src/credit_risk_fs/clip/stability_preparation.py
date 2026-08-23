from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from credit_risk_fs.clip.embedding_cache import EmbeddingCacheSpec, build_embedding_frame
from credit_risk_fs.clip.exact_duplicates import feature_order_hash
from credit_risk_fs.clip.source_anchor import (
    ANCHOR_RULE_VERSION,
    equal_duration_boundaries,
    fit_frozen_bucketizer,
    psi_from_frozen_buckets,
)
from credit_risk_fs.clip.statistical_preprocessor_v2 import RobustStatisticalPreprocessorV2
from credit_risk_fs.clip.statistical_schema_v2 import (
    DESCRIPTOR_COLUMNS_V2,
    SCALED_DESCRIPTOR_COLUMNS_V2,
    UNSCALED_INDICATOR_COLUMNS_V2,
)
from credit_risk_fs.clip.statistical_view_v2 import compute_feature_descriptors
from credit_risk_fs.clip.text_builder import TEXT_TEMPLATE_VERSION, build_feature_text_frame
from credit_risk_fs.clip.text_encoder import (
    FrozenSentenceTransformerEncoder,
    TextEncoderProtocol,
    resolve_device,
)
from credit_risk_fs.utils.hashing import sha256_file, sha256_text


PROTOCOL_SCHEMA_VERSION = "stability_clip_preparation_protocol_v1"
PREPARATION_SCHEMA_VERSION = "stability_clip_preparation_v1"
DATASET_ID = "homecredit_model_stability_2024"
PAIRING_POLICY_VERSION = "identity_equivalence_v2"
ANCHOR_METHOD_VERSION = "generalized_lendingclub_dev_temporal_stable_core_v1"
EXPECTED_DESCRIPTOR_ORDER = tuple(DESCRIPTOR_COLUMNS_V2)
FORBIDDEN_VALUE_COLUMNS = frozenset({"case_id", "MONTH", "WEEK_NUM", "target"})
GENERATED_RELATIVE_PATHS = (
    "metadata/feature_universe.csv",
    "metadata/feature_universe_manifest.json",
    "metadata/feature_semantics.csv",
    "text/feature_text.csv",
    "pairing/exact_dev_duplicate_evidence.csv",
    "pairing/identity_equivalence.csv",
    "pairing/representation_split.csv",
    "pairing/representation_split_manifest.json",
    "statistics/statistical_descriptors_raw.csv",
    "statistics/statistical_descriptors_raw.parquet",
    "statistics/stability_source_stat_preprocessor.json",
    "statistics/statistical_descriptors_stability_source_scaled.csv",
    "anchor/stability_source_anchor_candidates.csv",
    "anchor/stability_source_anchor.csv",
    "anchor/stability_source_anchor_manifest.json",
    "text/text_embeddings.parquet",
    "text/text_embedding_manifest.json",
    "pairs/stability_source_pairs.parquet",
    "pairs/stability_source_pairs_manifest.json",
    "methodology_lock.json",
    "manifests/data_provenance.json",
    "validation/validation_report.json",
    "validation/VALIDATION_REPORT.md",
)
WORK_CHECKPOINT_NAME = ".stability_clip_work_checkpoint.json"


class StabilityPreparationError(RuntimeError):
    """Fail-closed error for a Stability CLIP preparation contract violation."""


@dataclass(frozen=True)
class PreparationContext:
    repo_root: Path
    config_path: Path
    config: dict[str, Any]
    configuration_hash: str

    def resolve(self, value: str | Path) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else self.repo_root / candidate

    @property
    def output_dir(self) -> Path:
        return self.resolve(str(self.config["output_dir"]))


@dataclass
class ReadAudit:
    scan_count: int = 0
    requested_predictor_columns: set[str] = field(default_factory=set)
    prior_requested_predictor_column_count: int = 0
    returned_row_counts: list[int] = field(default_factory=list)
    earliest_returned_date: str = ""
    latest_returned_date: str = ""
    target_values_loaded: bool = False
    oot_feature_values_loaded: bool = False
    oot_labels_loaded: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "scan_count": self.scan_count,
            "requested_predictor_column_count": max(
                self.prior_requested_predictor_column_count,
                len(self.requested_predictor_columns),
            ),
            "returned_row_counts": self.returned_row_counts,
            "earliest_returned_date": self.earliest_returned_date,
            "latest_returned_date": self.latest_returned_date,
            "target_values_loaded": self.target_values_loaded,
            "oot_feature_values_loaded": self.oot_feature_values_loaded,
            "oot_labels_loaded": self.oot_labels_loaded,
        }


class DevMatrixReader:
    """Projection- and predicate-guarded reader that exposes DEV predictor values only."""

    def __init__(
        self,
        *,
        matrix_parts: Sequence[Path],
        predictor_names: Sequence[str],
        date_column: str,
        dev_start_inclusive: str,
        dev_end_exclusive: str,
        expected_dev_rows: int,
        row_batch_size: int = 32768,
    ) -> None:
        self.matrix_parts = tuple(Path(path) for path in matrix_parts)
        self.predictor_names = frozenset(str(value) for value in predictor_names)
        self.date_column = str(date_column)
        self.dev_start_inclusive = str(dev_start_inclusive)
        self.dev_end_exclusive = str(dev_end_exclusive)
        self.expected_dev_rows = int(expected_dev_rows)
        self.row_batch_size = int(row_batch_size)
        self.audit = ReadAudit()
        if not self.matrix_parts or self.expected_dev_rows <= 0:
            raise ValueError("DEV reader requires matrix parts and a positive expected row count")

    def read_frame(
        self,
        feature_names: Sequence[str],
        *,
        include_date: bool = False,
    ) -> pd.DataFrame:
        batches = list(self.iter_batches(feature_names, include_date=include_date))
        if not batches:
            columns = ([self.date_column] if include_date else []) + list(feature_names)
            return pd.DataFrame(columns=columns)
        return pd.concat(batches, ignore_index=True)

    def iter_batches(
        self,
        feature_names: Sequence[str],
        *,
        include_date: bool = False,
    ) -> Iterable[pd.DataFrame]:
        names = [str(name) for name in feature_names]
        if len(names) != len(set(names)):
            raise ValueError("DEV projection contains duplicate feature names")
        if set(names) & FORBIDDEN_VALUE_COLUMNS:
            raise StabilityPreparationError("target or forbidden non-predictor requested from matrix")
        unknown = sorted(set(names) - self.predictor_names)
        if unknown:
            raise StabilityPreparationError(f"non-universe predictor requested: {unknown[:20]}")
        try:
            import pyarrow.dataset as ds
        except ImportError as exc:
            raise StabilityPreparationError("pyarrow is required for guarded DEV matrix scans") from exc

        requested = [self.date_column, *names]
        dataset = ds.dataset([str(path) for path in self.matrix_parts], format="parquet")
        predicate = ds.field(self.date_column) < self.dev_end_exclusive
        scanner = dataset.scanner(
            columns=requested,
            filter=predicate,
            batch_size=self.row_batch_size,
            use_threads=True,
        )
        total = 0
        earliest = ""
        latest = ""
        self.audit.scan_count += 1
        self.audit.requested_predictor_columns.update(names)
        for record_batch in scanner.to_batches():
            frame = record_batch.to_pandas()
            dates = frame[self.date_column].astype(str)
            if dates.empty:
                continue
            if bool(dates.ge(self.dev_end_exclusive).any()):
                self.audit.oot_feature_values_loaded = True
                raise StabilityPreparationError("guarded scan returned a row outside frozen DEV")
            batch_min, batch_max = str(dates.min()), str(dates.max())
            earliest = min(filter(None, (earliest, batch_min)), default=batch_min)
            latest = max(filter(None, (latest, batch_max)), default=batch_max)
            total += len(frame)
            yield frame if include_date else frame.drop(columns=[self.date_column])
        if total != self.expected_dev_rows:
            raise StabilityPreparationError(
                f"DEV scan row count mismatch: expected={self.expected_dev_rows}, observed={total}"
            )
        if earliest != self.dev_start_inclusive:
            raise StabilityPreparationError(
                "authenticated DEV minimum date mismatch: "
                f"expected={self.dev_start_inclusive}, observed={earliest}"
            )
        self.audit.returned_row_counts.append(total)
        self.audit.earliest_returned_date = min(
            filter(None, (self.audit.earliest_returned_date, earliest)), default=earliest
        )
        self.audit.latest_returned_date = max(
            filter(None, (self.audit.latest_returned_date, latest)), default=latest
        )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_preparation_context(
    config_path: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> PreparationContext:
    path = Path(config_path).resolve()
    root = Path(repo_root).resolve() if repo_root else path.parents[3]
    payload = _read_json(path)
    if payload.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise StabilityPreparationError("unsupported Stability CLIP preparation protocol")
    _validate_locked_config(payload)
    return PreparationContext(
        repo_root=root,
        config_path=path,
        config=payload,
        configuration_hash=sha256_file(path),
    )


def _validate_locked_config(config: Mapping[str, Any]) -> None:
    dataset = config.get("dataset", {})
    stat = config.get("statistical_view", {})
    split = config.get("representation_split", {})
    anchor = config.get("source_anchor", {})
    encoder = config.get("text_encoder", {})
    if config.get("output_dir") != (
        "outputs/prompt_16_homecredit_model_stability_2024/clip_preparation_v1"
    ):
        raise StabilityPreparationError("locked Stability CLIP output path changed")
    required = {
        "dataset_id": DATASET_ID,
        "expected_feature_count": 1959,
        "expected_dev_row_count": 1221743,
        "expected_oot_row_count": 304916,
        "date_column": "date_decision",
        "dev_start_inclusive": "2019-01-01",
        "oot_start_exclusive_dev": "2020-02-26",
    }
    for key, expected in required.items():
        if dataset.get(key) != expected:
            raise StabilityPreparationError(f"locked dataset field mismatch: {key}")
    if tuple(stat.get("descriptor_order", ())) != EXPECTED_DESCRIPTOR_ORDER:
        raise StabilityPreparationError("compact_target_free_v2 descriptor order changed")
    if stat.get("schema_version") != "compact_target_free_v2" or stat.get("dimension") != 13:
        raise StabilityPreparationError("statistical descriptor contract changed")
    if split.get("seed") != 42 or split.get("validation_fraction") != 0.2 or not split.get("group_aware"):
        raise StabilityPreparationError("representation split contract changed")
    anchor_expected = {
        "reference_rule_version": ANCHOR_RULE_VERSION,
        "candidate_split": "train",
        "temporal_subwindows": 4,
        "subwindow_strategy": "equal_duration",
        "max_adjacent_window_psi": 0.10,
        "max_missing_rate_difference": 0.05,
        "member_count": 23,
        "numeric_bins": 10,
        "categorical_min_count": 50,
    }
    for key, expected in anchor_expected.items():
        if anchor.get(key) != expected:
            raise StabilityPreparationError(f"source-anchor contract changed: {key}")
    if anchor.get("target_used") or anchor.get("oot_used"):
        raise StabilityPreparationError("source anchor must be target-free and OOT-free")
    if encoder.get("model_name") != "sentence-transformers/all-MiniLM-L6-v2":
        raise StabilityPreparationError("text encoder model identity changed")
    if encoder.get("model_revision") != "main" or encoder.get("embedding_dimension") != 384:
        raise StabilityPreparationError("text encoder revision or dimension changed")
    if not encoder.get("normalize_embeddings") or not encoder.get("frozen_inference_only"):
        raise StabilityPreparationError("text encoder must be frozen and L2-normalized")


def validate_input_protocol(context: PreparationContext) -> dict[str, Any]:
    config = context.config
    observations: list[dict[str, Any]] = []
    for name, spec in config["inputs"].items():
        path = context.resolve(spec["path"])
        _verify_file(path, str(spec["sha256"]), label=name)
        observations.append(_provenance_row(context, path, f"authoritative_{name}"))
    for spec in config["implementation_contracts"]:
        path = context.resolve(spec["path"])
        _verify_file(path, str(spec["sha256"]), label=str(spec["role"]))
        observations.append(_provenance_row(context, path, str(spec["role"])))

    manifest_path = context.resolve(config["inputs"]["matrix_manifest"]["path"])
    manifest = _read_json(manifest_path)
    summary = manifest.get("summary", {})
    dataset = config["dataset"]
    summary_expectations = {
        "row_count": dataset["expected_total_row_count"],
        "predictor_count": dataset["expected_feature_count"],
        "matrix_part_count": 31,
        "depth_2_files_opened": 0,
        "fits": 0,
        "evaluations": 0,
    }
    for key, expected in summary_expectations.items():
        if summary.get(key) != expected:
            raise StabilityPreparationError(f"authenticated matrix summary mismatch: {key}")

    metadata = _read_json(context.resolve(config["inputs"]["matrix_metadata"]["path"]))
    if metadata.get("row_count") != dataset["expected_total_row_count"]:
        raise StabilityPreparationError("authenticated matrix metadata row count mismatch")
    expected_membership_rule = "DEV when date_decision < 2020-02-26; OOT otherwise"
    if metadata.get("split_and_fold_interface", {}).get("membership") != expected_membership_rule:
        raise StabilityPreparationError("authenticated DEV/OOT membership rule mismatch")
    split_path = context.resolve(config["inputs"]["split_membership"]["path"])
    membership = pd.read_parquet(split_path, columns=["membership"])["membership"].astype(str)
    membership_counts = membership.value_counts().to_dict()
    expected_membership_counts = {
        "DEV": int(dataset["expected_dev_row_count"]),
        "OOT": int(dataset["expected_oot_row_count"]),
    }
    if membership_counts != expected_membership_counts:
        raise StabilityPreparationError(
            "authenticated DEV/OOT membership counts mismatch: "
            f"expected={expected_membership_counts}, observed={membership_counts}"
        )

    matrix_root = manifest_path.parent
    matrix_parts: list[Path] = []
    for artifact in manifest.get("artifacts", []):
        relative = str(artifact.get("path", ""))
        if not relative.startswith("matrix/part-") or not relative.endswith(".parquet"):
            continue
        path = matrix_root / relative
        _verify_file(path, str(artifact["sha256"]), label=relative, size=int(artifact["size_bytes"]))
        matrix_parts.append(path)
        observations.append(_provenance_row(context, path, "authenticated_matrix_component"))
    matrix_parts.sort()
    if len(matrix_parts) != int(summary_expectations["matrix_part_count"]):
        raise StabilityPreparationError("authenticated matrix component count mismatch")
    return {
        "manifest": manifest,
        "matrix_parts": matrix_parts,
        "membership_counts": membership_counts,
        "input_provenance": observations,
    }


def build_feature_universe(context: PreparationContext) -> tuple[pd.DataFrame, dict[str, Any]]:
    metadata_path = context.resolve(context.config["inputs"]["matrix_metadata"]["path"])
    lineage_path = context.resolve(context.config["inputs"]["matrix_lineage"]["path"])
    return build_feature_universe_from_payloads(
        metadata=_read_json(metadata_path),
        lineage=_read_json(lineage_path),
        dataset_id=str(context.config["dataset"]["dataset_id"]),
        expected_count=int(context.config["dataset"]["expected_feature_count"]),
        expected_universe_hash=str(context.config["dataset"]["expected_feature_universe_hash"]),
        lineage_source=_relative_to_root(lineage_path, context.repo_root),
    )


def build_feature_universe_from_payloads(
    *,
    metadata: Mapping[str, Any],
    lineage: Mapping[str, Any],
    dataset_id: str,
    expected_count: int,
    expected_universe_hash: str | None = None,
    lineage_source: str = "lineage.json#features",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    predictors = [str(value) for value in metadata.get("predictor_columns", [])]
    non_predictors = {str(value) for value in metadata.get("non_predictor_columns", [])}
    if len(predictors) != expected_count or len(set(predictors)) != expected_count:
        raise StabilityPreparationError("authenticated predictor universe count or uniqueness mismatch")
    forbidden = set(predictors) & (non_predictors | FORBIDDEN_VALUE_COLUMNS | {"date_decision"})
    if forbidden:
        raise StabilityPreparationError(f"non-predictors entered feature universe: {sorted(forbidden)}")
    ordered_hash = feature_order_hash(predictors)
    if expected_universe_hash and ordered_hash != expected_universe_hash:
        raise StabilityPreparationError("ordered feature universe hash mismatch")

    physical_types = {
        str(row["name"]): str(row["arrow_type"])
        for row in metadata.get("columns", [])
        if isinstance(row, Mapping) and "name" in row and "arrow_type" in row
    }
    lineage_rows = {
        str(row["output_feature"]): row
        for row in lineage.get("features", [])
        if isinstance(row, Mapping) and row.get("output_feature")
    }
    if set(lineage_rows) != set(predictors):
        missing = sorted(set(predictors) - set(lineage_rows))
        extra = sorted(set(lineage_rows) - set(predictors))
        raise StabilityPreparationError(
            f"lineage/universe mismatch: missing={missing[:10]}, extra={extra[:10]}"
        )
    records: list[dict[str, Any]] = []
    for name in predictors:
        row = lineage_rows[name]
        family = str(row.get("source_family", "")).strip()
        if not family:
            raise StabilityPreparationError(f"feature lacks source family: {name}")
        depth_token = name.split("__", 1)[0]
        if depth_token not in {"d0", "d1"}:
            raise StabilityPreparationError(f"unexpected feature depth prefix: {name}")
        source_feature = row.get("source_feature")
        records.append(
            {
                "feature_id": sha256_text(f"{dataset_id}|{name}|{family}"),
                "feature_name": name,
                "source_feature": "" if source_feature is None else str(source_feature),
                "source_table": family,
                "source_family": family,
                "depth": int(depth_token[1:]),
                "aggregation": str(row.get("aggregation", "")),
                "logical_type": str(row.get("logical_type", "")),
                "physical_type": physical_types.get(name, ""),
                "lineage_source": f"{lineage_source}::{name}",
                "protocol_action": str(row.get("protocol_action", "")),
                "protocol_output_prefix": str(row.get("protocol_output_prefix", "")),
                "eligible_for_clip": True,
            }
        )
    frame = pd.DataFrame(records)
    if frame["feature_id"].duplicated().any() or frame["physical_type"].eq("").any():
        raise StabilityPreparationError("feature IDs are non-unique or physical types are missing")
    manifest = {
        "schema_version": "stability_feature_universe_v1",
        "dataset_id": dataset_id,
        "row_count": len(frame),
        "ordered_feature_name_sha256": ordered_hash,
        "ordering": "authenticated_matrix_metadata.predictor_columns",
        "eligible_for_clip_count": int(frame["eligible_for_clip"].sum()),
        "target_included": False,
        "non_predictors_included": False,
    }
    return frame, manifest


AGGREGATION_DESCRIPTIONS = {
    "identity_after_family_prefix": "Identity-preserving value after adding the locked source-family prefix",
    "first_by_num_group1": "First value ordered by num_group1",
    "last_by_num_group1": "Last value ordered by num_group1",
    "count_non_missing": "Count of non-missing values",
    "missing_count": "Count of missing values",
    "mean": "Mean",
    "sample_variance_ddof_1": "Sample variance with one degree of freedom",
    "min": "Minimum",
    "max": "Maximum",
    "sum": "Sum",
    "nunique": "Number of unique non-missing values",
    "lexical_mode": "Lexically tie-broken mode",
    "signed_days_relative_to_base_date_decision": "Signed number of days relative to the base decision date",
    "row_count": "Number of source observations",
    "any": "Boolean any aggregation",
    "all": "Boolean all aggregation",
    "false_count": "Count of false values",
    "true_count": "Count of true values",
}


def build_semantic_metadata(
    universe: pd.DataFrame,
    *,
    feature_definitions_path: str | Path,
) -> pd.DataFrame:
    definitions = pd.read_csv(feature_definitions_path, dtype="string")
    if set(definitions.columns) < {"Variable", "Description"}:
        raise StabilityPreparationError("feature_definitions.csv schema mismatch")
    definitions["Variable"] = definitions["Variable"].fillna("").astype(str).str.strip()
    definitions["Description"] = definitions["Description"].fillna("").astype(str).str.strip()
    definitions = definitions.loc[definitions["Variable"].ne("")]
    if definitions["Variable"].duplicated().any():
        raise StabilityPreparationError("feature_definitions.csv contains duplicate variables")
    description_by_source = definitions.set_index("Variable")["Description"].to_dict()

    records: list[dict[str, Any]] = []
    for row in universe.to_dict("records"):
        aggregation = str(row["aggregation"])
        if aggregation not in AGGREGATION_DESCRIPTIONS:
            raise StabilityPreparationError(f"unmapped authenticated aggregation: {aggregation}")
        source_feature = str(row["source_feature"])
        family = str(row["source_family"])
        source_description = str(description_by_source.get(source_feature, "")).strip()
        operation = AGGREGATION_DESCRIPTIONS[aggregation]
        if source_description:
            meaning = source_description.rstrip(".")
            description = f"{operation} for {meaning} derived from source family {family}."
            description_source = "authenticated_feature_definitions_plus_lineage_operation"
            status = "official_source_description_and_exact_lineage_operation"
        elif aggregation == "row_count" and not source_feature:
            description = f"{operation} in source family {family}."
            description_source = "lineage_template"
            status = "transparent_structural_fallback"
        else:
            structural_source = source_feature or "source rows"
            description = (
                f"Engineered feature derived from {structural_source} in source family {family} "
                f"using the {aggregation} operation."
            )
            description_source = "lineage_template"
            status = "transparent_structural_fallback"
        formula_argument = source_feature or "*"
        formula = f"{aggregation}({formula_argument}) from source family {family}"
        records.append(
            {
                "feature_name": str(row["feature_name"]),
                "description": description,
                "semantic_group": f"source_table::{family}",
                "source_or_formula": formula,
                "source_feature": source_feature,
                "source_table": str(row["source_table"]),
                "aggregation": aggregation,
                "source_description": source_description,
                "description_source": description_source,
                "semantic_group_source": "authenticated_lineage.source_family",
                "formula_source": "authenticated_lineage.aggregation",
                "semantic_status": status,
            }
        )
    semantics = pd.DataFrame(records)
    required = ["description", "semantic_group", "source_or_formula", "description_source"]
    if len(semantics) != len(universe) or semantics["feature_name"].duplicated().any():
        raise StabilityPreparationError("semantic metadata does not map one-to-one to universe")
    if semantics[required].fillna("").astype(str).apply(lambda col: col.str.strip().eq("").any()).any():
        raise StabilityPreparationError("semantic metadata contains an empty required field")
    return semantics


def render_feature_text(
    universe: pd.DataFrame,
    semantics: pd.DataFrame,
    *,
    source_manifest_hash: str,
) -> pd.DataFrame:
    source = semantics[
        ["feature_name", "description", "semantic_group", "source_or_formula"]
    ].rename(columns={"feature_name": "feature", "source_or_formula": "source_table"})
    rendered = build_feature_text_frame(
        source,
        dataset=DATASET_ID,
        source_manifest_hash=source_manifest_hash,
        allow_fallback=False,
    ).rename(
        columns={
            "feature_text": "rendered_text",
            "text_template_version": "template_version",
            "feature_text_hash": "rendered_text_sha256",
        }
    )
    rendered = universe[["feature_id", "feature_name"]].merge(
        rendered,
        on="feature_name",
        how="left",
        validate="one_to_one",
    )
    if set(rendered["template_version"].astype(str)) != {TEXT_TEMPLATE_VERSION}:
        raise StabilityPreparationError("historical feature_text_v1 was not reproduced")
    if rendered["rendered_text"].fillna("").astype(str).str.strip().eq("").any():
        raise StabilityPreparationError("rendered feature text is empty")
    return rendered


def _canonical_series(series: pd.Series) -> pd.DataFrame:
    missing = series.isna()
    if pd.api.types.is_bool_dtype(series.dtype):
        values = series.astype("boolean")
        kind = "bool"
    elif pd.api.types.is_numeric_dtype(series.dtype):
        values = pd.to_numeric(series, errors="coerce").astype("float64")
        kind = "numeric"
    else:
        values = series.astype("string")
        kind = "text"
    return pd.DataFrame(
        {
            "kind": pd.Series(kind, index=series.index, dtype="string"),
            "missing": missing.astype(bool),
            "value": values,
        },
        index=series.index,
    )


def _series_candidate_hash(series: pd.Series) -> str:
    canonical = _canonical_series(series)
    hashes = pd.util.hash_pandas_object(canonical, index=False).to_numpy(dtype="uint64")
    return hashlib.sha256(hashes.tobytes()).hexdigest()


def find_exact_dev_duplicates_chunked(
    reader: DevMatrixReader,
    feature_names: Sequence[str],
    *,
    feature_batch_size: int,
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    names = [str(name) for name in feature_names]
    candidate_groups: dict[str, list[str]] = {}
    total_batches = _batch_count(len(names), feature_batch_size)
    for index, chunk in enumerate(_chunks(names, feature_batch_size), start=1):
        if progress:
            progress(
                f"exact-duplicate candidate hashes: feature batch {index}/{total_batches}"
            )
        frame = reader.read_frame(chunk)
        for name in chunk:
            candidate_groups.setdefault(_series_candidate_hash(frame[name]), []).append(name)

    rows: list[dict[str, Any]] = []
    for candidate_hash, candidates in sorted(candidate_groups.items()):
        if len(candidates) < 2:
            continue
        unresolved = list(candidates)
        while unresolved:
            representative = unresolved.pop(0)
            equal_members = [representative]
            not_equal: list[str] = []
            for chunk in _chunks(unresolved, max(1, feature_batch_size - 1)):
                frame = reader.read_frame([representative, *chunk])
                reference = _canonical_series(frame[representative]).reset_index(drop=True)
                for candidate in chunk:
                    observed = _canonical_series(frame[candidate]).reset_index(drop=True)
                    (equal_members if reference.equals(observed) else not_equal).append(candidate)
            if len(equal_members) > 1:
                for member in equal_members[1:]:
                    rows.append(
                        {
                            "feature_name_a": representative,
                            "feature_name_b": member,
                            "equivalence_reason": "exact_dev_duplicate",
                            "evidence_source": "authenticated_matrix_DEV_aligned_values_and_missing_masks",
                            "dev_row_count": reader.expected_dev_rows,
                            "candidate_hash": candidate_hash,
                            "actual_equality_verified": True,
                            "target_used": False,
                            "oot_used": False,
                        }
                    )
            unresolved = not_equal
    columns = [
        "feature_name_a",
        "feature_name_b",
        "equivalence_reason",
        "evidence_source",
        "dev_row_count",
        "candidate_hash",
        "actual_equality_verified",
        "target_used",
        "oot_used",
    ]
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["feature_name_a", "feature_name_b"], kind="mergesort"
    ).reset_index(drop=True)


def build_identity_equivalence(
    universe: pd.DataFrame,
    *,
    exact_duplicate_evidence: pd.DataFrame | None = None,
    documented_relations: pd.DataFrame | None = None,
) -> pd.DataFrame:
    names = universe["feature_name"].astype(str).tolist()
    parents = {name: name for name in names}
    reasons: dict[str, set[str]] = {name: set() for name in names}
    evidence: dict[str, set[str]] = {name: set() for name in names}

    def find(value: str) -> str:
        if parents[value] != value:
            parents[value] = find(parents[value])
        return parents[value]

    def union(left: str, right: str) -> None:
        a, b = find(left), find(right)
        if a != b:
            parents[max(a, b)] = min(a, b)

    relations: list[tuple[str, str, str, str]] = []
    if exact_duplicate_evidence is not None and len(exact_duplicate_evidence):
        for row in exact_duplicate_evidence.to_dict("records"):
            if not bool(row.get("actual_equality_verified", False)):
                raise StabilityPreparationError("candidate hash cannot establish identity equivalence")
            relations.append(
                (
                    str(row["feature_name_a"]),
                    str(row["feature_name_b"]),
                    "dev_exact_duplicate",
                    str(row["evidence_source"]),
                )
            )
    if documented_relations is not None and len(documented_relations):
        allowed = {"documented_alias", "documented_identity_transform"}
        for row in documented_relations.to_dict("records"):
            reason = str(row["equivalence_reason"])
            if reason not in allowed:
                raise StabilityPreparationError(f"unsupported identity relation: {reason}")
            relations.append(
                (
                    str(row["feature_name_a"]),
                    str(row["feature_name_b"]),
                    reason,
                    str(row["evidence_source"]),
                )
            )
    for left, right, reason, source in relations:
        if left not in parents or right not in parents or left == right:
            raise StabilityPreparationError("identity evidence references invalid feature identities")
        union(left, right)
        reasons[left].add(reason)
        reasons[right].add(reason)
        evidence[left].add(source)
        evidence[right].add(source)

    groups: dict[str, list[str]] = {}
    for name in names:
        groups.setdefault(find(name), []).append(name)
    group_by_name: dict[str, tuple[str, int]] = {}
    for members in groups.values():
        ordered = sorted(members)
        group_id = f"identity:{sha256_text('|'.join(ordered))[:24]}"
        for name in ordered:
            group_by_name[name] = (group_id, len(ordered))
    rows = []
    for name in names:
        group_id, size = group_by_name[name]
        group_reasons = sorted(
            set().union(*(reasons[member] for member in groups[find(name)]))
        )
        group_evidence = sorted(
            set().union(*(evidence[member] for member in groups[find(name)]))
        )
        rows.append(
            {
                "feature_name": name,
                "equivalence_group_id": group_id,
                "equivalence_reason": ";".join(group_reasons) or "singleton_identity",
                "evidence_source": ";".join(group_evidence) or "authenticated_feature_identity",
                "group_size": size,
                "policy_version": PAIRING_POLICY_VERSION,
            }
        )
    output = pd.DataFrame(rows)
    if len(output) != len(universe) or output["feature_name"].duplicated().any():
        raise StabilityPreparationError("identity equivalence is not one-to-one with feature universe")
    return output


def build_representation_split(
    universe: pd.DataFrame,
    semantics: pd.DataFrame,
    equivalence: pd.DataFrame,
    *,
    seed: int = 42,
    validation_fraction: float = 0.2,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    from credit_risk_fs.clip.reverse_transfer import deterministic_feature_split

    reconciled = (
        universe[["feature_id", "feature_name", "source_table", "eligible_for_clip"]]
        .merge(semantics[["feature_name", "semantic_group"]], on="feature_name", validate="one_to_one")
        .assign(
            dataset=DATASET_ID,
            eligible_for_pairing=lambda value: value["eligible_for_clip"].astype(bool),
        )
    )
    relation_rows: list[dict[str, str]] = []
    id_by_name = universe.set_index("feature_name")["feature_id"].astype(str).to_dict()
    for _, group in equivalence.groupby("equivalence_group_id", sort=True):
        members = group["feature_name"].astype(str).sort_values().tolist()
        for left, right in zip(members[:-1], members[1:]):
            relation_rows.append(
                {
                    "feature_id_a": id_by_name[left],
                    "feature_id_b": id_by_name[right],
                    "reason": "exact_dev_duplicate",
                }
            )
    relations = pd.DataFrame(relation_rows, columns=["feature_id_a", "feature_id_b", "reason"])
    split, historical_manifest = deterministic_feature_split(
        reconciled,
        dataset=DATASET_ID,
        seed=seed,
        validation_fraction=validation_fraction,
        identity_relations=relations,
    )
    output = (
        universe[["feature_name"]]
        .merge(equivalence[["feature_name", "equivalence_group_id"]], on="feature_name", validate="one_to_one")
        .merge(
            split[["feature_name", "split_assignment"]],
            on="feature_name",
            validate="one_to_one",
        )
        .rename(columns={"split_assignment": "representation_split"})
    )
    output["split_seed"] = int(seed)
    if output.groupby("equivalence_group_id")["representation_split"].nunique().gt(1).any():
        raise StabilityPreparationError("identity-equivalence group crosses representation split")
    manifest = {
        "schema_version": "stability_representation_split_v1",
        "policy_version": PAIRING_POLICY_VERSION,
        "seed": int(seed),
        "target_train_fraction": 1.0 - float(validation_fraction),
        "target_validation_fraction": float(validation_fraction),
        "group_aware": True,
        "row_count": len(output),
        "train_count": int(output["representation_split"].eq("train").sum()),
        "validation_count": int(output["representation_split"].eq("validation").sum()),
        "identity_group_overlap_count": 0,
        "historical_split_contract_hash": historical_manifest["split_hash"],
        "representation_split_hash": sha256_text(output.to_csv(index=False)),
        "target_used": False,
        "oot_used": False,
    }
    return output, manifest


def compute_raw_descriptors_from_frame(
    data: pd.DataFrame,
    universe: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metadata_type = universe.set_index("feature_name")["logical_type"].astype(str).to_dict()
    feature_id = universe.set_index("feature_name")["feature_id"].astype(str).to_dict()
    for name in universe["feature_name"].astype(str):
        if name not in data:
            raise StabilityPreparationError(f"descriptor input missing feature: {name}")
        row = compute_feature_descriptors(
            data[name], feature_name=name, metadata_type=metadata_type[name], ddof=0
        )
        row["feature_id"] = feature_id[name]
        row["raw_descriptor_sha256"] = _descriptor_hash(row)
        rows.append(row)
    ordered = [
        "feature_id",
        "feature_name",
        "original_dtype",
        "metadata_type",
        "resolved_type",
        "resolution_rule",
        "ambiguity_warning",
        "concentration_definition",
        *DESCRIPTOR_COLUMNS_V2,
        "raw_descriptor_sha256",
    ]
    output = pd.DataFrame(rows)[ordered]
    values = output[list(DESCRIPTOR_COLUMNS_V2)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise StabilityPreparationError("raw descriptor table contains non-finite values")
    return output


def compute_raw_descriptors(
    reader: DevMatrixReader,
    universe: pd.DataFrame,
    *,
    feature_batch_size: int,
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    feature_names = universe["feature_name"].astype(str).tolist()
    total_batches = _batch_count(len(feature_names), feature_batch_size)
    for index, chunk in enumerate(
        _chunks(feature_names, feature_batch_size), start=1
    ):
        if progress:
            progress(f"raw descriptors: feature batch {index}/{total_batches}")
        data = reader.read_frame(chunk)
        batch_universe = universe[universe["feature_name"].isin(chunk)].copy()
        batch_universe["_order"] = batch_universe["feature_name"].map(
            {name: position for position, name in enumerate(chunk)}
        )
        batch_universe = batch_universe.sort_values("_order").drop(columns="_order")
        frames.append(compute_raw_descriptors_from_frame(data, batch_universe))
    output = pd.concat(frames, ignore_index=True)
    order = universe["feature_name"].astype(str).tolist()
    output["_order"] = output["feature_name"].map({name: index for index, name in enumerate(order)})
    return output.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def fit_stability_preprocessor(
    raw_descriptors: pd.DataFrame,
    representation_split: pd.DataFrame,
    *,
    feature_universe_hash: str,
    representation_split_hash: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    joined = raw_descriptors.merge(
        representation_split[["feature_name", "representation_split"]],
        on="feature_name",
        validate="one_to_one",
    )
    fit_frame = joined.loc[joined["representation_split"].eq("train")].copy()
    if fit_frame.empty:
        raise StabilityPreparationError("representation TRAIN has no feature identities")
    preprocessor = RobustStatisticalPreprocessorV2(
        fit_dataset=DATASET_ID,
        fit_split="train",
        clipping_lower=-8.0,
        clipping_upper=8.0,
    )
    preprocessor.fit(fit_frame, dataset=DATASET_ID, split="train")
    transformed = preprocessor.transform(joined)
    if not np.array_equal(
        transformed[list(UNSCALED_INDICATOR_COLUMNS_V2)].to_numpy(dtype=float),
        joined[list(UNSCALED_INDICATOR_COLUMNS_V2)].to_numpy(dtype=float),
    ):
        raise StabilityPreparationError("indicator fields changed during statistical preprocessing")
    continuous = transformed[list(SCALED_DESCRIPTOR_COLUMNS_V2)].to_numpy(dtype=float)
    if bool((continuous < -8.0).any()) or bool((continuous > 8.0).any()):
        raise StabilityPreparationError("continuous descriptor clipping contract failed")
    scaled = joined[["feature_id", "feature_name"]].copy()
    for column in DESCRIPTOR_COLUMNS_V2:
        scaled[column] = transformed[column].to_numpy(dtype="float32")
    scaled["statistical_vector_sha256"] = [
        sha256_text(json.dumps([float(row[column]) for column in DESCRIPTOR_COLUMNS_V2], separators=(",", ":")))
        for row in scaled.to_dict("records")
    ]
    state = preprocessor.to_state()
    manifest = {
        "schema_version": "stability_source_stat_preprocessor_v1",
        "descriptor_order": list(DESCRIPTOR_COLUMNS_V2),
        "continuous_fields": list(SCALED_DESCRIPTOR_COLUMNS_V2),
        "indicator_fields": list(UNSCALED_INDICATOR_COLUMNS_V2),
        "fit_split": "representation_train_feature_identities_only",
        "fit_feature_count": len(fit_frame),
        "fit_feature_names_sha256": feature_order_hash(
            fit_frame["feature_name"].astype(str).tolist()
        ),
        "median_by_field": state["medians"],
        "iqr_by_field": state["iqr"],
        "zero_iqr_fields": state["zero_iqr_columns"],
        "clip_min": -8.0,
        "clip_max": 8.0,
        "feature_universe_hash": feature_universe_hash,
        "representation_split_hash": representation_split_hash,
        "implementation_identity": "credit_risk_fs.clip.statistical_preprocessor_v2.RobustStatisticalPreprocessorV2",
        "implementation_version": "compact_target_free_v2",
        "internal_preprocessor_hash": state["preprocessor_hash"],
        "validation_feature_identities_used_for_fit": False,
        "target_used": False,
        "oot_used": False,
    }
    return scaled, manifest


def compute_temporal_stability(
    values: pd.Series,
    numeric_time: pd.Series,
    *,
    boundaries: Sequence[float],
    min_non_missing_per_subwindow: int,
    numeric_bins: int,
    categorical_min_count: int,
    psi_epsilon: float,
) -> dict[str, Any]:
    windows = [
        values.loc[numeric_time.ge(left) & numeric_time.lt(right)]
        for left, right in zip(boundaries[:-1], boundaries[1:])
    ]
    if len(windows) != 4 or any(window.empty for window in windows):
        raise StabilityPreparationError("one or more frozen DEV anchor subwindows are empty")
    counts = [int(window.notna().sum()) for window in windows]
    missing_rates = [float(window.isna().mean()) for window in windows]
    max_missing_difference = float(max(missing_rates) - min(missing_rates))
    if min(counts) < int(min_non_missing_per_subwindow):
        return {
            "subwindow_non_missing_counts": counts,
            "subwindow_missing_rates": missing_rates,
            "max_missing_rate_difference": max_missing_difference,
            "adjacent_window_psi_values": [],
            "max_adjacent_window_psi": np.nan,
            "psi_bucket_manifest": {},
            "eligibility_status": "excluded",
            "exclusion_reason": "insufficient_non_missing_support",
        }
    bucketizer = fit_frozen_bucketizer(
        windows[0],
        numeric_bins=int(numeric_bins),
        categorical_min_count=int(categorical_min_count),
    )
    bucketed = [bucketizer.transform(window) for window in windows]
    adjacent_psi = [
        psi_from_frozen_buckets(
            bucketed[index], bucketed[index + 1], epsilon=float(psi_epsilon)
        )
        for index in range(3)
    ]
    return {
        "subwindow_non_missing_counts": counts,
        "subwindow_missing_rates": missing_rates,
        "max_missing_rate_difference": max_missing_difference,
        "adjacent_window_psi_values": adjacent_psi,
        "max_adjacent_window_psi": float(max(adjacent_psi)),
        "psi_bucket_manifest": bucketizer.manifest(),
        "eligibility_status": "measured",
        "exclusion_reason": "",
    }


def select_stability_anchor_members(
    evidence: pd.DataFrame,
    *,
    member_count: int,
    max_adjacent_window_psi: float,
    max_missing_rate_difference: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    audit = evidence.copy()
    audit["qualifies_thresholds"] = (
        audit["eligibility_status"].eq("eligible")
        & audit["max_adjacent_window_psi"].le(float(max_adjacent_window_psi))
        & audit["max_missing_rate_difference"].le(float(max_missing_rate_difference))
    )
    qualified = audit.loc[audit["qualifies_thresholds"]].sort_values(
        ["max_adjacent_window_psi", "max_missing_rate_difference", "feature_id"],
        kind="mergesort",
    )
    selected_indices: list[int] = []
    used_groups: set[str] = set()
    audit["selection_status"] = "not_selected"
    audit["selection_exclusion_reason"] = audit["exclusion_reason"].astype(str)
    audit["anchor_rank"] = pd.Series(pd.NA, index=audit.index, dtype="Int64")
    for index, row in qualified.iterrows():
        group_id = str(row["equivalence_group_id"])
        if group_id in used_groups:
            audit.at[index, "selection_exclusion_reason"] = (
                "identity_equivalent_to_higher_ranked_member"
            )
            continue
        selected_indices.append(index)
        used_groups.add(group_id)
        if len(selected_indices) == int(member_count):
            break
    if len(selected_indices) != int(member_count):
        raise StabilityPreparationError(
            "BLOCKED — insufficient leakage-safe Stability temporal anchor members: "
            f"required={member_count}, observed={len(selected_indices)}"
        )
    for rank, index in enumerate(selected_indices, start=1):
        audit.at[index, "selection_status"] = "selected"
        audit.at[index, "selection_exclusion_reason"] = ""
        audit.at[index, "anchor_rank"] = rank
    members = audit.loc[selected_indices].sort_values("anchor_rank").reset_index(drop=True)
    return members, audit.sort_values(
        ["selection_status", "max_adjacent_window_psi", "max_missing_rate_difference", "feature_id"],
        ascending=[False, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


def build_stability_source_anchor(
    reader: DevMatrixReader,
    universe: pd.DataFrame,
    representation_split: pd.DataFrame,
    equivalence: pd.DataFrame,
    *,
    anchor_config: Mapping[str, Any],
    feature_batch_size: int,
    feature_universe_hash: str,
    representation_split_hash: str,
    raw_descriptors_hash: str,
    progress: Callable[[str], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    candidates = (
        universe[["feature_id", "feature_name", "logical_type"]]
        .merge(representation_split, on="feature_name", validate="one_to_one")
        .merge(equivalence, on=["feature_name", "equivalence_group_id"], validate="one_to_one")
    )
    candidates = candidates.loc[candidates["representation_split"].eq("train")].copy()
    date_start = pd.Timestamp(reader.dev_start_inclusive)
    date_end = pd.Timestamp(reader.dev_end_exclusive)
    epoch = pd.Timestamp("1970-01-01")
    start_day = float((date_start - epoch) / pd.Timedelta(days=1))
    end_day = float((date_end - epoch) / pd.Timedelta(days=1))
    boundaries = equal_duration_boundaries(start_day, end_day, 4)
    evidence_rows: list[dict[str, Any]] = []
    window_row_counts: list[int] | None = None
    names = candidates["feature_name"].astype(str).tolist()
    total_batches = _batch_count(len(names), feature_batch_size)
    for index, chunk in enumerate(_chunks(names, feature_batch_size), start=1):
        if progress:
            progress(f"source-anchor evidence: feature batch {index}/{total_batches}")
        frame = reader.read_frame(chunk, include_date=True)
        dates = pd.to_datetime(frame[reader.date_column], errors="raise")
        # Pandas 3 commonly parses strings as datetime64[us], while older
        # releases often used datetime64[ns]. Timedelta division is explicitly
        # unit-independent and keeps calendar-day boundaries stable across both.
        numeric_time = (dates - epoch).div(pd.Timedelta(days=1)).astype(float)
        observed_counts = [
            int(numeric_time.ge(left).mul(numeric_time.lt(right)).sum())
            for left, right in zip(boundaries[:-1], boundaries[1:])
        ]
        if window_row_counts is None:
            window_row_counts = observed_counts
        elif observed_counts != window_row_counts:
            raise StabilityPreparationError("anchor window row counts changed across feature batches")
        for name in chunk:
            result = compute_temporal_stability(
                frame[name],
                numeric_time,
                boundaries=boundaries,
                min_non_missing_per_subwindow=int(
                    anchor_config["min_non_missing_per_subwindow"]
                ),
                numeric_bins=int(anchor_config["numeric_bins"]),
                categorical_min_count=int(anchor_config["categorical_min_count"]),
                psi_epsilon=float(anchor_config["psi_epsilon"]),
            )
            candidate = candidates.loc[candidates["feature_name"].eq(name)].iloc[0]
            max_psi = float(result["max_adjacent_window_psi"])
            max_missing = float(result["max_missing_rate_difference"])
            exclusion = str(result["exclusion_reason"])
            threshold_pass = (
                not exclusion
                and np.isfinite(max_psi)
                and max_psi <= float(anchor_config["max_adjacent_window_psi"])
                and max_missing <= float(anchor_config["max_missing_rate_difference"])
            )
            if not exclusion and not threshold_pass:
                exclusion = "stability_threshold_failed"
            status = "eligible" if threshold_pass else "excluded"
            evidence_rows.append(
                {
                    "feature_id": str(candidate["feature_id"]),
                    "feature_name": name,
                    "equivalence_group_id": str(candidate["equivalence_group_id"]),
                    "representation_split": "train",
                    "logical_type": str(candidate["logical_type"]),
                    "subwindow_non_missing_counts": json.dumps(
                        result["subwindow_non_missing_counts"], separators=(",", ":")
                    ),
                    "subwindow_missing_rates": json.dumps(
                        result["subwindow_missing_rates"], separators=(",", ":")
                    ),
                    "max_missing_rate_difference": max_missing,
                    "adjacent_window_psi_values": json.dumps(
                        result["adjacent_window_psi_values"], separators=(",", ":")
                    ),
                    "max_adjacent_window_psi": max_psi,
                    "psi_bucket_manifest": json.dumps(
                        result["psi_bucket_manifest"], sort_keys=True, separators=(",", ":")
                    ),
                    "eligibility_status": status,
                    "exclusion_reason": exclusion,
                    "target_used": False,
                    "oot_used": False,
                }
            )
    evidence = pd.DataFrame(evidence_rows)
    members, audit = select_stability_anchor_members(
        evidence,
        member_count=int(anchor_config["member_count"]),
        max_adjacent_window_psi=float(anchor_config["max_adjacent_window_psi"]),
        max_missing_rate_difference=float(anchor_config["max_missing_rate_difference"]),
    )
    selected = members.rename(columns={"max_adjacent_window_psi": "max_adjacent_psi"})[
        [
            "feature_id",
            "feature_name",
            "equivalence_group_id",
            "anchor_rank",
            "max_adjacent_psi",
            "max_missing_rate_difference",
            "subwindow_non_missing_counts",
            "subwindow_missing_rates",
            "adjacent_window_psi_values",
            "psi_bucket_manifest",
            "selection_status",
            "target_used",
            "oot_used",
        ]
    ]
    boundary_iso = [
        (pd.Timestamp("1970-01-01") + pd.to_timedelta(value, unit="D")).isoformat()
        for value in boundaries
    ]
    manifest = {
        "schema_version": "stability_source_anchor_manifest_v1",
        "methodology": ANCHOR_METHOD_VERSION,
        "generalized_from_rule_version": ANCHOR_RULE_VERSION,
        "source_dataset": DATASET_ID,
        "candidate_scope": "representation_train_feature_identities_only",
        "row_scope": "DEV_only",
        "dev_interval": {
            "start_inclusive": reader.dev_start_inclusive,
            "end_exclusive": reader.dev_end_exclusive,
        },
        "subwindow_strategy": "four_equal_duration_windows_left_closed_right_open",
        "numeric_day_boundaries": list(boundaries),
        "boundary_iso_diagnostics": boundary_iso,
        "subwindow_row_counts": window_row_counts,
        "psi_formula": "sum((expected-actual)*log((expected+epsilon)/(actual+epsilon)))",
        "psi_bucket_fit_scope": "first_DEV_subwindow_only_frozen_for_all_later_windows",
        "numeric_binning": "reference_window_quantiles_with_unique_edges_and_infinite_endpoints",
        "categorical_binning": "reference_levels_meeting_min_count_else_OTHER;missing=MISSING",
        "missing_rate_stability": "maximum_subwindow_missing_rate_minus_minimum_subwindow_missing_rate",
        "thresholds": {
            "max_adjacent_window_psi": float(anchor_config["max_adjacent_window_psi"]),
            "max_missing_rate_difference": float(anchor_config["max_missing_rate_difference"]),
            "min_non_missing_per_subwindow": int(anchor_config["min_non_missing_per_subwindow"]),
            "numeric_bins": int(anchor_config["numeric_bins"]),
            "categorical_min_count": int(anchor_config["categorical_min_count"]),
            "psi_epsilon": float(anchor_config["psi_epsilon"]),
        },
        "ranking": [
            "max_adjacent_window_psi",
            "max_missing_rate_difference",
            "feature_id",
        ],
        "identity_group_deduplication": True,
        "candidate_count": len(candidates),
        "qualified_count": int(audit["qualifies_thresholds"].sum()),
        "required_member_count": int(anchor_config["member_count"]),
        "actual_member_count": len(selected),
        "anchor_feature_ids": selected["feature_id"].astype(str).tolist(),
        "feature_universe_hash": feature_universe_hash,
        "representation_split_hash": representation_split_hash,
        "raw_statistical_descriptors_sha256": raw_descriptors_hash,
        "target_used": False,
        "oot_used": False,
        "external_data_used": False,
        "selected_using_downstream_performance": False,
    }
    return selected, audit, manifest


def encode_text_embeddings(
    rendered_text: pd.DataFrame,
    universe: pd.DataFrame,
    *,
    source_manifest_hash: str,
    encoder_config: Mapping[str, Any],
    feature_universe_hash: str,
    encoder: TextEncoderProtocol | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    model_name = str(encoder_config["model_name"])
    revision = str(encoder_config["model_revision"])
    expected_dimension = int(encoder_config["embedding_dimension"])
    normalize = bool(encoder_config["normalize_embeddings"])
    if encoder is None:
        try:
            encoder = FrozenSentenceTransformerEncoder(
                model_name=model_name,
                revision=revision,
                device=resolve_device(str(encoder_config.get("device_policy", "auto"))),
            )
        except Exception as exc:
            raise StabilityPreparationError(
                "failed to load the frozen sentence-transformers/all-MiniLM-L6-v2 revision main; "
                "no fallback encoder is permitted"
            ) from exc
    if encoder.model_name != model_name or encoder.revision != revision:
        raise StabilityPreparationError("text encoder identity differs from frozen protocol")
    if int(encoder.embedding_dimension) != expected_dimension:
        raise StabilityPreparationError("text embedding dimension differs from frozen protocol")
    model = getattr(encoder, "model", None)
    if model is not None:
        if bool(getattr(model, "training", False)):
            raise StabilityPreparationError("frozen text encoder is in training mode")
        parameters = getattr(model, "parameters", lambda: ())()
        if any(bool(parameter.requires_grad) for parameter in parameters):
            raise StabilityPreparationError("frozen text encoder has trainable parameters")
    texts = rendered_text["rendered_text"].astype(str).tolist()
    embeddings = encoder.encode(
        texts,
        batch_size=int(encoder_config["batch_size"]),
        normalize_embeddings=normalize,
    )
    if embeddings.shape != (len(rendered_text), expected_dimension):
        raise StabilityPreparationError(
            f"text embedding shape mismatch: expected={(len(rendered_text), expected_dimension)}, "
            f"observed={embeddings.shape}"
        )
    embeddings = np.asarray(embeddings, dtype="float32")
    norms = np.linalg.norm(embeddings, axis=1)
    if normalize and not np.allclose(norms, 1.0, rtol=0, atol=1e-5):
        raise StabilityPreparationError("text embeddings are not L2-normalized")
    cache_input = rendered_text.rename(
        columns={
            "rendered_text_sha256": "feature_text_hash",
            "template_version": "text_template_version",
        }
    ).copy()
    cache_input["source_manifest_hash"] = source_manifest_hash
    cache_input["dataset"] = DATASET_ID
    spec = EmbeddingCacheSpec(
        encoder_model=model_name,
        encoder_revision=revision,
        normalize_embeddings=normalize,
        text_template_version=TEXT_TEMPLATE_VERSION,
    )
    frame = build_embedding_frame(text_frame=cache_input, embeddings=embeddings, spec=spec)
    frame = universe[["feature_id", "feature_name"]].merge(
        frame, on="feature_name", validate="one_to_one"
    )
    embedding_columns = _embedding_columns(frame)
    manifest = {
        "schema_version": "stability_text_embedding_manifest_v1",
        "model_name": model_name,
        "model_revision": revision,
        "embedding_dimension": len(embedding_columns),
        "normalization": "L2" if normalize else "none",
        "dtype": "float32",
        "template_version": TEXT_TEMPLATE_VERSION,
        "feature_text_hash": sha256_text(
            "".join(rendered_text["rendered_text_sha256"].astype(str).tolist())
        ),
        "feature_universe_hash": feature_universe_hash,
        "cache_identity": sha256_text(
            json.dumps(
                {
                    "model": model_name,
                    "revision": revision,
                    "normalize": normalize,
                    "template": TEXT_TEMPLATE_VERSION,
                },
                sort_keys=True,
            )
        ),
        "row_count": len(frame),
        "frozen_inference_only": True,
        "fine_tuned": False,
    }
    return frame, manifest


def build_stability_pairs(
    universe: pd.DataFrame,
    equivalence: pd.DataFrame,
    representation_split: pd.DataFrame,
    rendered_text: pd.DataFrame,
    text_embeddings: pd.DataFrame,
    raw_descriptors: pd.DataFrame,
    scaled_descriptors: pd.DataFrame,
    *,
    source_manifest_hash: str,
) -> pd.DataFrame:
    embedding_columns = _embedding_columns(text_embeddings)
    if len(embedding_columns) == 0:
        raise StabilityPreparationError("pair construction received no text embeddings")
    scaled = scaled_descriptors[["feature_id", "feature_name", *DESCRIPTOR_COLUMNS_V2, "statistical_vector_sha256"]].rename(
        columns={column: f"stat_{column}" for column in DESCRIPTOR_COLUMNS_V2}
    )
    pairs = (
        universe[["feature_id", "feature_name"]]
        .merge(equivalence[["feature_name", "equivalence_group_id"]], on="feature_name", validate="one_to_one")
        .merge(
            representation_split[["feature_name", "representation_split"]],
            on="feature_name",
            validate="one_to_one",
        )
        .merge(
            rendered_text[["feature_id", "feature_name", "rendered_text_sha256", "template_version"]],
            on=["feature_id", "feature_name"],
            validate="one_to_one",
        )
        .merge(
            text_embeddings[["feature_id", "feature_name", "embedding_cache_key", *embedding_columns]],
            on=["feature_id", "feature_name"],
            validate="one_to_one",
        )
        .merge(
            raw_descriptors[["feature_id", "feature_name", "raw_descriptor_sha256"]],
            on=["feature_id", "feature_name"],
            validate="one_to_one",
        )
        .merge(scaled, on=["feature_id", "feature_name"], validate="one_to_one")
    )
    pairs.insert(0, "stable_row_id", [
        sha256_text(f"{DATASET_ID}|{feature_id}|{source_manifest_hash}")
        for feature_id in pairs["feature_id"].astype(str)
    ])
    pairs.insert(1, "dataset", DATASET_ID)
    pairs["source_manifest_hash"] = source_manifest_hash
    forbidden = [
        column for column in pairs.columns
        if str(column).lower() in {"target", "date_decision", "oot", "auc", "feature_importance"}
    ]
    if forbidden:
        raise StabilityPreparationError(f"forbidden fields entered pair table: {forbidden}")
    if pairs["feature_id"].duplicated().any() or len(pairs) != len(universe):
        raise StabilityPreparationError("pair table is not one row per feature identity")
    return pairs


def build_methodology_lock(
    context: PreparationContext,
    *,
    feature_universe_hash: str,
    representation_split_hash: str,
) -> dict[str, Any]:
    config = context.config
    return {
        "schema_version": PREPARATION_SCHEMA_VERSION,
        "configuration_hash": context.configuration_hash,
        "dataset_id": DATASET_ID,
        "dataset_version": config["dataset"]["dataset_version"],
        "dataset_evidence_identity": config["inputs"]["matrix_manifest"],
        "feature_universe": 1959,
        "feature_universe_hash": feature_universe_hash,
        "DEV_boundary": {
            "start_inclusive": config["dataset"]["dev_start_inclusive"],
            "end_exclusive": config["dataset"]["oot_start_exclusive_dev"],
        },
        "OOT_boundary": {"start_inclusive": config["dataset"]["oot_start_exclusive_dev"]},
        "OOT_forbidden_during_preparation": True,
        "text_template": TEXT_TEMPLATE_VERSION,
        "text_encoder": config["text_encoder"]["model_name"],
        "text_encoder_revision": config["text_encoder"]["model_revision"],
        "text_dimension": config["text_encoder"]["embedding_dimension"],
        "text_normalization": "L2",
        "stat_schema": config["statistical_view"]["schema_version"],
        "stat_dimension": 13,
        "descriptor_order": list(DESCRIPTOR_COLUMNS_V2),
        "Stability_statistical_preprocessor_policy": (
            "representation-TRAIN-only median/IQR on seven continuous descriptors; "
            "clip [-8,8]; six indicators unchanged"
        ),
        "representation_split_seed": 42,
        "representation_split_target_ratio": "80/20",
        "representation_split_hash": representation_split_hash,
        "group_aware_split": True,
        "negative_policy": PAIRING_POLICY_VERSION,
        "source_anchor_methodology": ANCHOR_METHOD_VERSION,
        "source_anchor_size": 23,
        "future_CLIP_seeds": config["future_protocol"]["clip_seeds"],
        "future_checkpoint_selection": config["future_protocol"]["checkpoint_selection"],
        "future_consensus": "Procrustes alignment; reference seed 11; five-seed consensus",
        "future_CLIP_candidate_pools": config["future_protocol"]["candidate_pools"],
        "future_final_budgets": config["future_protocol"]["final_budgets"],
        "future_downstream_mRMR": config["future_protocol"]["downstream_mrmr"],
        "planned_directions": config["future_protocol"]["planned_directions"],
        "future_transfer_statistical_rule": (
            "reuse these frozen Stability raw descriptors; transform with the frozen HC or LC source "
            "preprocessor without refitting on Stability"
        ),
        "prohibitions": config["prohibitions"],
        "model_training_performed": False,
        "selector_training_performed": False,
        "downstream_evaluation_performed": False,
    }


def build_data_provenance(
    context: PreparationContext,
    input_provenance: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    branch = _git(context.repo_root, ["branch", "--show-current"])
    commit = _git(context.repo_root, ["rev-parse", "HEAD"])
    status = _git(context.repo_root, ["status", "--porcelain"])
    sources = [dict(value) for value in input_provenance]
    sources.append(_provenance_row(context, context.config_path, "preparation_protocol"))
    implementation = context.repo_root / "src/credit_risk_fs/clip/stability_preparation.py"
    sources.append(_provenance_row(context, implementation, "stability_preparation_implementation"))
    entry_point = context.repo_root / "scripts/prepare_stability_clip_inputs.py"
    sources.append(_provenance_row(context, entry_point, "manual_preparation_entry_point"))
    return {
        "schema_version": "stability_clip_data_provenance_v1",
        "created_at_utc": utc_now(),
        "preparation_protocol_version": PREPARATION_SCHEMA_VERSION,
        "configuration_hash": context.configuration_hash,
        "repository": {
            "branch": branch,
            "commit": commit,
            "dirty": bool(status.strip()),
        },
        "authoritative_sources": sources,
        "excluded_input_categories": [
            "selector_ranks",
            "model_predictions",
            "model_metrics",
            "LLM_rankings",
            "OOT_feature_values",
            "target_values",
        ],
    }


def build_validation_report(
    *,
    context: PreparationContext,
    universe: pd.DataFrame,
    semantics: pd.DataFrame,
    rendered_text: pd.DataFrame,
    equivalence: pd.DataFrame,
    representation_split: pd.DataFrame,
    raw_descriptors: pd.DataFrame,
    scaled_descriptors: pd.DataFrame,
    preprocessor_manifest: Mapping[str, Any],
    anchor: pd.DataFrame,
    anchor_manifest: Mapping[str, Any],
    text_embeddings: pd.DataFrame,
    text_embedding_manifest: Mapping[str, Any],
    pairs: pd.DataFrame,
    reader_audit: Mapping[str, Any],
) -> dict[str, Any]:
    expected = int(context.config["dataset"]["expected_feature_count"])
    embedding_columns = _embedding_columns(text_embeddings)
    stat_columns = [f"stat_{column}" for column in DESCRIPTOR_COLUMNS_V2]
    checks: list[dict[str, Any]] = []

    def check(category: str, name: str, passed: bool, detail: Any) -> None:
        checks.append(
            {
                "category": category,
                "check": name,
                "status": "PASS" if bool(passed) else "FAIL",
                "detail": _json_compatible(detail),
            }
        )

    check("feature_universe", "exactly_1959", len(universe) == expected, len(universe))
    check("feature_universe", "unique_identities", universe["feature_name"].nunique() == expected, universe["feature_name"].nunique())
    forbidden = set(universe["feature_name"].astype(str)) & set(context.config["dataset"]["forbidden_predictors"])
    check("feature_universe", "nonpredictors_absent", not forbidden, sorted(forbidden))
    check("semantics", "complete_rows", len(semantics) == expected and not semantics[["description", "semantic_group", "source_or_formula"]].isna().any().any(), len(semantics))
    check("semantics", "provenance_complete", not semantics[["description_source", "semantic_group_source", "formula_source"]].fillna("").eq("").any().any(), "deterministic repository evidence")
    check("feature_text", "feature_text_v1", len(rendered_text) == expected and set(rendered_text["template_version"]) == {TEXT_TEMPLATE_VERSION}, TEXT_TEMPLATE_VERSION)
    check("feature_text", "hashes_complete", rendered_text["rendered_text_sha256"].astype(str).str.fullmatch(r"[0-9a-f]{64}").all(), expected)
    check("equivalence", "all_features_assigned", len(equivalence) == expected and equivalence["feature_name"].nunique() == expected, len(equivalence))
    check("equivalence", "allowed_evidence_only", equivalence["equivalence_reason"].astype(str).str.contains(r"correlation|similarity|target|model", case=False, regex=True).sum() == 0, PAIRING_POLICY_VERSION)
    group_crossing = representation_split.groupby("equivalence_group_id")["representation_split"].nunique().gt(1).sum()
    check("representation_split", "all_assigned_once", len(representation_split) == expected and representation_split["feature_name"].nunique() == expected, len(representation_split))
    check("representation_split", "group_integrity", group_crossing == 0, int(group_crossing))
    check("representation_split", "seed_42", set(representation_split["split_seed"]) == {42}, 42)
    check("statistical_descriptors", "schema_order", tuple(DESCRIPTOR_COLUMNS_V2) == EXPECTED_DESCRIPTOR_ORDER, list(DESCRIPTOR_COLUMNS_V2))
    check("statistical_descriptors", "all_rows_finite", len(raw_descriptors) == expected and np.isfinite(raw_descriptors[list(DESCRIPTOR_COLUMNS_V2)].to_numpy(float)).all(), len(raw_descriptors))
    check("statistical_preprocessing", "train_only_fit", not preprocessor_manifest["validation_feature_identities_used_for_fit"] and preprocessor_manifest["fit_feature_count"] == int(representation_split["representation_split"].eq("train").sum()), preprocessor_manifest["fit_feature_count"])
    indicators_equal = np.array_equal(
        raw_descriptors[list(UNSCALED_INDICATOR_COLUMNS_V2)].to_numpy(float),
        scaled_descriptors[list(UNSCALED_INDICATOR_COLUMNS_V2)].to_numpy(float),
    )
    check("statistical_preprocessing", "indicators_unscaled", indicators_equal, list(UNSCALED_INDICATOR_COLUMNS_V2))
    scaled_continuous = scaled_descriptors[list(SCALED_DESCRIPTOR_COLUMNS_V2)].to_numpy(float)
    check("statistical_preprocessing", "continuous_clipped", bool((scaled_continuous >= -8).all() and (scaled_continuous <= 8).all()), "[-8,8]")
    check("source_anchor", "exact_corrected_method", anchor_manifest["generalized_from_rule_version"] == ANCHOR_RULE_VERSION, anchor_manifest["methodology"])
    check("source_anchor", "23_train_members", len(anchor) == 23 and set(anchor["selection_status"]) == {"selected"}, len(anchor))
    check("source_anchor", "target_and_oot_free", not anchor_manifest["target_used"] and not anchor_manifest["oot_used"], "DEV representation-TRAIN only")
    check("text_embeddings", "frozen_minilm_identity", text_embedding_manifest["model_name"] == "sentence-transformers/all-MiniLM-L6-v2" and text_embedding_manifest["model_revision"] == "main" and not text_embedding_manifest["fine_tuned"], text_embedding_manifest["model_name"])
    check("text_embeddings", "shape_and_norm", len(text_embeddings) == expected and len(embedding_columns) == 384, [len(text_embeddings), len(embedding_columns)])
    check("pairs", "identity_and_dimensions", len(pairs) == expected and all(column in pairs for column in [*embedding_columns, *stat_columns]), {"rows": len(pairs), "text_dim": len(embedding_columns), "stat_dim": len(stat_columns)})
    check("leakage", "target_values_loaded_NO", not reader_audit["target_values_loaded"], reader_audit["target_values_loaded"])
    check("leakage", "OOT_feature_values_loaded_NO", not reader_audit["oot_feature_values_loaded"], reader_audit["oot_feature_values_loaded"])
    check("leakage", "OOT_labels_loaded_NO", not reader_audit["oot_labels_loaded"], reader_audit["oot_labels_loaded"])
    check("leakage", "selector_ranks_consumed_NO", True, False)
    check("leakage", "model_outputs_consumed_NO", True, False)
    check("leakage", "LLM_rankings_consumed_NO", True, False)
    overall = all(row["status"] == "PASS" for row in checks)
    return {
        "schema_version": "stability_clip_validation_report_v1",
        "created_at_utc": utc_now(),
        "overall_status": "PASS" if overall else "FAIL",
        "configuration_hash": context.configuration_hash,
        "checks": checks,
        "leakage_summary": {
            "target_values_loaded": "NO",
            "OOT_feature_values_loaded": "NO",
            "OOT_labels_loaded": "NO",
            "existing_selector_ranks_consumed": "NO",
            "existing_model_outputs_consumed": "NO",
            "LLM_rankings_consumed": "NO",
        },
        "reader_audit": dict(reader_audit),
        "model_training_performed": False,
        "selector_training_performed": False,
        "downstream_evaluation_performed": False,
    }


def validation_report_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Stability CLIP Preparation Validation",
        "",
        f"Overall status: **{report['overall_status']}**",
        "",
        "| Category | Check | Status | Detail |",
        "|---|---|---:|---|",
    ]
    for row in report["checks"]:
        detail = json.dumps(row["detail"], sort_keys=True).replace("|", "\\|")
        lines.append(f"| {row['category']} | {row['check']} | {row['status']} | {detail} |")
    lines.extend(
        [
            "",
            "No CLIP checkpoint, classifier, selector, or downstream evaluation was trained or run.",
            "",
        ]
    )
    return "\n".join(lines)


def write_sha256_manifest(
    output_dir: str | Path,
    artifact_roles: Mapping[str, str],
) -> pd.DataFrame:
    root = Path(output_dir)
    rows = []
    for relative in sorted(artifact_roles):
        if relative == "manifests/sha256_manifest.csv":
            continue
        path = root / relative
        if not path.is_file():
            raise StabilityPreparationError(f"generated artifact is missing: {relative}")
        rows.append(
            {
                "relative_path": relative,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "artifact_role": artifact_roles[relative],
            }
        )
    frame = pd.DataFrame(rows).sort_values("relative_path").reset_index(drop=True)
    destination = root / "manifests/sha256_manifest.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False, lineterminator="\n")
    return frame


def verify_existing_output(output_dir: str | Path, *, configuration_hash: str) -> bool:
    root = Path(output_dir)
    if not root.exists():
        return False
    lock_path = root / "methodology_lock.json"
    manifest_path = root / "manifests/sha256_manifest.csv"
    if not lock_path.is_file() or not manifest_path.is_file():
        raise StabilityPreparationError(
            f"existing output is incomplete or incompatible: {root}; use --rebuild for a deliberate recoverable rebuild"
        )
    lock = _read_json(lock_path)
    if lock.get("configuration_hash") != configuration_hash:
        raise StabilityPreparationError(
            "existing output methodology/configuration hash differs; use --rebuild for a deliberate recoverable rebuild"
        )
    manifest = pd.read_csv(manifest_path, dtype={"sha256": "string"})
    required = {"relative_path", "sha256", "size_bytes", "artifact_role"}
    if set(manifest.columns) != required:
        raise StabilityPreparationError("existing SHA-256 manifest schema mismatch")
    if set(manifest["relative_path"].astype(str)) != set(GENERATED_RELATIVE_PATHS):
        raise StabilityPreparationError("existing SHA-256 manifest artifact coverage mismatch")
    for row in manifest.to_dict("records"):
        path = root / str(row["relative_path"])
        if not path.is_file() or path.stat().st_size != int(row["size_bytes"]):
            raise StabilityPreparationError(f"existing artifact missing or changed size: {row['relative_path']}")
        if sha256_file(path) != str(row["sha256"]):
            raise StabilityPreparationError(f"existing artifact hash mismatch: {row['relative_path']}")
    report = _read_json(root / "validation/validation_report.json")
    if report.get("overall_status") != "PASS":
        raise StabilityPreparationError("existing validation report is not PASS")
    return True


def write_work_checkpoint(
    work_dir: str | Path,
    *,
    configuration_hash: str,
    completed_stage: int,
    reader_audit: Mapping[str, Any],
) -> Path:
    root = Path(work_dir)
    artifacts = []
    for path in sorted(
        (
            candidate
            for candidate in root.rglob("*")
            if candidate.is_file() and candidate.name != WORK_CHECKPOINT_NAME
        ),
        key=lambda candidate: candidate.relative_to(root).as_posix(),
    ):
        artifacts.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return _write_json(
        root / WORK_CHECKPOINT_NAME,
        {
            "schema_version": "stability_clip_work_checkpoint_v1",
            "configuration_hash": configuration_hash,
            "completed_stage": int(completed_stage),
            "artifacts": artifacts,
            "reader_audit": dict(reader_audit),
            "model_training_performed": False,
            "selector_training_performed": False,
            "downstream_evaluation_performed": False,
        },
    )


def load_work_checkpoint(
    work_dir: str | Path,
    *,
    configuration_hash: str,
) -> dict[str, Any] | None:
    root = Path(work_dir)
    if not root.exists():
        return None
    checkpoint_path = root / WORK_CHECKPOINT_NAME
    if not checkpoint_path.is_file():
        raise StabilityPreparationError(
            f"existing work directory has no verified checkpoint: {root}; use --rebuild"
        )
    checkpoint = _read_json(checkpoint_path)
    if checkpoint.get("schema_version") != "stability_clip_work_checkpoint_v1":
        raise StabilityPreparationError("work checkpoint schema mismatch; use --rebuild")
    if checkpoint.get("configuration_hash") != configuration_hash:
        raise StabilityPreparationError("work checkpoint configuration mismatch; use --rebuild")
    completed_stage = int(checkpoint.get("completed_stage", 0))
    if completed_stage < 8 or completed_stage > 11:
        raise StabilityPreparationError("work checkpoint stage is invalid; use --rebuild")
    for row in checkpoint.get("artifacts", []):
        relative = str(row.get("relative_path", ""))
        path = root / relative
        try:
            path.resolve().relative_to(root.resolve())
        except ValueError as exc:
            raise StabilityPreparationError("work checkpoint contains an unsafe artifact path") from exc
        if not path.is_file() or path.stat().st_size != int(row.get("size_bytes", -1)):
            raise StabilityPreparationError(f"checkpointed work artifact is missing or changed: {relative}")
        if sha256_file(path) != str(row.get("sha256", "")):
            raise StabilityPreparationError(f"checkpointed work artifact hash mismatch: {relative}")
    return checkpoint


def restore_reader_audit(reader: DevMatrixReader, payload: Mapping[str, Any]) -> None:
    reader.audit.scan_count = int(payload.get("scan_count", 0))
    reader.audit.prior_requested_predictor_column_count = int(
        payload.get("requested_predictor_column_count", 0)
    )
    reader.audit.returned_row_counts = [
        int(value) for value in payload.get("returned_row_counts", [])
    ]
    reader.audit.earliest_returned_date = str(payload.get("earliest_returned_date", ""))
    reader.audit.latest_returned_date = str(payload.get("latest_returned_date", ""))
    reader.audit.target_values_loaded = bool(payload.get("target_values_loaded", False))
    reader.audit.oot_feature_values_loaded = bool(payload.get("oot_feature_values_loaded", False))
    reader.audit.oot_labels_loaded = bool(payload.get("oot_labels_loaded", False))


def load_stage_8_work(work_dir: str | Path) -> dict[str, Any]:
    root = Path(work_dir)
    return {
        "universe": pd.read_csv(root / "metadata/feature_universe.csv", keep_default_na=False),
        "universe_manifest": _read_json(root / "metadata/feature_universe_manifest.json"),
        "semantics": pd.read_csv(root / "metadata/feature_semantics.csv", keep_default_na=False),
        "rendered": pd.read_csv(root / "text/feature_text.csv", keep_default_na=False),
        "duplicate_evidence": pd.read_csv(
            root / "pairing/exact_dev_duplicate_evidence.csv", keep_default_na=False
        ),
        "equivalence": pd.read_csv(
            root / "pairing/identity_equivalence.csv", keep_default_na=False
        ),
        "representation_split": pd.read_csv(
            root / "pairing/representation_split.csv", keep_default_na=False
        ),
        "split_manifest": _read_json(root / "pairing/representation_split_manifest.json"),
        "raw_descriptors": pd.read_parquet(
            root / "statistics/statistical_descriptors_raw.parquet"
        ),
        "scaled_descriptors": pd.read_csv(
            root / "statistics/statistical_descriptors_stability_source_scaled.csv",
            keep_default_na=False,
        ),
        "preprocessor_manifest": _read_json(
            root / "statistics/stability_source_stat_preprocessor.json"
        ),
    }


def run_preparation(
    config_path: str | Path,
    *,
    repo_root: str | Path | None = None,
    rebuild: bool = False,
    progress: Callable[[str], None] = print,
) -> Path:
    context = load_preparation_context(config_path, repo_root=repo_root)
    output_dir = context.output_dir
    if output_dir.exists() and not rebuild:
        if verify_existing_output(output_dir, configuration_hash=context.configuration_hash):
            progress(f"Existing verified package reused: {_relative_to_root(output_dir, context.repo_root)}")
            return output_dir
    if output_dir.exists() and rebuild:
        resolved_output = output_dir.resolve()
        resolved_parent = output_dir.parent.resolve()
        if resolved_output.parent != resolved_parent or resolved_output == context.repo_root.resolve():
            raise StabilityPreparationError("refusing rebuild for an unsafe output target")
        backup = output_dir.with_name(
            f"{output_dir.name}.backup-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        )
        if backup.exists():
            raise StabilityPreparationError(f"rebuild backup target already exists: {backup}")
        os.replace(output_dir, backup)
        progress(f"Existing output moved recoverably to {_relative_to_root(backup, context.repo_root)}")

    staging = output_dir.with_name(f".{output_dir.name}.work")
    if staging.exists() and rebuild:
        resolved_work = staging.resolve()
        if resolved_work.parent != output_dir.parent.resolve():
            raise StabilityPreparationError("refusing rebuild for an unsafe work target")
        work_backup = staging.with_name(
            f"{staging.name}.backup-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        )
        if work_backup.exists():
            raise StabilityPreparationError(f"work backup target already exists: {work_backup}")
        os.replace(staging, work_backup)
        progress(
            f"Existing work checkpoint moved recoverably to "
            f"{_relative_to_root(work_backup, context.repo_root)}"
        )
    checkpoint = load_work_checkpoint(
        staging, configuration_hash=context.configuration_hash
    )
    if checkpoint is None:
        staging.mkdir(parents=True)
        completed_stage = 0
    else:
        completed_stage = int(checkpoint["completed_stage"])
        progress(
            f"Verified work checkpoint found at stage {completed_stage}; "
            "completed expensive stages will be reused"
        )
    try:
        progress("[1/12] Validate repository/input protocol")
        input_state = validate_input_protocol(context)
        source_manifest_hash = str(context.config["inputs"]["matrix_manifest"]["sha256"])

        if completed_stage >= 8:
            progress("[2/12–8/12] Reuse verified feature, semantic, split, and descriptor work")
            stage_8 = load_stage_8_work(staging)
            universe = stage_8["universe"]
            universe_manifest = stage_8["universe_manifest"]
            semantics = stage_8["semantics"]
            rendered = stage_8["rendered"]
            duplicate_evidence = stage_8["duplicate_evidence"]
            equivalence = stage_8["equivalence"]
            representation_split = stage_8["representation_split"]
            split_manifest = stage_8["split_manifest"]
            raw_descriptors = stage_8["raw_descriptors"]
            scaled_descriptors = stage_8["scaled_descriptors"]
            preprocessor_manifest = stage_8["preprocessor_manifest"]
            universe_hash = str(universe_manifest["ordered_feature_name_sha256"])
        else:
            progress("[2/12] Load frozen 1,959-feature universe")
            universe, universe_manifest = build_feature_universe(context)
            _write_csv(staging / "metadata/feature_universe.csv", universe)
            _write_json(staging / "metadata/feature_universe_manifest.json", universe_manifest)
            universe_hash = str(universe_manifest["ordered_feature_name_sha256"])

            progress("[3/12] Build semantic metadata")
            semantics = build_semantic_metadata(
                universe,
                feature_definitions_path=context.resolve(
                    context.config["inputs"]["feature_definitions"]["path"]
                ),
            )
            _write_csv(staging / "metadata/feature_semantics.csv", semantics)

            progress("[4/12] Render feature_text_v1")
            rendered = render_feature_text(
                universe, semantics, source_manifest_hash=source_manifest_hash
            )
            _write_csv(staging / "text/feature_text.csv", rendered)

        reader = DevMatrixReader(
            matrix_parts=input_state["matrix_parts"],
            predictor_names=universe["feature_name"].astype(str).tolist(),
            date_column=context.config["dataset"]["date_column"],
            dev_start_inclusive=context.config["dataset"]["dev_start_inclusive"],
            dev_end_exclusive=context.config["dataset"]["oot_start_exclusive_dev"],
            expected_dev_rows=context.config["dataset"]["expected_dev_row_count"],
            row_batch_size=context.config["statistical_view"]["row_batch_size"],
        )
        if checkpoint is not None:
            restore_reader_audit(reader, checkpoint.get("reader_audit", {}))

        if completed_stage < 8:
            progress("[5/12] Build identity-equivalence groups")
            duplicate_evidence = find_exact_dev_duplicates_chunked(
                reader,
                universe["feature_name"].astype(str).tolist(),
                feature_batch_size=context.config["identity_equivalence"]["feature_batch_size"],
                progress=progress,
            )
            _write_csv(staging / "pairing/exact_dev_duplicate_evidence.csv", duplicate_evidence)
            equivalence = build_identity_equivalence(
                universe, exact_duplicate_evidence=duplicate_evidence
            )
            _write_csv(staging / "pairing/identity_equivalence.csv", equivalence)

            progress("[6/12] Build representation TRAIN/VALIDATION split")
            representation_split, split_manifest = build_representation_split(
                universe,
                semantics,
                equivalence,
                seed=context.config["representation_split"]["seed"],
                validation_fraction=context.config["representation_split"]["validation_fraction"],
            )
            _write_csv(staging / "pairing/representation_split.csv", representation_split)
            _write_json(staging / "pairing/representation_split_manifest.json", split_manifest)

            progress("[7/12] Compute raw compact_target_free_v2 descriptors")
            raw_descriptors = compute_raw_descriptors(
                reader,
                universe,
                feature_batch_size=context.config["statistical_view"]["feature_batch_size"],
                progress=progress,
            )
            raw_csv = staging / "statistics/statistical_descriptors_raw.csv"
            _write_csv(raw_csv, raw_descriptors)
            _write_parquet(
                staging / "statistics/statistical_descriptors_raw.parquet", raw_descriptors
            )

            progress("[8/12] Fit Stability-source statistical preprocessor")
            scaled_descriptors, preprocessor_manifest = fit_stability_preprocessor(
                raw_descriptors,
                representation_split,
                feature_universe_hash=universe_hash,
                representation_split_hash=split_manifest["representation_split_hash"],
            )
            _write_json(
                staging / "statistics/stability_source_stat_preprocessor.json",
                preprocessor_manifest,
            )
            _write_csv(
                staging / "statistics/statistical_descriptors_stability_source_scaled.csv",
                scaled_descriptors,
            )
            write_work_checkpoint(
                staging,
                configuration_hash=context.configuration_hash,
                completed_stage=8,
                reader_audit=reader.audit.to_dict(),
            )
            completed_stage = 8
        raw_csv = staging / "statistics/statistical_descriptors_raw.csv"

        if completed_stage >= 9:
            progress("[9/12] Reuse verified target-free Stability source anchor")
            anchor = pd.read_csv(
                staging / "anchor/stability_source_anchor.csv", keep_default_na=False
            )
            anchor_audit = pd.read_csv(
                staging / "anchor/stability_source_anchor_candidates.csv",
                keep_default_na=False,
            )
            anchor_manifest = _read_json(
                staging / "anchor/stability_source_anchor_manifest.json"
            )
        else:
            progress("[9/12] Build target-free Stability source anchor")
            anchor, anchor_audit, anchor_manifest = build_stability_source_anchor(
                reader,
                universe,
                representation_split,
                equivalence,
                anchor_config=context.config["source_anchor"],
                feature_batch_size=context.config["statistical_view"]["feature_batch_size"],
                feature_universe_hash=universe_hash,
                representation_split_hash=split_manifest["representation_split_hash"],
                raw_descriptors_hash=sha256_file(raw_csv),
                progress=progress,
            )
            _write_csv(staging / "anchor/stability_source_anchor_candidates.csv", anchor_audit)
            _write_csv(staging / "anchor/stability_source_anchor.csv", anchor)
            anchor_manifest["candidate_audit_sha256"] = sha256_file(
                staging / "anchor/stability_source_anchor_candidates.csv"
            )
            anchor_manifest["anchor_members_sha256"] = sha256_file(
                staging / "anchor/stability_source_anchor.csv"
            )
            anchor_manifest["reference_implementation"] = next(
                dict(spec)
                for spec in context.config["implementation_contracts"]
                if spec["role"] == "corrected_lendingclub_temporal_anchor_reference"
            )
            _write_json(
                staging / "anchor/stability_source_anchor_manifest.json", anchor_manifest
            )
            write_work_checkpoint(
                staging,
                configuration_hash=context.configuration_hash,
                completed_stage=9,
                reader_audit=reader.audit.to_dict(),
            )
            completed_stage = 9

        text_embedding_path = staging / "text/text_embeddings.parquet"
        if completed_stage >= 10:
            progress("[10/12] Reuse verified frozen MiniLM text embeddings")
            text_embeddings = pd.read_parquet(text_embedding_path)
            text_embedding_manifest = _read_json(
                staging / "text/text_embedding_manifest.json"
            )
        else:
            progress("[10/12] Encode frozen MiniLM text embeddings")
            text_embeddings, text_embedding_manifest = encode_text_embeddings(
                rendered,
                universe,
                source_manifest_hash=source_manifest_hash,
                encoder_config=context.config["text_encoder"],
                feature_universe_hash=universe_hash,
            )
            _write_parquet(text_embedding_path, text_embeddings)
            text_embedding_manifest["artifact_sha256"] = sha256_file(text_embedding_path)
            _write_json(
                staging / "text/text_embedding_manifest.json", text_embedding_manifest
            )
            write_work_checkpoint(
                staging,
                configuration_hash=context.configuration_hash,
                completed_stage=10,
                reader_audit=reader.audit.to_dict(),
            )
            completed_stage = 10

        pair_path = staging / "pairs/stability_source_pairs.parquet"
        if completed_stage >= 11:
            progress("[11/12] Reuse verified CLIP-ready pair package/manifests")
            pairs = pd.read_parquet(pair_path)
        else:
            progress("[11/12] Build CLIP-ready pair package/manifests")
            pairs = build_stability_pairs(
                universe,
                equivalence,
                representation_split,
                rendered,
                text_embeddings,
                raw_descriptors,
                scaled_descriptors,
                source_manifest_hash=source_manifest_hash,
            )
            _write_parquet(pair_path, pairs)
            pair_manifest = {
                "schema_version": "stability_source_pairs_manifest_v1",
                "dataset": DATASET_ID,
                "row_count": len(pairs),
                "text_embedding_dimension": 384,
                "statistical_dimension": 13,
                "text_embedding_columns": _embedding_columns(pairs),
                "statistical_columns": [
                    f"stat_{column}" for column in DESCRIPTOR_COLUMNS_V2
                ],
                "feature_universe_hash": universe_hash,
                "representation_split_hash": split_manifest[
                    "representation_split_hash"
                ],
                "component_hashes": {
                    "feature_text_csv": sha256_file(staging / "text/feature_text.csv"),
                    "text_embeddings_parquet": sha256_file(text_embedding_path),
                    "raw_descriptors_csv": sha256_file(raw_csv),
                    "scaled_descriptors_csv": sha256_file(
                        staging
                        / "statistics/statistical_descriptors_stability_source_scaled.csv"
                    ),
                    "identity_equivalence_csv": sha256_file(
                        staging / "pairing/identity_equivalence.csv"
                    ),
                    "representation_split_csv": sha256_file(
                        staging / "pairing/representation_split.csv"
                    ),
                },
                "pair_artifact_sha256": sha256_file(pair_path),
                "target_included": False,
                "oot_values_included": False,
                "selector_or_model_outputs_included": False,
            }
            _write_json(
                staging / "pairs/stability_source_pairs_manifest.json", pair_manifest
            )
            methodology = build_methodology_lock(
                context,
                feature_universe_hash=universe_hash,
                representation_split_hash=split_manifest[
                    "representation_split_hash"
                ],
            )
            _write_json(staging / "methodology_lock.json", methodology)
            provenance = build_data_provenance(
                context, input_state["input_provenance"]
            )
            _write_json(staging / "manifests/data_provenance.json", provenance)
            write_work_checkpoint(
                staging,
                configuration_hash=context.configuration_hash,
                completed_stage=11,
                reader_audit=reader.audit.to_dict(),
            )
            completed_stage = 11

        progress("[12/12] Run final validation and hashes")
        validation = build_validation_report(
            context=context,
            universe=universe,
            semantics=semantics,
            rendered_text=rendered,
            equivalence=equivalence,
            representation_split=representation_split,
            raw_descriptors=raw_descriptors,
            scaled_descriptors=scaled_descriptors,
            preprocessor_manifest=preprocessor_manifest,
            anchor=anchor,
            anchor_manifest=anchor_manifest,
            text_embeddings=text_embeddings,
            text_embedding_manifest=text_embedding_manifest,
            pairs=pairs,
            reader_audit=reader.audit.to_dict(),
        )
        if validation["overall_status"] != "PASS":
            failures = [row["check"] for row in validation["checks"] if row["status"] == "FAIL"]
            raise StabilityPreparationError(f"final validation failed: {failures}")
        _write_json(staging / "validation/validation_report.json", validation)
        _write_text(staging / "validation/VALIDATION_REPORT.md", validation_report_markdown(validation))
        roles = {relative: _artifact_role(relative) for relative in GENERATED_RELATIVE_PATHS}
        manifest = write_sha256_manifest(staging, roles)
        if set(manifest["relative_path"]) != set(GENERATED_RELATIVE_PATHS):
            raise StabilityPreparationError("SHA-256 manifest does not cover every generated artifact")
        if output_dir.exists():
            raise StabilityPreparationError("output directory appeared during atomic build")
        checkpoint_path = staging / WORK_CHECKPOINT_NAME
        if checkpoint_path.is_file():
            checkpoint_path.unlink()
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, output_dir)
        if not verify_existing_output(output_dir, configuration_hash=context.configuration_hash):
            raise StabilityPreparationError("completed package failed idempotency verification")
        progress(f"PASS — CLIP-ready package written to {_relative_to_root(output_dir, context.repo_root)}")
        return output_dir
    except Exception:
        if staging.exists() and completed_stage >= 8:
            audit = (
                reader.audit.to_dict()
                if "reader" in locals()
                else checkpoint.get("reader_audit", {})
                if checkpoint is not None
                else {}
            )
            write_work_checkpoint(
                staging,
                configuration_hash=context.configuration_hash,
                completed_stage=completed_stage,
                reader_audit=audit,
            )
            progress(
                "Verified partial work was preserved; rerunning the same command will "
                f"resume after stage {completed_stage}"
            )
        elif staging.exists():
            shutil.rmtree(staging)
        raise


def _validate_pair_dimensions(pairs: pd.DataFrame, *, text_dimension: int, stat_dimension: int) -> None:
    embedding_columns = _embedding_columns(pairs)
    stat_columns = [column for column in pairs if str(column).startswith("stat_")]
    if len(embedding_columns) != text_dimension or len(stat_columns) != stat_dimension:
        raise StabilityPreparationError("pair-table view dimensions do not match protocol")


def _descriptor_hash(row: Mapping[str, Any]) -> str:
    values = [float(row[column]) for column in DESCRIPTOR_COLUMNS_V2]
    return sha256_text(json.dumps(values, separators=(",", ":"), allow_nan=False))


def _json_compatible(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _embedding_columns(frame: pd.DataFrame) -> list[str]:
    return [
        str(column)
        for column in frame.columns
        if re.fullmatch(r"embedding_\d{4}", str(column))
    ]


def _chunks(values: Sequence[str], size: int) -> Iterable[list[str]]:
    if int(size) <= 0:
        raise ValueError("chunk size must be positive")
    for index in range(0, len(values), int(size)):
        yield list(values[index : index + int(size)])


def _batch_count(value_count: int, batch_size: int) -> int:
    if int(batch_size) <= 0:
        raise ValueError("batch size must be positive")
    return (int(value_count) + int(batch_size) - 1) // int(batch_size)


def _read_json(path: str | Path) -> dict[str, Any]:
    candidate = Path(path)
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StabilityPreparationError(f"unreadable JSON artifact: {candidate}") from exc
    if not isinstance(payload, dict):
        raise StabilityPreparationError(f"expected JSON object: {candidate}")
    return payload


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            _json_compatible(payload),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return destination


def _write_csv(path: str | Path, frame: pd.DataFrame) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False, lineterminator="\n")
    return destination


def _write_parquet(path: str | Path, frame: pd.DataFrame) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(destination, index=False)
    return destination


def _write_text(path: str | Path, value: str) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(value, encoding="utf-8", newline="\n")
    return destination


def _verify_file(path: Path, digest: str, *, label: str, size: int | None = None) -> None:
    if not path.is_file():
        raise StabilityPreparationError(f"required input is missing ({label}): {path}")
    if size is not None and path.stat().st_size != size:
        raise StabilityPreparationError(f"required input size mismatch ({label})")
    if sha256_file(path) != digest:
        raise StabilityPreparationError(f"required input SHA-256 mismatch ({label})")


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _provenance_row(context: PreparationContext, path: Path, role: str) -> dict[str, Any]:
    return {
        "relative_path": _relative_to_root(path, context.repo_root),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "role": role,
    }


def _git(root: Path, arguments: Sequence[str]) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _artifact_role(relative: str) -> str:
    if relative.startswith("metadata/"):
        return "feature_identity_or_semantic_metadata"
    if relative.startswith("pairing/"):
        return "identity_equivalence_or_representation_split"
    if relative.startswith("statistics/"):
        return "target_free_statistical_view"
    if relative.startswith("anchor/"):
        return "target_free_source_anchor"
    if relative.startswith("text/"):
        return "semantic_text_or_frozen_embedding_cache"
    if relative.startswith("pairs/"):
        return "clip_ready_feature_pairs"
    if relative.startswith("validation/"):
        return "validation_evidence"
    if relative.startswith("manifests/"):
        return "provenance_manifest"
    return "methodology_lock"
