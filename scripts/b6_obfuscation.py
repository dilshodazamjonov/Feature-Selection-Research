"""Safety-critical helpers for the Home Credit B6 name-obfuscation ablation.

The B6 output is a single-draw name-obfuscation sensitivity analysis.  It is
not evidence that memorisation has been removed or that the name effect has
been causally isolated.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from credit_risk_fs.evaluation.paired_inference import (  # noqa: E402
    BOOTSTRAP_MINIMUM_VALID,
    BOOTSTRAP_REPETITIONS,
    BOOTSTRAP_SEED,
    align_paired_predictions as canonical_align_paired_predictions,
    paired_delong_test,
    paired_stratified_bootstrap,
)
from credit_risk_fs.feature_metadata.builder import infer_semantic_group  # noqa: E402


DATASET = "homecredit"
PARTITIONS = ("fold1", "fold2", "fold3", "fold4", "fold5", "full_dev")
BACKBONES = ("lr", "catboost")
MODEL_SNAPSHOT = "gpt-4.1-mini-2025-04-14"
TEMPERATURE = 0
EVIDENCE_MODE = "definitions_only"
RANKING_BUDGET = 100
GLOBAL_FEATURE_COUNT = 529
MAPPING_SEED = 20260914
CALL_ORDER_SEED = 20260915
HO_ROW_COUNT = 120_053
PREFIX_BY_BACKBONE = {"lr": 20, "catboost": 40}

RANKINGS_FILENAME = "rankings.csv"
PERFORMANCE_FILENAME = "performance.csv"
PAIRWISE_FILENAME = "pairwise.csv"
OUTPUT_FILENAMES = (RANKINGS_FILENAME, PERFORMANCE_FILENAME, PAIRWISE_FILENAME)

RANKING_COLUMNS = (
    "dataset",
    "partition",
    "rank",
    "feature_id",
    "original_feature_name",
)
PERFORMANCE_COLUMNS = (
    "dataset",
    "backbone",
    "K",
    "ho_auc",
    "fold1_auc",
    "fold2_auc",
    "fold3_auc",
    "fold4_auc",
    "fold5_auc",
)
PAIRWISE_COLUMNS = (
    "dataset",
    "backbone",
    "method_A",
    "method_B",
    "auc_A",
    "auc_B",
    "delta",
    "ci95_low",
    "ci95_high",
    "p_value",
    "n_ho",
)


class B6ProtocolError(RuntimeError):
    """Raised when a frozen B6 protocol condition cannot be proved."""


def _validated_feature_names(
    feature_names: Iterable[str], *, expected_count: int | None
) -> list[str]:
    names = list(feature_names)
    if any(not isinstance(name, str) or not name for name in names):
        raise B6ProtocolError("feature universe contains a non-string or empty name")
    if len(set(names)) != len(names):
        raise B6ProtocolError("feature universe contains duplicate names")
    if expected_count is not None and len(names) != expected_count:
        raise B6ProtocolError(
            f"feature universe has {len(names)} names; expected {expected_count}"
        )
    return names


def build_global_feature_mapping(
    feature_names: Iterable[str],
    *,
    expected_count: int | None = GLOBAL_FEATURE_COUNT,
    seed: int = MAPPING_SEED,
) -> dict[str, str]:
    """Build the frozen global original-name -> opaque-ID mapping."""

    names = _validated_feature_names(feature_names, expected_count=expected_count)
    ordered = sorted(names)
    permutation = np.random.default_rng(seed).permutation(len(ordered))
    shuffled_names = [ordered[int(index)] for index in permutation]
    width = max(3, len(str(len(shuffled_names))))
    mapping = {
        name: f"F{position:0{width}d}"
        for position, name in enumerate(shuffled_names, start=1)
    }
    expected_ids = {f"F{position:0{width}d}" for position in range(1, len(names) + 1)}
    if set(mapping) != set(names) or set(mapping.values()) != expected_ids:
        raise AssertionError("internal error: generated mapping is not a bijection")
    return mapping


def reverse_feature_mapping(mapping: Mapping[str, str]) -> dict[str, str]:
    if len(mapping) != len(set(mapping.values())):
        raise B6ProtocolError("cannot reverse a non-bijective feature mapping")
    return {opaque_id: original for original, opaque_id in mapping.items()}


def _replacement_pattern(mapping: Mapping[str, str]) -> re.Pattern[str]:
    if not mapping:
        raise B6ProtocolError("cannot scrub candidate records with an empty mapping")
    names = _validated_feature_names(mapping.keys(), expected_count=None)
    # One regex substitution prevents a replacement from being matched again.
    alternatives = (re.escape(name) for name in sorted(names, key=lambda x: (-len(x), x)))
    return re.compile("|".join(alternatives))


def scrub_text(text: str, mapping: Mapping[str, str]) -> str:
    """Replace possibly overlapping original names in one non-cascading pass."""

    pattern = _replacement_pattern(mapping)
    return pattern.sub(lambda match: mapping[match.group(0)], str(text))


def scrub_candidate_records(
    records: Sequence[Mapping[str, Any]], mapping: Mapping[str, str]
) -> list[dict[str, Any]]:
    """Replace original-name literals in every string while preserving structure/order."""

    pattern = _replacement_pattern(mapping)

    def scrub(value: Any) -> Any:
        if isinstance(value, str):
            return pattern.sub(lambda match: mapping[match.group(0)], value)
        if isinstance(value, Mapping):
            return {key: scrub(item) for key, item in value.items()}
        if isinstance(value, list):
            return [scrub(item) for item in value]
        if isinstance(value, tuple):
            return tuple(scrub(item) for item in value)
        return value

    scrubbed = [scrub(record) for record in records]
    serialized = json.dumps(scrubbed, ensure_ascii=False, separators=(",", ":"))
    leaks = [name for name in mapping if name in serialized]
    if leaks:
        digest = hashlib.sha256(leaks[0].encode("utf-8")).hexdigest()
        raise B6ProtocolError(
            "candidate-record scrubbing left an original-name literal; "
            f"name_sha256={digest}"
        )
    return scrubbed


_SOURCE_BY_PREFIX = {
    "BURO": "bureau",
    "PREV": "previous_application",
    "POS": "POS_CASH_balance",
    "INSTAL": "installments_payments",
    "CC": "credit_card_balance",
}
_AGGREGATIONS = {"MIN", "MAX", "MEAN", "SUM", "VAR"}


def _description_registry(description_csv_path: str | Path) -> list[dict[str, str]]:
    try:
        frame = pd.read_csv(description_csv_path, encoding="utf-8")
    except UnicodeDecodeError:
        frame = pd.read_csv(description_csv_path, encoding="latin1")
    normalized = {str(column).casefold(): str(column) for column in frame.columns}
    required = {"table", "row", "description"}
    if not required <= set(normalized):
        raise B6ProtocolError(
            "Home Credit description CSV must contain Table, Row, and Description"
        )
    records: list[dict[str, str]] = []
    for row in frame.itertuples(index=False, name=None):
        values = dict(zip(map(str, frame.columns), row, strict=True))
        feature = values[normalized["row"]]
        if pd.isna(feature) or not str(feature).strip():
            continue
        description = values[normalized["description"]]
        table = values[normalized["table"]]
        records.append(
            {
                "name": str(feature).strip(),
                "table": "" if pd.isna(table) else str(table).strip(),
                "description": (
                    "" if pd.isna(description) else " ".join(str(description).split())
                ),
            }
        )
    return records


def _source_and_lineage(feature_name: str) -> tuple[str, str, str, str]:
    parts = feature_name.split("_")
    prefix = parts[0] if parts else ""
    if prefix in _SOURCE_BY_PREFIX and len(parts) >= 3 and parts[-1] in _AGGREGATIONS:
        return (
            _SOURCE_BY_PREFIX[prefix],
            "_".join(parts[1:-1]),
            parts[-1].lower(),
            "1",
        )
    return "application_train", feature_name, "identity", "0"


def _lookup_definition(
    registry: Sequence[Mapping[str, str]], original_feature: str, source_table: str
) -> str:
    matches = [row for row in registry if row["name"] == original_feature]
    if matches:
        source_token = source_table.casefold().replace("_", "")
        matching_source = [
            row
            for row in matches
            if source_token in row["table"].casefold().replace("_", "")
        ]
        selected = (matching_source or matches)[0]
        if selected["description"]:
            return selected["description"]
    return original_feature.replace("_", " ").casefold()


def _render_definition_record(fields: Mapping[str, Any]) -> dict[str, Any]:
    ordered_fields = {
        "name": str(fields["name"]),
        "source_family": str(fields["source_family"]),
        "source_table": str(fields["source_table"]),
        "original_feature": str(fields["original_feature"]),
        "depth": str(fields["depth"]),
        "aggregation": str(fields["aggregation"]),
        "dtype": str(fields["dtype"]),
        "logical_type": str(fields["logical_type"]),
        "approved_definition": str(fields["approved_definition"]),
    }
    rendered = "- " + json.dumps(
        ordered_fields,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    return {
        **ordered_fields,
        "rendered_description": rendered,
        "description_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
    }


def build_homecredit_definition_records(
    candidate_features: Sequence[str],
    *,
    dtypes: Mapping[str, Any],
    description_csv_path: str | Path,
    expected_count: int = GLOBAL_FEATURE_COUNT,
) -> list[dict[str, Any]]:
    """Build deterministic, outcome-independent records for the prospective B6 run."""

    features = _validated_feature_names(candidate_features, expected_count=expected_count)
    if set(features) != set(map(str, dtypes)):
        raise B6ProtocolError("dtype registry does not exactly cover the candidate universe")
    registry = _description_registry(description_csv_path)
    records: list[dict[str, Any]] = []
    for feature in features:
        source_table, original_feature, aggregation, depth = _source_and_lineage(feature)
        description = _lookup_definition(registry, original_feature, source_table)
        semantic_group = infer_semantic_group(
            feature, description=description, table=source_table
        )
        if aggregation == "identity":
            lineage = f"identity column from {source_table}"
        else:
            lineage = (
                f"{aggregation} of {original_feature} from {source_table}, "
                "grouped by applicant"
            )
        dtype = dtypes[feature]
        logical_type = "numeric" if pd.api.types.is_numeric_dtype(dtype) else "categorical"
        approved = (
            f"{description}. Semantic group: {semantic_group}. Lineage: {lineage}."
        )
        records.append(
            _render_definition_record(
                {
                    "name": feature,
                    "source_family": source_table,
                    "source_table": source_table,
                    "original_feature": original_feature,
                    "depth": depth,
                    "aggregation": aggregation,
                    "dtype": str(dtype),
                    "logical_type": logical_type,
                    "approved_definition": approved,
                }
            )
        )
    return records


def obfuscate_definition_records(
    named_records: Sequence[Mapping[str, Any]], mapping: Mapping[str, str]
) -> list[dict[str, Any]]:
    """Scrub names and rebuild the derived rendered text/hash fields."""

    scrubbed = scrub_candidate_records(named_records, mapping)
    records: list[dict[str, Any]] = []
    for record in scrubbed:
        fields = {
            key: value
            for key, value in record.items()
            if key not in {"rendered_description", "description_sha256"}
        }
        records.append(_render_definition_record(fields))
    return records


def _walk_keys(value: Any, path: str = "request") -> Iterable[tuple[str, str]]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{path}.{key}"
            yield str(key), child
            yield from _walk_keys(item, child)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _walk_keys(item, f"{path}[{index}]")


def _candidate_record_id(record: Mapping[str, Any]) -> str:
    for field in ("feature_name", "name", "feature_id"):
        if field in record:
            value = record[field]
            if not isinstance(value, str):
                raise B6ProtocolError(f"candidate record {field} must be a string")
            return value
    raise B6ProtocolError("candidate record has no feature-name/ID field")


_FORBIDDEN_EVIDENCE_KEYS = {
    "target",
    "target_column",
    "label",
    "labels",
    "outcome",
    "outcomes",
    "ho",
    "ho_label",
    "ho_target",
    "oot",
    "prediction",
    "predictions",
    "prediction_probability",
    "metric",
    "metrics",
    "iv",
    "woe",
    "mutual_information",
    "model_importance",
    "shap",
    "validation_auc",
    "auc",
    "reverse_mapping",
    "original_feature_name",
}


def build_request_payload(
    *,
    messages: Sequence[Mapping[str, Any]],
    response_format: Mapping[str, Any],
    model: str = MODEL_SNAPSHOT,
    temperature: int = TEMPERATURE,
) -> dict[str, Any]:
    """Construct only the fields accepted by the canonical chat-completions call.

    The local mapping is intentionally not an argument and therefore cannot be
    serialized by this boundary helper.
    """

    if model != MODEL_SNAPSHOT:
        raise B6ProtocolError(f"B6 model must be the exact snapshot {MODEL_SNAPSHOT}")
    if temperature != TEMPERATURE:
        raise B6ProtocolError("B6 temperature must be zero")
    if not messages:
        raise B6ProtocolError("B6 API request requires at least one message")
    return {
        "model": model,
        "temperature": temperature,
        "messages": [dict(message) for message in messages],
        "response_format": dict(response_format),
    }


def assert_request_payload_safe(
    payload: Mapping[str, Any],
    *,
    original_feature_names: Iterable[str],
    candidate_records: Sequence[Mapping[str, Any]],
    expected_candidate_ids: Sequence[str],
    evidence_mode: str = EVIDENCE_MODE,
    ranking_budget: int = RANKING_BUDGET,
) -> None:
    """Fail immediately before an API call if the fully rendered request leaks."""

    if evidence_mode != EVIDENCE_MODE:
        raise B6ProtocolError(f"evidence_mode must be {EVIDENCE_MODE}")
    if ranking_budget != RANKING_BUDGET:
        raise B6ProtocolError(f"ranking budget must be {RANKING_BUDGET}")

    expected = list(expected_candidate_ids)
    observed = [_candidate_record_id(record) for record in candidate_records]
    if observed != expected or len(observed) != len(set(observed)):
        raise B6ProtocolError(
            "candidate record IDs/order do not exactly match the expected eligible partition"
        )

    normalized_forbidden = {key.casefold() for key in _FORBIDDEN_EVIDENCE_KEYS}
    for key, path in (
        *list(_walk_keys(payload)),
        *list(_walk_keys(candidate_records, "candidate_records")),
    ):
        if key.casefold() in normalized_forbidden:
            raise B6ProtocolError(f"forbidden evidence field in API payload at {path}")

    response_format = payload.get("response_format")
    schema_text = json.dumps(response_format, ensure_ascii=False, separators=(",", ":"))
    if (
        f'"minItems":{RANKING_BUDGET}' not in schema_text
        or f'"maxItems":{RANKING_BUDGET}' not in schema_text
    ):
        raise B6ProtocolError("API response schema does not require exactly 100 selections")

    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    rendered_messages = "\n".join(
        str(message.get("content", ""))
        for message in payload.get("messages", [])
        if isinstance(message, Mapping)
    )
    missing_rendered_ids = [
        candidate_id
        for candidate_id in expected
        if re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(candidate_id)}(?![A-Za-z0-9_])",
            rendered_messages,
        )
        is None
    ]
    if missing_rendered_ids:
        digest = hashlib.sha256(missing_rendered_ids[0].encode("utf-8")).hexdigest()
        raise B6ProtocolError(
            "expected candidate identifier is absent from the rendered request; "
            f"identifier_sha256={digest}"
        )
    for name in original_feature_names:
        if name in serialized:
            digest = hashlib.sha256(name.encode("utf-8")).hexdigest()
            context_digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
            raise B6ProtocolError(
                "original-name literal detected in rendered API request; "
                f"name_sha256={digest}; request_sha256={context_digest}"
            )
    if "reverse_mapping" in serialized or "original_feature_name" in serialized:
        raise B6ProtocolError("de-obfuscation metadata detected in rendered API request")


def validate_ranking_response(
    response: Mapping[str, Any],
    *,
    candidate_ids: Iterable[str],
    original_feature_names: Iterable[str],
    expected_count: int = RANKING_BUDGET,
) -> tuple[str, ...]:
    """Validate and return the first schema-valid ranked opaque-ID sequence."""

    if not isinstance(response, Mapping):
        raise B6ProtocolError("ranking response is not a JSON object")
    if "status" in response and response["status"] not in {
        "success",
        "succeeded",
        "completed",
        "ok",
    }:
        raise B6ProtocolError("ranking response status is not successful")
    selected = response.get("selected_features")
    if not isinstance(selected, list) or any(not isinstance(item, str) for item in selected):
        raise B6ProtocolError("ranking response selected_features must be a string list")
    if len(selected) != expected_count:
        raise B6ProtocolError(
            f"ranking response returned {len(selected)} identifiers; expected {expected_count}"
        )
    if len(set(selected)) != len(selected):
        raise B6ProtocolError("ranking response contains duplicate identifiers")
    original_names = set(original_feature_names)
    returned_original = [item for item in selected if item in original_names]
    if returned_original:
        digest = hashlib.sha256(returned_original[0].encode("utf-8")).hexdigest()
        raise B6ProtocolError(
            "ranking response returned an original feature name; "
            f"name_sha256={digest}"
        )
    eligible = set(candidate_ids)
    invented = [item for item in selected if item not in eligible]
    if invented:
        digest = hashlib.sha256(invented[0].encode("utf-8")).hexdigest()
        raise B6ProtocolError(
            "ranking response contains an ineligible identifier; "
            f"identifier_sha256={digest}"
        )
    return tuple(selected)


def translate_ranking_prefix(
    ranked_ids: Sequence[str], reverse_mapping: Mapping[str, str], k: int
) -> tuple[str, ...]:
    if k not in PREFIX_BY_BACKBONE.values():
        raise B6ProtocolError("B6 ranking prefix must be 20 or 40")
    if len(ranked_ids) < k:
        raise B6ProtocolError(f"ranking has fewer than the requested {k} features")
    prefix = list(ranked_ids[:k])
    if len(prefix) != len(set(prefix)):
        raise B6ProtocolError("ranking prefix contains duplicate identifiers")
    try:
        translated = tuple(reverse_mapping[feature_id] for feature_id in prefix)
    except KeyError as exc:
        raise B6ProtocolError("ranking prefix contains an unmapped identifier") from exc
    return translated


@dataclass
class HOAccessGate:
    """A small explicit state gate guarding held-out population access."""

    full_dev_subset_frozen: bool = False
    full_dev_model_frozen: bool = False

    def mark_subset_frozen(self) -> None:
        self.full_dev_subset_frozen = True

    def mark_model_frozen(self) -> None:
        self.full_dev_model_frozen = True

    def assert_access_allowed(self) -> None:
        if not (self.full_dev_subset_frozen and self.full_dev_model_frozen):
            raise B6ProtocolError(
                "HO access is forbidden until the full-DEV subset, preprocessing, "
                "and fitted model are frozen"
            )


def align_paired_predictions(
    method_a: pd.DataFrame,
    method_b: pd.DataFrame,
    *,
    production: bool = True,
    expected_rows: int = HO_ROW_COUNT,
) -> pd.DataFrame:
    """Apply the canonical alignment, with the B6 production population check."""

    aligned = canonical_align_paired_predictions(method_a, method_b)
    if production and len(aligned) != expected_rows:
        raise B6ProtocolError(
            f"aligned HO population has {len(aligned)} rows; expected {expected_rows}"
        )
    return aligned


def draw_paired_stratified_indices(
    target: Sequence[int] | np.ndarray,
    *,
    repetitions: int,
    seed: int = BOOTSTRAP_SEED,
) -> list[np.ndarray]:
    """Expose the canonical shared-index sampling contract for focused tests."""

    target_array = np.asarray(target, dtype=int)
    positives = np.flatnonzero(target_array == 1)
    negatives = np.flatnonzero(target_array == 0)
    if not len(positives) or not len(negatives):
        raise B6ProtocolError("stratified sampling requires both target classes")
    generator = np.random.default_rng(seed)
    draws: list[np.ndarray] = []
    for _ in range(repetitions):
        sampled_positive = generator.choice(positives, size=len(positives), replace=True)
        sampled_negative = generator.choice(negatives, size=len(negatives), replace=True)
        draws.append(np.concatenate([sampled_positive, sampled_negative]))
    return draws


def build_pairwise_row(
    *,
    backbone: str,
    aligned: pd.DataFrame,
    production: bool = True,
    repetitions: int = BOOTSTRAP_REPETITIONS,
    seed: int = BOOTSTRAP_SEED,
    minimum_valid: int = BOOTSTRAP_MINIMUM_VALID,
) -> dict[str, Any]:
    """Calculate one B6 row using canonical paired DeLong and bootstrap routines."""

    if backbone not in BACKBONES:
        raise B6ProtocolError(f"unsupported backbone: {backbone}")
    if production:
        if len(aligned) != HO_ROW_COUNT:
            raise B6ProtocolError(
                f"paired inference has {len(aligned)} rows; expected {HO_ROW_COUNT}"
            )
        if (repetitions, seed, minimum_valid) != (
            BOOTSTRAP_REPETITIONS,
            BOOTSTRAP_SEED,
            BOOTSTRAP_MINIMUM_VALID,
        ):
            raise B6ProtocolError("production paired-inference constants were changed")
    target = aligned["target"].to_numpy(dtype=int)
    score_a = aligned["score_a"].to_numpy(dtype=float)
    score_b = aligned["score_b"].to_numpy(dtype=float)
    auc_a = float(roc_auc_score(target, score_a))
    auc_b = float(roc_auc_score(target, score_b))
    delta = auc_a - auc_b
    delong = paired_delong_test(target, score_a, score_b)
    bootstrap = paired_stratified_bootstrap(
        aligned,
        repetitions=repetitions,
        seed=seed,
        minimum_valid=minimum_valid,
    )
    auc_interval = bootstrap["metrics"]["auc"]
    if not auc_interval["interval_valid"]:
        raise B6ProtocolError("paired bootstrap did not produce a valid AUC interval")
    if delong["auc_difference_a_minus_b"] != delong["auc_a"] - delong["auc_b"]:
        raise AssertionError("canonical DeLong difference is internally inconsistent")
    return {
        "dataset": DATASET,
        "backbone": backbone,
        "method_A": "Obfuscated Pure LLM",
        "method_B": "Pure LLM",
        "auc_A": auc_a,
        "auc_B": auc_b,
        "delta": delta,
        "ci95_low": float(auc_interval["ci95_percentile_lower"]),
        "ci95_high": float(auc_interval["ci95_percentile_upper"]),
        "p_value": float(delong["two_sided_p_value"]),
        "n_ho": len(aligned),
    }


def rankings_frame(
    rankings: Mapping[str, Sequence[str]], reverse_mapping: Mapping[str, str]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for partition in PARTITIONS:
        if partition not in rankings:
            raise B6ProtocolError(f"ranking is missing partition {partition}")
        identifiers = list(rankings[partition])
        if len(identifiers) != RANKING_BUDGET or len(set(identifiers)) != RANKING_BUDGET:
            raise B6ProtocolError(f"{partition} must contain 100 distinct identifiers")
        for rank, feature_id in enumerate(identifiers, start=1):
            if feature_id not in reverse_mapping:
                raise B6ProtocolError(f"{partition} contains an unmapped identifier")
            rows.append(
                {
                    "dataset": DATASET,
                    "partition": partition,
                    "rank": rank,
                    "feature_id": feature_id,
                    "original_feature_name": reverse_mapping[feature_id],
                }
            )
    return pd.DataFrame(rows, columns=RANKING_COLUMNS)


def performance_frame(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(list(rows), columns=PERFORMANCE_COLUMNS)


def pairwise_frame(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(list(rows), columns=PAIRWISE_COLUMNS)


def _assert_columns(frame: pd.DataFrame, expected: Sequence[str], label: str) -> None:
    if tuple(frame.columns) != tuple(expected):
        raise B6ProtocolError(
            f"{label} columns are {list(frame.columns)}; expected {list(expected)}"
        )
    if any(str(column).casefold().startswith("unnamed") for column in frame.columns):
        raise B6ProtocolError(f"{label} contains an unnamed index column")


def validate_output_frames(
    ranking: pd.DataFrame,
    performance: pd.DataFrame,
    pairwise: pd.DataFrame,
) -> None:
    """Validate all three deliverable tables before and after publication."""

    _assert_columns(ranking, RANKING_COLUMNS, RANKINGS_FILENAME)
    _assert_columns(performance, PERFORMANCE_COLUMNS, PERFORMANCE_FILENAME)
    _assert_columns(pairwise, PAIRWISE_COLUMNS, PAIRWISE_FILENAME)

    if len(ranking) != len(PARTITIONS) * RANKING_BUDGET:
        raise B6ProtocolError("rankings.csv must contain exactly 600 rows")
    if ranking.loc[:, ["dataset", "partition", "rank", "feature_id", "original_feature_name"]].isna().any().any():
        raise B6ProtocolError("rankings.csv contains missing values")
    if set(ranking["dataset"].astype(str)) != {DATASET}:
        raise B6ProtocolError("rankings.csv dataset must be homecredit")
    expected_partition_order = [
        partition for partition in PARTITIONS for _ in range(RANKING_BUDGET)
    ]
    if ranking["partition"].astype(str).tolist() != expected_partition_order:
        raise B6ProtocolError("rankings.csv partition ordering is invalid")
    expected_ranks = list(range(1, RANKING_BUDGET + 1)) * len(PARTITIONS)
    numeric_ranks = pd.to_numeric(ranking["rank"], errors="raise")
    if not np.equal(numeric_ranks.to_numpy(dtype=float), numeric_ranks.astype(int)).all():
        raise B6ProtocolError("rankings.csv ranks must be integers")
    observed_ranks = numeric_ranks.astype(int).tolist()
    if observed_ranks != expected_ranks:
        raise B6ProtocolError("rankings.csv ranks are not contiguous 1 through 100")
    if ranking.duplicated(["partition", "rank"]).any():
        raise B6ProtocolError("rankings.csv has duplicate partition/rank pairs")
    if ranking.duplicated(["partition", "feature_id"]).any():
        raise B6ProtocolError("rankings.csv has duplicate partition/feature_id pairs")

    if len(performance) != 2 or performance["backbone"].astype(str).tolist() != list(
        BACKBONES
    ):
        raise B6ProtocolError("performance.csv must contain ordered lr, catboost rows")
    if set(performance["dataset"].astype(str)) != {DATASET}:
        raise B6ProtocolError("performance.csv dataset must be homecredit")
    numeric_k = pd.to_numeric(performance["K"], errors="raise")
    if not np.equal(numeric_k.to_numpy(dtype=float), numeric_k.astype(int)).all():
        raise B6ProtocolError("performance.csv K values must be integers")
    if numeric_k.astype(int).tolist() != [20, 40]:
        raise B6ProtocolError("performance.csv K values must be 20 and 40")
    performance_values = performance.loc[:, PERFORMANCE_COLUMNS[3:]].apply(
        pd.to_numeric, errors="raise"
    )
    if not np.isfinite(performance_values.to_numpy(dtype=float)).all():
        raise B6ProtocolError("performance.csv metrics must be finite")

    if len(pairwise) != 2 or pairwise["backbone"].astype(str).tolist() != list(BACKBONES):
        raise B6ProtocolError("pairwise.csv must contain ordered lr, catboost rows")
    if set(pairwise["dataset"].astype(str)) != {DATASET}:
        raise B6ProtocolError("pairwise.csv dataset must be homecredit")
    if set(pairwise["method_A"].astype(str)) != {"Obfuscated Pure LLM"}:
        raise B6ProtocolError("pairwise.csv method_A is invalid")
    if set(pairwise["method_B"].astype(str)) != {"Pure LLM"}:
        raise B6ProtocolError("pairwise.csv method_B is invalid")
    numeric_n_ho = pd.to_numeric(pairwise["n_ho"], errors="raise")
    if not np.equal(numeric_n_ho.to_numpy(dtype=float), numeric_n_ho.astype(int)).all():
        raise B6ProtocolError("pairwise.csv n_ho must be an integer")
    if numeric_n_ho.astype(int).tolist() != [
        HO_ROW_COUNT,
        HO_ROW_COUNT,
    ]:
        raise B6ProtocolError("pairwise.csv n_ho must be 120053")
    numeric = pairwise.loc[
        :, ["auc_A", "auc_B", "delta", "ci95_low", "ci95_high", "p_value"]
    ].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise B6ProtocolError("pairwise.csv inference values must be finite")
    if not numeric["auc_A"].between(0.0, 1.0).all() or not numeric["auc_B"].between(0.0, 1.0).all():
        raise B6ProtocolError("pairwise.csv AUC values must be within [0, 1]")
    if not numeric["p_value"].between(0.0, 1.0).all():
        raise B6ProtocolError("pairwise.csv p_value must be within [0, 1]")
    for row in pairwise.itertuples(index=False):
        if float(row.delta) != float(row.auc_A) - float(row.auc_B):
            raise B6ProtocolError("pairwise.csv delta is not exactly auc_A minus auc_B")


def refuse_existing_output(output_dir: str | Path) -> Path:
    output = Path(output_dir).resolve()
    if output.exists() or output.is_symlink():
        raise B6ProtocolError(
            f"refusing to overwrite or reuse existing output directory: {output}"
        )
    return output


def validate_output_directory(output_dir: str | Path) -> None:
    output = Path(output_dir)
    observed = sorted(path.name for path in output.iterdir())
    if observed != sorted(OUTPUT_FILENAMES):
        raise B6ProtocolError(
            f"output directory must contain exactly {list(OUTPUT_FILENAMES)}; "
            f"observed {observed}"
        )
    frames = {
        name: pd.read_csv(output / name, float_precision="round_trip")
        for name in OUTPUT_FILENAMES
    }
    validate_output_frames(
        frames[RANKINGS_FILENAME],
        frames[PERFORMANCE_FILENAME],
        frames[PAIRWISE_FILENAME],
    )
    for name in OUTPUT_FILENAMES:
        raw = (output / name).read_bytes()
        raw.decode("utf-8")
        if b"\r\n" in raw:
            raise B6ProtocolError(f"{name} does not use canonical LF line endings")


def publish_output_frames_atomic(
    output_dir: str | Path,
    ranking: pd.DataFrame,
    performance: pd.DataFrame,
    pairwise: pd.DataFrame,
) -> Path:
    """Write to sibling staging and rename only after complete validation."""

    output = refuse_existing_output(output_dir)
    validate_output_frames(ranking, performance, pairwise)
    parent = output.parent
    if not parent.is_dir():
        raise B6ProtocolError(f"output parent directory does not exist: {parent}")
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=parent))
    published = False
    frames = {
        RANKINGS_FILENAME: ranking,
        PERFORMANCE_FILENAME: performance,
        PAIRWISE_FILENAME: pairwise,
    }
    try:
        for filename in OUTPUT_FILENAMES:
            frames[filename].to_csv(
                staging / filename,
                index=False,
                encoding="utf-8",
                lineterminator="\n",
                float_format="%.17g",
            )
        validate_output_directory(staging)
        refuse_existing_output(output)
        os.rename(staging, output)
        published = True
        validate_output_directory(output)
    except Exception:
        if staging.exists():
            resolved_staging = staging.resolve()
            if resolved_staging.parent != parent.resolve():
                raise AssertionError("refusing unsafe staging cleanup")
            shutil.rmtree(resolved_staging)
        if published and output.exists():
            resolved_output = output.resolve()
            if resolved_output.parent != parent.resolve():
                raise AssertionError("refusing unsafe failed-publication cleanup")
            shutil.rmtree(resolved_output)
        raise
    return output


def publish_mocked_ablation(
    *,
    output_dir: str | Path,
    rankings: Mapping[str, Sequence[str]],
    reverse_mapping: Mapping[str, str],
    performance_rows: Sequence[Mapping[str, Any]],
    pairwise_rows: Sequence[Mapping[str, Any]],
) -> Path:
    """Dependency-injected publication seam for a no-network/no-training E2E test."""

    ranking = rankings_frame(rankings, reverse_mapping)
    performance = performance_frame(performance_rows)
    pairwise = pairwise_frame(pairwise_rows)
    return publish_output_frames_atomic(output_dir, ranking, performance, pairwise)


__all__ = [
    "BACKBONES",
    "BOOTSTRAP_MINIMUM_VALID",
    "BOOTSTRAP_REPETITIONS",
    "BOOTSTRAP_SEED",
    "B6ProtocolError",
    "CALL_ORDER_SEED",
    "DATASET",
    "EVIDENCE_MODE",
    "GLOBAL_FEATURE_COUNT",
    "HOAccessGate",
    "HO_ROW_COUNT",
    "MAPPING_SEED",
    "MODEL_SNAPSHOT",
    "OUTPUT_FILENAMES",
    "PAIRWISE_COLUMNS",
    "PARTITIONS",
    "PERFORMANCE_COLUMNS",
    "PREFIX_BY_BACKBONE",
    "RANKING_BUDGET",
    "RANKING_COLUMNS",
    "TEMPERATURE",
    "align_paired_predictions",
    "assert_request_payload_safe",
    "build_global_feature_mapping",
    "build_homecredit_definition_records",
    "build_pairwise_row",
    "build_request_payload",
    "draw_paired_stratified_indices",
    "pairwise_frame",
    "obfuscate_definition_records",
    "performance_frame",
    "publish_mocked_ablation",
    "publish_output_frames_atomic",
    "rankings_frame",
    "refuse_existing_output",
    "reverse_feature_mapping",
    "scrub_candidate_records",
    "scrub_text",
    "translate_ranking_prefix",
    "validate_output_directory",
    "validate_output_frames",
    "validate_ranking_response",
]
