"""Machine-stable ordered row-alignment hashes for scientific split contracts."""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from decimal import Decimal, InvalidOperation
from numbers import Integral, Real
from typing import Any, Iterable


ROW_ALIGNMENT_HASH_VERSION = "credit_risk_ordered_row_alignment_v1"


def normalize_alignment_value(value: Any) -> str:
    """Return a typed, Unicode-normalized scalar independent of pandas hashing."""

    if value is None or type(value).__name__ == "NAType":
        return "null"
    try:
        if value != value:  # NaN and pandas.NA-like values.
            return "null"
    except (TypeError, ValueError):
        pass
    if isinstance(value, bool):
        return "bool:true" if value else "bool:false"
    if isinstance(value, Integral):
        return f"number:{int(value)}"
    if isinstance(value, Real):
        if not math.isfinite(float(value)):
            raise ValueError("alignment values must not contain infinity")
        value = Decimal(str(float(value)))
    if isinstance(value, Decimal):
        try:
            normalized = value.normalize()
        except InvalidOperation as exc:
            raise ValueError(f"invalid decimal alignment value: {value}") from exc
        text = format(normalized, "f")
        if text == "-0":
            text = "0"
        return f"number:{text}"
    return "string:" + unicodedata.normalize("NFC", str(value))


def _ordered_digest(rows: Iterable[tuple[Any, ...]]) -> str:
    digest = hashlib.sha256()
    digest.update((ROW_ALIGNMENT_HASH_VERSION + "\n").encode("utf-8"))
    for row in rows:
        payload = [normalize_alignment_value(value) for value in row]
        digest.update(
            (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
                "utf-8"
            )
        )
    return digest.hexdigest()


def ordered_values_sha256(rows: Iterable[tuple[Any, ...]]) -> str:
    """Hash ordered scalar tuples using the canonical alignment serialization."""

    return _ordered_digest(rows)


def ordered_row_id_sha256(row_ids: Iterable[Any]) -> str:
    """Hash row IDs in their supplied order."""

    return _ordered_digest((row_id,) for row_id in row_ids)


def ordered_row_id_target_sha256(
    row_ids: Iterable[Any], targets: Iterable[Any]
) -> str:
    """Hash ordered row-ID/target pairs and reject length mismatches."""

    ids = list(row_ids)
    target_values = list(targets)
    if len(ids) != len(target_values):
        raise ValueError("row ID and target counts differ")
    return _ordered_digest(zip(ids, target_values, strict=True))


def split_alignment_summary(
    row_ids: Iterable[Any], targets: Iterable[Any]
) -> dict[str, Any]:
    """Return counts and ordered hashes without retaining an ID-list artifact."""

    ids = list(row_ids)
    target_values = list(targets)
    if len(ids) != len(target_values):
        raise ValueError("row ID and target counts differ")
    normalized_ids = [normalize_alignment_value(value) for value in ids]
    missing = sum(value == "null" for value in normalized_ids)
    return {
        "hash_version": ROW_ALIGNMENT_HASH_VERSION,
        "row_count": len(ids),
        "unique_id_count": len(set(normalized_ids)) - int(missing > 0),
        "missing_id_count": missing,
        "duplicate_id_count": len(ids) - missing - (len(set(normalized_ids)) - int(missing > 0)),
        "positive_count": sum(normalize_alignment_value(value) == "number:1" for value in target_values),
        "positive_rate": (
            sum(normalize_alignment_value(value) == "number:1" for value in target_values)
            / len(target_values)
            if target_values
            else None
        ),
        "ordered_row_id_sha256": ordered_row_id_sha256(ids),
        "ordered_row_id_target_sha256": ordered_row_id_target_sha256(ids, target_values),
    }


def split_id_overlap_count(left_ids: Iterable[Any], right_ids: Iterable[Any]) -> int:
    """Count normalized identifiers present in both splits, excluding nulls."""

    left = {normalize_alignment_value(value) for value in left_ids} - {"null"}
    right = {normalize_alignment_value(value) for value in right_ids} - {"null"}
    return len(left & right)
