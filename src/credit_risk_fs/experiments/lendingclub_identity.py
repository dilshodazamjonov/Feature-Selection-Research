"""Reproducible LendingClub source identity and sidecar validation."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd

from credit_risk_fs.experiments.row_alignment import (
    ROW_ALIGNMENT_HASH_VERSION,
    normalize_alignment_value,
    ordered_row_id_sha256,
    ordered_row_id_target_sha256,
    ordered_values_sha256,
    split_alignment_summary,
    split_id_overlap_count,
)
from credit_risk_fs.preprocessing.labeling import (
    LENDINGCLUB_FINAL_BAD_STATUSES,
    LENDINGCLUB_FINAL_GOOD_STATUSES,
)
from credit_risk_fs.utils.hashing import sha256_file


IDENTITY_SCHEMA_VERSION = "lendingclub_original_loan_id_sidecar_v1"
SOURCE_RECORD_KEY_VERSION = "lc_source_record_v1"
CANONICAL_ORDERING = "time_value_int_days_asc_then_utf8_nfc_loan_id_asc"
SIDECAR_COLUMNS = (
    "loan_id",
    "split",
    "time_value",
    "target",
    "processed_row_position",
    "source_row_position",
    "issue_month",
)


@dataclass(frozen=True)
class LendingClubIdentityBundle:
    frame: pd.DataFrame
    manifest: dict[str, Any]


def source_record_key_v1(
    source_file_sha256: str,
    zero_based_source_row_number: int,
    *,
    canonical_relative_filename: str | None = None,
) -> str:
    """Return the approved fallback key; LendingClub currently does not use it."""

    if len(source_file_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in source_file_sha256.lower()
    ):
        raise ValueError("source_file_sha256 must be a 64-character hexadecimal SHA-256")
    if zero_based_source_row_number < 0:
        raise ValueError("zero_based_source_row_number must be non-negative")
    fields = [SOURCE_RECORD_KEY_VERSION, source_file_sha256.lower()]
    if canonical_relative_filename is not None:
        name = str(canonical_relative_filename).replace("\\", "/")
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("canonical_relative_filename must be traversal-free and relative")
        fields.append(name)
    fields.append(str(int(zero_based_source_row_number)))
    return hashlib.sha256("\x1f".join(fields).encode("utf-8")).hexdigest()


def canonical_loan_id(value: Any) -> str:
    """Normalize a raw LendingClub decimal ID without accepting generated values."""

    if pd.isna(value):
        raise ValueError("loan_id contains a missing value")
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    if not text.isdigit():
        raise ValueError(f"loan_id is not an unsigned decimal source ID: {text!r}")
    return str(int(text))


def _manifest_hash(manifest: dict[str, Any]) -> str:
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    ).hexdigest()


def _header_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.sha256(handle.readline().rstrip(b"\r\n")).hexdigest()


def _peak_rss_bytes() -> int:
    """Return peak working set on Windows, or best-effort POSIX max RSS."""

    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        ok = psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(),
            ctypes.byref(counters),
            counters.cb,
        )
        return int(counters.PeakWorkingSetSize) if ok else 0
    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value if value > 10_000_000 else value * 1024
    except (ImportError, AttributeError):
        return 0


def _split_name(time_value: int, *, dev_start: int, oot_start: int, oot_end: int) -> str:
    if time_value < dev_start:
        return "pre_dev"
    if time_value < oot_start:
        return "DEV"
    if time_value <= oot_end:
        return "OOT"
    return "post_oot"


def stable_chronological_order(
    frame: pd.DataFrame,
    *,
    time_column: str,
    identity_column: str,
) -> pd.DataFrame:
    """Sort chronologically and resolve every timestamp tie by canonical identity."""

    missing = {time_column, identity_column} - set(frame.columns)
    if missing:
        raise ValueError(f"stable chronological ordering columns missing: {sorted(missing)}")
    if frame[time_column].isna().any():
        raise ValueError("canonical chronological ordering forbids missing time values")
    if frame[identity_column].isna().any():
        raise ValueError("canonical chronological ordering forbids missing identities")
    ordered = frame.copy()
    ordered["__canonical_identity_order__"] = ordered[identity_column].map(
        normalize_alignment_value
    )
    ordered = ordered.sort_values(
        [time_column, "__canonical_identity_order__"], kind="mergesort"
    )
    return ordered.drop(columns="__canonical_identity_order__").reset_index(drop=True)


def _alignment_fingerprint(frame: pd.DataFrame) -> str:
    return ordered_values_sha256(
        zip(
            frame["processed_row_position"],
            frame["time_value"],
            frame["target"],
            frame["issue_month"],
            strict=True,
        )
    )


def _split_evidence(frame: pd.DataFrame, split: str) -> dict[str, Any]:
    selected = stable_chronological_order(
        frame.loc[frame["split"].eq(split), ["loan_id", "time_value", "target"]],
        time_column="time_value",
        identity_column="loan_id",
    )
    evidence = split_alignment_summary(selected["loan_id"], selected["target"])
    counts = selected.groupby("time_value", sort=False).size()
    tied = counts[counts > 1]
    evidence.update(
        {
            "equal_timestamp_group_count": int(len(tied)),
            "maximum_equal_timestamp_group_size": int(tied.max()) if len(tied) else 1,
            "time_min": int(selected["time_value"].min()),
            "time_max": int(selected["time_value"].max()),
            "ordering_rule": CANONICAL_ORDERING,
        }
    )
    return evidence


def build_lendingclub_identity_sidecar(
    *,
    raw_path: str | Path,
    processed_path: str | Path,
    sidecar_path: str | Path,
    manifest_path: str | Path,
    audit_path: str | Path | None = None,
    chunk_size: int = 100_000,
    dev_start: int = -1795,
    oot_start: int = -1065,
    oot_end: int = -730,
    overwrite_generated_outputs: bool = False,
) -> dict[str, Any]:
    """Stream raw identity into a validated sidecar without loading the feature matrix."""

    started = time.perf_counter()
    raw = Path(raw_path).resolve()
    processed = Path(processed_path).resolve()
    sidecar = Path(sidecar_path).resolve()
    manifest_file = Path(manifest_path).resolve()
    if not raw.is_file() or not processed.is_file():
        raise FileNotFoundError("raw and processed LendingClub artifacts must exist")
    if sidecar == raw or sidecar == processed:
        raise ValueError("sidecar must not overwrite a source artifact")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = sidecar.with_name(f".{sidecar.name}.{os.getpid()}.retained.tmp.csv")
    if temporary.exists() or (
        not overwrite_generated_outputs and (sidecar.exists() or manifest_file.exists())
    ):
        raise FileExistsError("identity generation is fail-closed; remove prior generated outputs first")

    final_statuses = LENDINGCLUB_FINAL_BAD_STATUSES | LENDINGCLUB_FINAL_GOOD_STATUSES
    seen_ids: set[str] = set()
    source_rows = 0
    source_data_rows = 0
    non_record_footer_rows = 0
    retained_rows = 0
    max_issue: pd.Timestamp | None = None
    first_write = True
    try:
        for chunk in pd.read_csv(
            raw,
            usecols=["id", "loan_status", "issue_d"],
            dtype={"id": "string", "loan_status": "string", "issue_d": "string"},
            chunksize=chunk_size,
            low_memory=True,
        ):
            positions = pd.Series(
                range(source_rows, source_rows + len(chunk)), index=chunk.index, dtype="int64"
            )
            source_rows += len(chunk)
            record_mask = chunk["id"].astype("string").str.fullmatch(r"\d+").fillna(False)
            invalid_records = (~record_mask) & (
                chunk["loan_status"].notna() | chunk["issue_d"].notna()
            )
            if invalid_records.any():
                raise ValueError(
                    "raw LendingClub source contains a non-decimal ID on a data-like row"
                )
            non_record_footer_rows += int((~record_mask).sum())
            source_data_rows += int(record_mask.sum())
            canonical_ids = chunk.loc[record_mask, "id"].map(canonical_loan_id)
            if canonical_ids.duplicated().any() or any(value in seen_ids for value in canonical_ids):
                raise ValueError("raw LendingClub id is not globally unique")
            seen_ids.update(canonical_ids)
            keep = record_mask & chunk["loan_status"].isin(final_statuses)
            retained = pd.DataFrame(
                {
                    "loan_id": canonical_ids.loc[keep],
                    "source_row_position": positions.loc[keep],
                    "issue_month": pd.to_datetime(
                        chunk.loc[keep, "issue_d"], format="%b-%Y", errors="coerce"
                    ),
                    "target": chunk.loc[keep, "loan_status"]
                    .isin(LENDINGCLUB_FINAL_BAD_STATUSES)
                    .astype("int8"),
                }
            )
            if retained["issue_month"].isna().any():
                raise ValueError("final-status raw rows contain an invalid issue_d")
            current_max = retained["issue_month"].max()
            if max_issue is None or current_max > max_issue:
                max_issue = current_max
            retained["issue_month"] = retained["issue_month"].dt.strftime("%Y-%m-01")
            retained.to_csv(temporary, mode="w" if first_write else "a", header=first_write, index=False)
            first_write = False
            retained_rows += len(retained)

        if max_issue is None:
            raise ValueError("raw LendingClub source has no finalized rows")

        processed_reader = pd.read_csv(
            processed,
            usecols=["TARGET", "recent_decision", "issue_d"],
            chunksize=chunk_size,
            low_memory=True,
        )
        identity_reader = pd.read_csv(
            temporary,
            dtype={"loan_id": "string", "issue_month": "string"},
            chunksize=chunk_size,
        )
        output_first = True
        processed_position = 0
        for identity_chunk, processed_chunk in zip(
            identity_reader, processed_reader, strict=True
        ):
            if len(identity_chunk) != len(processed_chunk):
                raise ValueError("raw-retained and processed chunk row counts differ")
            derived_time = (
                pd.to_datetime(identity_chunk["issue_month"], format="%Y-%m-%d") - max_issue
            ).dt.days.astype("int64")
            observed_time = pd.to_numeric(processed_chunk["recent_decision"], errors="raise").astype(
                "int64"
            )
            observed_target = pd.to_numeric(processed_chunk["TARGET"], errors="raise").astype(
                "int8"
            )
            observed_issue = pd.to_datetime(
                processed_chunk["issue_d"], format="%Y-%m-%d", errors="coerce"
            ).dt.strftime("%Y-%m-01")
            if not derived_time.reset_index(drop=True).equals(observed_time.reset_index(drop=True)):
                raise ValueError("processed recent_decision is misaligned with raw issue_d")
            if not identity_chunk["target"].astype("int8").reset_index(drop=True).equals(
                observed_target.reset_index(drop=True)
            ):
                raise ValueError("processed TARGET is misaligned with raw loan_id")
            if not identity_chunk["issue_month"].reset_index(drop=True).equals(
                observed_issue.astype("string").reset_index(drop=True)
            ):
                raise ValueError("processed issue_d is misaligned with raw loan_id")
            output = pd.DataFrame(
                {
                    "loan_id": identity_chunk["loan_id"].astype("string"),
                    "split": [
                        _split_name(
                            int(value),
                            dev_start=dev_start,
                            oot_start=oot_start,
                            oot_end=oot_end,
                        )
                        for value in derived_time
                    ],
                    "time_value": derived_time,
                    "target": identity_chunk["target"].astype("int8"),
                    "processed_row_position": range(
                        processed_position, processed_position + len(identity_chunk)
                    ),
                    "source_row_position": identity_chunk["source_row_position"].astype("int64"),
                    "issue_month": identity_chunk["issue_month"].astype("string"),
                }
            )
            output.to_csv(sidecar, mode="w" if output_first else "a", header=output_first, index=False)
            output_first = False
            processed_position += len(output)
        if processed_position != retained_rows:
            raise ValueError(
                f"processed row count {processed_position} differs from retained raw rows {retained_rows}"
            )

        identity = pd.read_csv(
            sidecar,
            dtype={"loan_id": "string", "split": "string", "issue_month": "string"},
        )
        if tuple(identity.columns) != SIDECAR_COLUMNS:
            raise ValueError("generated identity sidecar schema mismatch")
        if identity["loan_id"].isna().any() or identity["loan_id"].duplicated().any():
            raise ValueError("generated identity sidecar IDs are null or duplicated")
        dev = identity.loc[identity["split"].eq("DEV")]
        oot = identity.loc[identity["split"].eq("OOT")]
        split_evidence = {"DEV": _split_evidence(identity, "DEV"), "OOT": _split_evidence(identity, "OOT")}
        raw_hash = sha256_file(raw)
        processed_hash = sha256_file(processed)
        sidecar_hash = sha256_file(sidecar)
        manifest: dict[str, Any] = {
            "schema_version": IDENTITY_SCHEMA_VERSION,
            "identity_type": "authenticated_original_loan_id",
            "stable_row_id_column": "loan_id",
            "raw_source": {
                "path": raw.as_posix(),
                "sha256": raw_hash,
                "size_bytes": raw.stat().st_size,
                "header_sha256": _header_sha256(raw),
            },
            "processed_dataset": {
                "path": processed.as_posix(),
                "sha256": processed_hash,
                "size_bytes": processed.stat().st_size,
                "header_sha256": _header_sha256(processed),
                "alignment_fingerprint": _alignment_fingerprint(identity),
            },
            "sidecar": {
                "path": sidecar.as_posix(),
                "sha256": sidecar_hash,
                "size_bytes": sidecar.stat().st_size,
                "columns": list(SIDECAR_COLUMNS),
            },
            "source_row_count": source_rows,
            "source_data_row_count": source_data_rows,
            "non_record_footer_row_count": non_record_footer_rows,
            "retained_row_count": retained_rows,
            "stable_id_unique_count": int(identity["loan_id"].nunique()),
            "stable_id_null_count": int(identity["loan_id"].isna().sum()),
            "stable_id_uniqueness_status": "unique",
            "source_stable_id_values_hash": ordered_row_id_sha256(
                sorted(identity["loan_id"].astype(str))
            ),
            "stable_id_target_alignment_hash": ordered_row_id_target_sha256(
                *zip(
                    *sorted(
                        zip(identity["loan_id"].astype(str), identity["target"], strict=True)
                    )
                )
            ),
            "dev_oot_overlap_count": split_id_overlap_count(dev["loan_id"], oot["loan_id"]),
            "splits": split_evidence,
            "canonical_serialization_version": ROW_ALIGNMENT_HASH_VERSION,
            "canonical_ordering": CANONICAL_ORDERING,
            "split_boundaries": {
                "dev_start_inclusive": dev_start,
                "oot_start_inclusive": oot_start,
                "oot_end_inclusive": oot_end,
            },
            "creation_code_version": "build_lendingclub_identity_sidecar_v1",
            "regeneration_command": (
                ".\\.venv\\Scripts\\python.exe scripts\\build_lendingclub_identity_sidecar.py"
            ),
            "resource_usage": {
                "elapsed_seconds": round(time.perf_counter() - started, 6),
                "peak_rss_bytes": _peak_rss_bytes(),
                "process_count": 1,
                "chunk_size": chunk_size,
                "files_read": [raw.as_posix(), processed.as_posix()],
                "columns_read": {
                    raw.as_posix(): ["id", "loan_status", "issue_d"],
                    processed.as_posix(): ["TARGET", "recent_decision", "issue_d"],
                },
                "bytes_written": sidecar.stat().st_size,
            },
        }
        manifest["manifest_sha256"] = _manifest_hash(manifest)
        manifest_file.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        if audit_path is not None:
            audit = Path(audit_path).resolve()
            audit.parent.mkdir(parents=True, exist_ok=True)
            audit.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return manifest
    finally:
        if temporary.exists():
            temporary.unlink()


def load_lendingclub_identity_sidecar(
    *,
    sidecar_path: str | Path,
    manifest_path: str | Path,
    processed_frame: pd.DataFrame | None = None,
    memory_gate: Callable[[str], None] | None = None,
    csv_chunk_rows: int = 25_000,
) -> LendingClubIdentityBundle:
    """Load a sidecar and fail closed on schema, hashes, order, target, or time."""

    sidecar = Path(sidecar_path).resolve()
    manifest_file = Path(manifest_path).resolve()
    if not sidecar.is_file() or not manifest_file.is_file():
        raise FileNotFoundError("LendingClub identity sidecar and manifest are required")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != IDENTITY_SCHEMA_VERSION:
        raise ValueError("LendingClub identity schema version mismatch")
    if manifest.get("manifest_sha256") != _manifest_hash(manifest):
        raise ValueError("LendingClub identity manifest hash mismatch")
    expected_sidecar = manifest.get("sidecar", {})
    if memory_gate is not None:
        memory_gate("lendingclub_identity:before_hash_validation")
    if expected_sidecar.get("sha256") != sha256_file(sidecar):
        raise ValueError("LendingClub identity sidecar hash mismatch")
    if int(csv_chunk_rows) <= 0:
        raise ValueError("csv_chunk_rows must be positive")
    reader = iter(
        pd.read_csv(
            sidecar,
            dtype={
                "loan_id": "string",
                "split": "string",
                "issue_month": "string",
            },
            chunksize=int(csv_chunk_rows),
        )
    )
    chunks: list[pd.DataFrame] = []
    chunk_index = 0
    while True:
        if memory_gate is not None:
            memory_gate(f"lendingclub_identity:before_csv_chunk:{chunk_index}")
        try:
            chunks.append(next(reader))
        except StopIteration:
            break
        chunk_index += 1
    if memory_gate is not None:
        memory_gate("lendingclub_identity:before_chunk_concat")
    identity = pd.concat(chunks, ignore_index=True)
    if tuple(identity.columns) != SIDECAR_COLUMNS:
        raise ValueError("LendingClub identity sidecar columns mismatch")
    expected_rows = int(manifest.get("retained_row_count", -1))
    if len(identity) != expected_rows:
        raise ValueError("LendingClub identity sidecar row count mismatch")
    expected_positions = pd.Series(range(len(identity)), dtype="int64")
    if not identity["processed_row_position"].reset_index(drop=True).equals(expected_positions):
        raise ValueError("LendingClub identity sidecar processed row ordering mismatch")
    if identity["loan_id"].isna().any() or identity["loan_id"].duplicated().any():
        raise ValueError("LendingClub identity sidecar IDs are null or duplicated")
    if _alignment_fingerprint(identity) != manifest["processed_dataset"]["alignment_fingerprint"]:
        raise ValueError("LendingClub identity alignment fingerprint mismatch")
    if processed_frame is not None:
        if memory_gate is not None:
            memory_gate("lendingclub_identity:before_frame_alignment")
        required = {"TARGET", "recent_decision", "issue_d"}
        missing = required - set(processed_frame.columns)
        if missing:
            raise ValueError(f"processed LendingClub frame lacks alignment columns: {sorted(missing)}")
        if len(processed_frame) != len(identity):
            raise ValueError("LendingClub sidecar/dataset row count mismatch")
        targets = pd.to_numeric(processed_frame["TARGET"], errors="raise").astype("int64")
        times = pd.to_numeric(processed_frame["recent_decision"], errors="raise").astype("int64")
        issues = pd.to_datetime(
            processed_frame["issue_d"], errors="coerce"
        ).dt.strftime("%Y-%m-01").astype("string")
        if not targets.reset_index(drop=True).equals(identity["target"].astype("int64").reset_index(drop=True)):
            raise ValueError("LendingClub sidecar/dataset target mismatch")
        if not times.reset_index(drop=True).equals(identity["time_value"].astype("int64").reset_index(drop=True)):
            raise ValueError("LendingClub sidecar/dataset time mismatch")
        if not issues.reset_index(drop=True).equals(identity["issue_month"].reset_index(drop=True)):
            raise ValueError("LendingClub sidecar/dataset row ordering mismatch")
    return LendingClubIdentityBundle(frame=identity, manifest=manifest)


__all__ = [
    "CANONICAL_ORDERING",
    "IDENTITY_SCHEMA_VERSION",
    "LendingClubIdentityBundle",
    "build_lendingclub_identity_sidecar",
    "canonical_loan_id",
    "load_lendingclub_identity_sidecar",
    "source_record_key_v1",
    "stable_chronological_order",
]
