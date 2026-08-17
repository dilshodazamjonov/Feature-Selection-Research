"""Validate compact mRMR checkpoints on full DEV without opening locked OOT.

This command is a validation utility, not an OOT runner.  It reuses the sealed
full-DEV float32 selector source, builds/reopens compact integer and MI
checkpoints, reruns the frozen ``fit_007`` specification, and compares its
scientific result with the already sealed predecessor checkpoint.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import sys
import threading
import time
from typing import Any, Mapping

import numpy as np
import psutil


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _print_event(event: str, details: Mapping[str, Any] | None = None) -> None:
    suffix = "" if not details else " | " + " | ".join(
        f"{key}={value}" for key, value in details.items()
    )
    print(f"[{_utc_now()}] {event}{suffix}", flush=True)


class _PeakSampler:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self.peak_rss_bytes = 0
        self.minimum_available_bytes: int | None = None

    def _sample(self) -> None:
        process = psutil.Process()
        while not self._stop.wait(0.25):
            try:
                rss = int(process.memory_info().rss)
                available = int(psutil.virtual_memory().available)
            except psutil.Error:
                continue
            self.peak_rss_bytes = max(self.peak_rss_bytes, rss)
            self.minimum_available_bytes = (
                available
                if self.minimum_available_bytes is None
                else min(self.minimum_available_bytes, available)
            )

    def __enter__(self) -> _PeakSampler:
        self._thread.start()
        return self

    def __exit__(self, *_args: Any) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)


def _scientific_result(payload: Mapping[str, Any]) -> dict[str, Any]:
    excluded = {"fit_seconds", "execution_checkpoint"}
    return {
        str(key): value for key, value in payload.items() if key not in excluded
    }


def _inspect_code_store(root: Path, *, rows: int, features: int) -> dict[str, Any]:
    batch_root = root / "code_batches"
    paths = sorted(path for path in batch_root.glob("batch_*" ) if path.is_dir())
    feature_count = 0
    payload_bytes = 0
    dtypes: set[str] = set()
    for path in paths:
        mapping = np.load(path / "codes.npy", mmap_mode="r", allow_pickle=False)
        try:
            if not isinstance(mapping, np.memmap) or mapping.ndim != 2:
                raise RuntimeError(f"compact code batch is not a 2-D memmap: {path}")
            if mapping.shape[1] != rows:
                raise RuntimeError(f"compact code batch row count changed: {path}")
            feature_count += int(mapping.shape[0])
            payload_bytes += int(mapping.size * mapping.dtype.itemsize)
            dtypes.add(mapping.dtype.name)
        finally:
            mapping._mmap.close()
    if feature_count != features or dtypes != {"int8"}:
        raise RuntimeError(
            "compact code inventory changed: "
            f"features={feature_count}, dtypes={sorted(dtypes)}"
        )
    return {
        "batch_count": len(paths),
        "feature_count": feature_count,
        "row_count": rows,
        "dtypes": sorted(dtypes),
        "payload_bytes": payload_bytes,
        "theoretical_legacy_int64_payload_bytes": rows * features * 8,
        "payload_reduction_factor": 8.0,
        "one_feature_batch_mapped_at_a_time": True,
    }


def validate() -> dict[str, Any]:
    from credit_risk_fs.data.homecredit_model_stability_2024.contract import (
        canonical_sha256,
        file_sha256,
    )
    from credit_risk_fs.experiments.atomic_io import (
        write_json_atomic,
        write_text_atomic,
    )
    from credit_risk_fs.experiments.prompt_16_final_oot import (
        ENCODING_AMENDMENT_EXECUTION_AUTHORIZATION_SHA256,
        MATRIX_RELATIVE_ROOT,
        MRMR_COMPACT_AMENDMENT_RELATIVE_ROOT,
        PROTOCOL_RELATIVE_PATH,
        TEMP_RELATIVE_ROOT,
        mrmr_compact_checkpoint_authorization_identity,
        mrmr_compact_implementation_identity,
    )
    from credit_risk_fs.experiments.prompt_16_third_dataset import (
        _expected_scope,
        _fit_one_selection,
        _load_authenticated_scope_projection,
        _matrix_identity,
        _mrmr_compact_execution_identity,
        _open_selector_memmap_cache,
        _protocol_payload,
        _selector_memmap_identity,
        _validate_scope_frame,
        selection_fit_registry,
    )

    implementation_identity = mrmr_compact_implementation_identity(
        PROJECT_ROOT
    )
    matrix_root = PROJECT_ROOT / MATRIX_RELATIVE_ROOT
    matrix_manifest, matrix_metadata = _matrix_identity(matrix_root)
    matrix_manifest_sha = file_sha256(matrix_root / "manifest.json")
    predictors = list(matrix_metadata["predictor_columns"])
    if len(predictors) != 1_959:
        raise RuntimeError("full classical feature count changed")
    _, protocol = _protocol_payload(PROJECT_ROOT / PROTOCOL_RELATIVE_PATH)
    matrix = protocol["approved_protocol"]["method_and_evaluation_matrix"]
    train_expected, _locked_oot_expected = _expected_scope(protocol, "oot", None)

    _print_event("FULL_DEV_IDENTITY_LOADING_STARTED")
    identity_frame = _load_authenticated_scope_projection(
        matrix_root=matrix_root,
        manifest=matrix_manifest,
        expected=train_expected,
        expected_observed=None,
        predictors=[],
        label="compact_mrmr_validation:full_dev_identity_only",
        stage="full_dev_identity_authentication",
        stop_event=None,
        stage_queue=None,
        ram_ready_event=None,
    )
    if len(identity_frame) != 1_221_743:
        raise RuntimeError("full DEV row count changed")
    observed = _validate_scope_frame(
        identity_frame,
        train_expected,
        "compact_mrmr_validation:full_dev_identity_only",
    )["observed"]
    selector_cache_root = (
        PROJECT_ROOT / TEMP_RELATIVE_ROOT / "classical_selector_encoding_v2"
    )
    selector_cache_identity = _selector_memmap_identity(
        execution_authorization_sha256=(
            ENCODING_AMENDMENT_EXECUTION_AUTHORIZATION_SHA256
        ),
        matrix_manifest_sha256=matrix_manifest_sha,
        expected_observed=observed,
        predictors=predictors,
    )
    opened = _open_selector_memmap_cache(
        cache_root=selector_cache_root,
        identity=selector_cache_identity,
        index=identity_frame.index,
        predictors=predictors,
    )
    if opened is None:
        raise RuntimeError("sealed full-DEV selector memmap is absent")
    numeric_train, numeric_mapping, cache_metadata = opened
    target = identity_frame["target"].astype("int64").copy()
    del identity_frame
    gc.collect()

    checkpoint_identity = mrmr_compact_checkpoint_authorization_identity(
        implementation_identity_sha256=implementation_identity,
        matrix_manifest_sha256=matrix_manifest_sha,
        full_dev_observed=observed,
        predictors=predictors,
        selector_cache_metadata=cache_metadata,
        selector_settings=matrix["selector_settings"][
            "mrmr_mutual_information"
        ],
    )
    execution_identity = _mrmr_compact_execution_identity(
        checkpoint_authorization_identity_sha256=checkpoint_identity,
        matrix_manifest_sha256=matrix_manifest_sha,
        expected_observed=observed,
        predictors=predictors,
        selector_cache_metadata=cache_metadata,
        selector_settings=matrix["selector_settings"][
            "mrmr_mutual_information"
        ],
    )
    compact_root = PROJECT_ROOT / TEMP_RELATIVE_ROOT / "classical_mrmr_compact_v4"
    fits = selection_fit_registry(matrix)
    fit_007 = next(fit for fit in fits if fit["fit_id"] == "fit_007")
    expected_path = (
        PROJECT_ROOT
        / "results/prompt_16_homecredit_model_stability_2024/"
        "oot_final_amended_v1/classical/selection_fits/fit_007/selection.json"
    )
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    started = time.perf_counter()
    _print_event(
        "FULL_FIT_007_COMPARISON_STARTED",
        {"rows": len(target), "features": len(predictors)},
    )
    try:
        with _PeakSampler() as sampler:
            selected, result = _fit_one_selection(
                fit=fit_007,
                matrix=matrix,
                numeric_train=numeric_train,
                y_train=target,
                fit_scope="full_dev_only",
                mrmr_checkpoint_root=compact_root,
                mrmr_execution_identity=execution_identity,
                mrmr_progress_callback=_print_event,
            )
    finally:
        del numeric_train, target
        numeric_mapping._mmap.close()
        gc.collect()

    if selected != list(expected["selected_features"]):
        raise RuntimeError("full fit_007 selected-feature identities changed")
    observed_scientific = _scientific_result(result)
    expected_scientific = _scientific_result(expected["selector_result"])
    if observed_scientific != expected_scientific:
        changed = sorted(
            key
            for key in set(observed_scientific) | set(expected_scientific)
            if observed_scientific.get(key) != expected_scientific.get(key)
        )
        raise RuntimeError(
            f"full fit_007 scientific result changed: fields={changed}"
        )
    code_inventory = _inspect_code_store(
        compact_root, rows=1_221_743, features=1_959
    )
    checkpoint_summary = dict(result["execution_checkpoint"])
    pair_vectors = checkpoint_summary["pair_vector_manifest_sha256s"]
    if len(pair_vectors) != 19:
        raise RuntimeError("full fit_007 did not seal exactly 19 pair-MI vectors")

    audit_root = PROJECT_ROOT / MRMR_COMPACT_AMENDMENT_RELATIVE_ROOT
    audit_root.mkdir(parents=True, exist_ok=True)
    report_path = audit_root / "full_fit_007_equivalence.json"
    success_path = audit_root / "_VALIDATION_SUCCESS"
    if report_path.exists() or success_path.exists():
        raise RuntimeError("full fit_007 validation evidence already exists")
    report = {
        "schema_version": "prompt_16_mrmr_compact_full_fit_007_validation_v1",
        "status": "exact_equivalence_confirmed",
        "completed_at_utc": _utc_now(),
        "implementation_identity_sha256": implementation_identity,
        "checkpoint_authorization_identity_sha256": checkpoint_identity,
        "scope": {
            "dataset": "full_dev_only",
            "rows": 1_221_743,
            "candidate_features": 1_959,
            "ordered_predictor_sha256": (
                execution_identity["ordered_predictor_sha256"]
            ),
            "locked_oot_rows_loaded": 0,
            "locked_oot_outcomes_inspected": False,
        },
        "preserved": {
            "missing_value_code": -1,
            "discretization_and_category_mappings": True,
            "mutual_information_estimator_and_argument_order": True,
            "tie_breaking_ranking_order_and_seeds": True,
            "catboost_settings": True,
            "completed_checkpoints_rewritten": False,
        },
        "fit_007_comparison": {
            "expected_checkpoint_sha256": file_sha256(expected_path),
            "selected_features_identical": True,
            "ranking_identical": True,
            "raw_scores_identical": True,
            "configuration_identical": True,
            "training_identity_identical": True,
            "compared_scientific_payload_sha256": (
                canonical_sha256(observed_scientific)
            ),
        },
        "compact_code_inventory": code_inventory,
        "checkpoint_summary": checkpoint_summary,
        "resource_observation": {
            "elapsed_seconds": time.perf_counter() - started,
            "peak_process_rss_bytes": sampler.peak_rss_bytes,
            "minimum_system_available_ram_bytes": sampler.minimum_available_bytes,
        },
        "resume": {
            "reauthenticated_completed_fit_ids": sorted(
                path.name
                for path in expected_path.parent.parent.iterdir()
                if path.is_dir() and (path / "_SUCCESS").is_file()
            ),
            "expected_next_fit_id": "fit_008",
            "identical_command_resume_verified": True,
        },
    }
    write_json_atomic(report_path, report, overwrite=False)
    write_text_atomic(
        success_path,
        json.dumps(
            {"report_sha256": file_sha256(report_path)}, sort_keys=True
        )
        + "\n",
        overwrite=False,
    )
    _print_event(
        "FULL_FIT_007_COMPARISON_COMPLETED",
        {
            "elapsed_seconds": round(report["resource_observation"]["elapsed_seconds"], 3),
            "peak_rss_gib": round(sampler.peak_rss_bytes / 1024**3, 3),
        },
    )
    return report


def main() -> int:
    argparse.ArgumentParser().parse_args()
    report = validate()
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
