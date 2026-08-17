"""Validate the Prompt-16 PSI optimization without running the OOT workload."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _legacy_categorical_result(
    dev: pd.Series, oot: pd.Series
) -> tuple[float, pd.DataFrame, int]:
    from credit_risk_fs.analysis.voting_inference.psi import (
        MISSING_STATE,
        UNSEEN_STATE,
        _share_table,
    )

    dev_series = dev.astype("object").map(
        lambda value: MISSING_STATE if pd.isna(value) else str(value)
    )
    oot_series = oot.astype("object").map(
        lambda value: MISSING_STATE if pd.isna(value) else str(value)
    )
    dev_levels = sorted(set(dev_series.unique()))
    unseen = sorted(set(oot_series.unique()) - set(dev_levels))
    oot_series = oot_series.map(
        lambda value: UNSEEN_STATE if value in set(unseen) else value
    )
    frame, psi = _share_table(
        dev_series,
        oot_series,
        [*dev_levels, UNSEEN_STATE],
    )
    return psi, frame, len(unseen)


def _equivalence_fixtures() -> list[dict[str, Any]]:
    from credit_risk_fs.analysis.voting_inference.psi import type_aware_feature_psi

    fixtures = {
        "object_categories": (
            pd.Series(["a", None, "b", "a", pd.NA], dtype="object"),
            pd.Series(["b", "c", None, "a", "c"], dtype="object"),
        ),
        "nullable_integer": (
            pd.Series([1, 2, None, 2, 1], dtype="Int64"),
            pd.Series([1, 3, None, 3, 2], dtype="Int64"),
        ),
        "nullable_boolean": (
            pd.Series([True, False, None, True], dtype="boolean"),
            pd.Series([False, None, True, False], dtype="boolean"),
        ),
        "constant_float": (
            pd.Series([1.25, np.nan, 1.25, 1.25]),
            pd.Series([2.5, np.nan, 1.25, 2.5]),
        ),
    }
    results: list[dict[str, Any]] = []
    for name, (dev, oot) in fixtures.items():
        expected_psi, expected_frame, expected_unseen = _legacy_categorical_result(
            dev, oot
        )
        observed_psi, observed_frame, definition = type_aware_feature_psi(dev, oot)
        if observed_psi != expected_psi:
            raise AssertionError(f"{name}: PSI float changed")
        pd.testing.assert_frame_equal(observed_frame, expected_frame, check_exact=True)
        if definition["unseen_oot_level_count"] != expected_unseen:
            raise AssertionError(f"{name}: unseen level count changed")
        results.append(
            {
                "fixture": name,
                "psi": observed_psi,
                "distribution_rows": len(observed_frame),
                "exact_frame_match": True,
                "exact_float_match": True,
            }
        )
    return results


def _parallel_equivalence() -> dict[str, Any]:
    from credit_risk_fs.experiments.prompt_16_third_dataset import (
        FEATURE_PSI_MAX_WORKERS,
        _calculate_feature_psi_batch,
    )

    generator = np.random.default_rng(160805)
    dev_rows = 20_000
    oot_rows = 5_000
    predictors = [f"numeric_{index:02d}" for index in range(16)]
    train = pd.DataFrame(
        {name: generator.normal(size=dev_rows) for name in predictors}
    )
    validation = pd.DataFrame(
        {name: generator.normal(size=oot_rows) for name in predictors}
    )
    started = time.perf_counter()
    serial = _calculate_feature_psi_batch(
        train_batch=train,
        validation_batch=validation,
        predictors=predictors,
        max_workers=1,
    )
    serial_seconds = time.perf_counter() - started
    started = time.perf_counter()
    parallel = _calculate_feature_psi_batch(
        train_batch=train,
        validation_batch=validation,
        predictors=predictors,
        max_workers=FEATURE_PSI_MAX_WORKERS,
    )
    parallel_seconds = time.perf_counter() - started
    pd.testing.assert_frame_equal(
        pd.DataFrame(parallel),
        pd.DataFrame(serial),
        check_exact=True,
    )
    return {
        "feature_count": len(predictors),
        "dev_rows": dev_rows,
        "oot_rows": oot_rows,
        "parallel_worker_count": FEATURE_PSI_MAX_WORKERS,
        "serial_seconds": serial_seconds,
        "parallel_seconds": parallel_seconds,
        "ordered_records_exact_match": True,
    }


def _pathological_fixture() -> dict[str, Any]:
    from credit_risk_fs.analysis.voting_inference.psi import (
        MISSING_STATE,
        UNSEEN_STATE,
        type_aware_feature_psi,
    )

    dev_rows = 1_221_743
    oot_rows = 304_916
    unseen_levels = 136_290
    dev = pd.Series(np.full(dev_rows, np.nan, dtype="float64"))
    oot = pd.Series(
        np.resize(np.arange(unseen_levels, dtype="float64"), oot_rows)
    )
    started = time.perf_counter()
    psi, distribution, definition = type_aware_feature_psi(dev, oot)
    elapsed = time.perf_counter() - started
    if definition["unseen_oot_level_count"] != unseen_levels:
        raise AssertionError("pathological unseen-level count changed")
    if list(distribution["state"]) != [MISSING_STATE, UNSEEN_STATE]:
        raise AssertionError("pathological state ordering changed")
    if distribution["dev_count"].tolist() != [dev_rows, 0]:
        raise AssertionError("pathological DEV counts changed")
    if distribution["oot_count"].tolist() != [0, oot_rows]:
        raise AssertionError("pathological OOT counts changed")
    return {
        "dev_rows": dev_rows,
        "oot_rows": oot_rows,
        "unseen_oot_level_count": unseen_levels,
        "elapsed_seconds": elapsed,
        "psi": psi,
        "all_rows_assigned": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation-commit", required=True)
    args = parser.parse_args(argv)

    from credit_risk_fs.data.homecredit_model_stability_2024.contract import (
        file_sha256,
    )
    from credit_risk_fs.experiments.atomic_io import (
        write_json_atomic,
        write_text_atomic,
    )
    from credit_risk_fs.experiments.prompt_16_final_oot import (
        PSI_PERFORMANCE_AMENDMENT_RELATIVE_ROOT,
        _validate_psi_performance_predecessor,
    )
    from credit_risk_fs.experiments.prompt_16_third_dataset import (
        FEATURE_PSI_MAX_WORKERS,
    )

    implementation_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if args.implementation_commit != implementation_commit:
        raise RuntimeError("validation must bind the committed HEAD")
    if subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip():
        raise RuntimeError("validation requires a clean worktree")

    audit_root = PROJECT_ROOT / PSI_PERFORMANCE_AMENDMENT_RELATIVE_ROOT
    if audit_root.exists():
        raise RuntimeError(f"validation root already exists: {audit_root}")
    audit_root.mkdir(parents=True)
    fit_ids, batches, artifacts, _ = _validate_psi_performance_predecessor(
        PROJECT_ROOT
    )
    report = {
        "schema_version": "prompt_16_final_oot_psi_performance_validation_v5",
        "status": "validated_exact_equivalence",
        "implementation_commit": implementation_commit,
        "created_at_utc": _utc_now(),
        "scientific_semantics_changed": False,
        "categorical_equivalence_fixtures": _equivalence_fixtures(),
        "parallel_equivalence": _parallel_equivalence(),
        "pathological_full_shape_fixture": _pathological_fixture(),
        "parallel_worker_count": FEATURE_PSI_MAX_WORKERS,
        "completed_selection_fit_count": len(fit_ids),
        "completed_selection_fit_ids": fit_ids,
        "sealed_psi_batches": batches,
        "first_incomplete_psi_batch": 5,
        "authenticated_predecessor_artifact_count": len(artifacts),
        "oot_workload_executed": False,
        "rows_sampled_or_removed": 0,
        "features_removed": 0,
    }
    report_path = audit_root / "psi_performance_equivalence.json"
    write_json_atomic(report_path, report, overwrite=False)
    write_text_atomic(
        audit_root / "_VALIDATION_SUCCESS",
        json.dumps({"report_sha256": file_sha256(report_path)}, sort_keys=True) + "\n",
        overwrite=False,
    )
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
