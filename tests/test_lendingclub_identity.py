from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from credit_risk_fs.experiments.lendingclub_identity import (
    SOURCE_RECORD_KEY_VERSION,
    load_lendingclub_identity_sidecar,
    source_record_key_v1,
    stable_chronological_order,
)
from credit_risk_fs.experiments.row_alignment import (
    ordered_row_id_sha256,
    ordered_row_id_target_sha256,
    ordered_values_sha256,
)
from credit_risk_fs.utils.hashing import sha256_file


def test_source_record_fallback_key_is_exact_reproducible_and_target_free():
    source_hash = "ab" * 32
    expected = hashlib.sha256(
        f"{SOURCE_RECORD_KEY_VERSION}\x1f{source_hash}\x1f7".encode("utf-8")
    ).hexdigest()
    assert source_record_key_v1(source_hash, 7) == expected
    assert source_record_key_v1(source_hash, 7) == expected
    # No API accepts a target or prediction input; only immutable source coordinates.
    with pytest.raises(TypeError):
        source_record_key_v1(source_hash, 7, target=1)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="traversal-free"):
        source_record_key_v1(source_hash, 7, canonical_relative_filename="../raw.csv")


def test_timestamp_ties_are_ordered_by_canonical_identity():
    frame = pd.DataFrame(
        {"time": [-1, -2, -1, -1], "loan_id": ["20", "9", "10", "2"]}
    )
    first = stable_chronological_order(
        frame, time_column="time", identity_column="loan_id"
    )
    second = stable_chronological_order(
        frame.sample(frac=1, random_state=4),
        time_column="time",
        identity_column="loan_id",
    )
    assert first.equals(second)
    assert first["loan_id"].tolist() == ["9", "10", "2", "20"]


def test_order_and_target_hash_contracts_change_on_mutation():
    ids = ["a", "b", "c"]
    assert ordered_row_id_sha256(ids) != ordered_row_id_sha256(ids[::-1])
    assert ordered_row_id_target_sha256(ids, [0, 1, 0]) != (
        ordered_row_id_target_sha256(ids, [0, 0, 0])
    )


def _write_tiny_sidecar(tmp_path):
    sidecar = tmp_path / "identity.csv"
    manifest_path = tmp_path / "identity.manifest.json"
    frame = pd.DataFrame(
        {
            "loan_id": ["10", "20"],
            "split": ["DEV", "OOT"],
            "time_value": [-2, -1],
            "target": [0, 1],
            "processed_row_position": [0, 1],
            "source_row_position": [3, 7],
            "issue_month": ["2018-10-01", "2018-11-01"],
        }
    )
    frame.to_csv(sidecar, index=False)
    fingerprint = ordered_values_sha256(
        zip(
            frame["processed_row_position"],
            frame["time_value"],
            frame["target"],
            frame["issue_month"],
            strict=True,
        )
    )
    manifest = {
        "schema_version": "lendingclub_original_loan_id_sidecar_v1",
        "retained_row_count": 2,
        "processed_dataset": {"alignment_fingerprint": fingerprint},
        "sidecar": {"sha256": sha256_file(sidecar)},
    }
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    manifest["manifest_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return sidecar, manifest_path


def test_sidecar_validates_dataset_row_order_and_target(tmp_path):
    sidecar, manifest = _write_tiny_sidecar(tmp_path)
    processed = pd.DataFrame(
        {
            "TARGET": [0, 1],
            "recent_decision": [-2, -1],
            "issue_d": ["2018-10-01", "2018-11-01"],
        }
    )
    bundle = load_lendingclub_identity_sidecar(
        sidecar_path=sidecar, manifest_path=manifest, processed_frame=processed
    )
    assert bundle.frame["loan_id"].tolist() == ["10", "20"]

    with pytest.raises(ValueError, match="target mismatch"):
        load_lendingclub_identity_sidecar(
            sidecar_path=sidecar,
            manifest_path=manifest,
            processed_frame=processed.assign(TARGET=[1, 1]),
        )
    with pytest.raises(ValueError, match="mismatch"):
        load_lendingclub_identity_sidecar(
            sidecar_path=sidecar,
            manifest_path=manifest,
            processed_frame=processed.iloc[::-1].reset_index(drop=True),
        )


def test_tracked_lendingclub_identity_evidence_freezes_disjoint_split_hashes():
    root = Path(__file__).resolve().parents[1]
    evidence = json.loads(
        (
            root
            / "cleanup/audits/foundation_protocol_freeze/lendingclub_identity_evidence.json"
        ).read_text(encoding="utf-8")
    )
    assert evidence["identity_type"] == "authenticated_original_loan_id"
    assert evidence["dev_oot_overlap_count"] == 0
    assert evidence["stable_id_unique_count"] == evidence["retained_row_count"]
    assert evidence["stable_id_null_count"] == 0
    assert evidence["splits"]["DEV"]["ordered_row_id_sha256"] == (
        "4d4cd7973f00eb946fef0a6bb09e61fe6d2b9be92892786f352446660c68818e"
    )
    assert evidence["splits"]["OOT"]["ordered_row_id_target_sha256"] == (
        "9787d44d278d7965b0a966f19717dec1d9718ea8dfb6b96aa531d0f52d0a53e2"
    )
