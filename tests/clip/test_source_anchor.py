from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from credit_risk_fs.clip.source_anchor import (
    EXPECTED_RANKING,
    SourceAnchorConfig,
    build_feature_stability_evidence,
    build_seed_anchor,
    build_source_anchor_manifest,
    equal_duration_boundaries,
    fit_frozen_bucketizer,
    select_anchor_members,
    validate_seed_anchor,
    validate_source_anchor_artifacts,
    vector_hash,
)
from credit_risk_fs.utils.hashing import sha256_file


def _config(**overrides) -> SourceAnchorConfig:
    values = {
        "source_dataset": "lendingclub_v2",
        "dev_start": -1795.0,
        "dev_end": -1065.0,
        "temporal_subwindows": 4,
        "subwindow_strategy": "equal_duration",
        "subwindow_boundaries": equal_duration_boundaries(-1795, -1065, 4),
        "training_split_only": True,
        "target_used": False,
        "oot_used": False,
        "external_data_used": False,
        "max_adjacent_window_psi": 0.10,
        "max_missing_rate_difference": 0.05,
        "member_count": 23,
        "ranking": EXPECTED_RANKING,
        "fail_if_insufficient_members": True,
        "min_non_missing_per_subwindow": 5,
        "numeric_bins": 5,
        "categorical_min_count": 2,
        "psi_epsilon": 1e-6,
    }
    values.update(overrides)
    return SourceAnchorConfig(**values)


def _data(feature_count: int = 25) -> tuple[pd.DataFrame, pd.DataFrame]:
    boundaries = equal_duration_boundaries(-1795, -1065, 4)
    rows = []
    for window_index, (left, right) in enumerate(
        zip(boundaries[:-1], boundaries[1:])
    ):
        for index in range(10):
            row = {
                "recent_decision": left + (right - left) * (index + 0.5) / 10,
                "TARGET": (index + window_index) % 2,
            }
            for feature_index in range(feature_count):
                row[f"f{feature_index:02d}"] = (
                    ("A" if index % 2 else "B")
                    if feature_index == feature_count - 1
                    else float(index)
                )
            rows.append(row)
    data = pd.DataFrame(rows)
    candidates = pd.DataFrame(
        {
            "feature_id": [f"id-{index:02d}" for index in range(feature_count)],
            "feature_name": [f"f{index:02d}" for index in range(feature_count)],
            "semantic_group": ["synthetic"] * feature_count,
            "eligible_for_pairing": [True] * feature_count,
            "split_assignment": ["train"] * feature_count,
            "exclusion_reason": [""] * feature_count,
        }
    )
    return data, candidates


def _evidence_rows(count: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature_id": [f"id-{index:02d}" for index in range(count)],
            "feature_name": [f"f{index:02d}" for index in range(count)],
            "semantic_group": ["synthetic"] * count,
            "subwindow_non_missing_counts": ["[10, 10, 10, 10]"] * count,
            "subwindow_missing_rates": ["[0, 0, 0, 0]"] * count,
            "max_missing_rate_difference": [0.0] * count,
            "adjacent_window_psi_values": ["[0, 0, 0]"] * count,
            "max_adjacent_window_psi": [index / 10000 for index in range(count)],
            "psi_bucket_manifest": ["{}"] * count,
            "exact_duplicate_group": [""] * count,
            "alias_group": [""] * count,
            "identity_transform_group": [""] * count,
            "contrastive_split": ["train"] * count,
            "eligibility_status": ["eligible"] * count,
            "exclusion_reason": [""] * count,
            "target_used": [False] * count,
            "oot_used": [False] * count,
            "external_data_used": [False] * count,
        }
    )


def test_four_equal_duration_subwindows_are_deterministic() -> None:
    first = equal_duration_boundaries(-1795, -1065, 4)
    second = equal_duration_boundaries(-1795, -1065, 4)
    assert first == second == (-1795.0, -1612.5, -1430.0, -1247.5, -1065.0)
    assert np.allclose(np.diff(first), [182.5] * 4)


def test_stability_ignores_target_and_lendingclub_oot() -> None:
    data, candidates = _data()
    baseline, _ = build_feature_stability_evidence(
        data, candidates, time_column="recent_decision", config=_config()
    )
    changed = data.copy()
    changed["TARGET"] = 1 - changed["TARGET"]
    oot = changed.head(10).copy()
    oot["recent_decision"] = -1000
    for column in candidates.feature_name:
        oot[column] = 999999
    observed, _ = build_feature_stability_evidence(
        pd.concat([changed, oot], ignore_index=True),
        candidates,
        time_column="recent_decision",
        config=_config(),
    )
    pd.testing.assert_frame_equal(baseline, observed)
    assert not baseline.target_used.any()
    assert not baseline.oot_used.any()
    assert not baseline.external_data_used.any()


def test_homecredit_data_cannot_enter_stability_evidence() -> None:
    data, candidates = _data()
    data["dataset"] = "homecredit"
    with pytest.raises(ValueError, match="Home Credit"):
        build_feature_stability_evidence(
            data,
            candidates,
            time_column="recent_decision",
            config=_config(),
        )


def test_numeric_and_categorical_bins_are_fit_on_first_window_and_frozen() -> None:
    numeric = fit_frozen_bucketizer(
        pd.Series([0, 1, 2, 3, 4], name="numeric"),
        numeric_bins=4,
        categorical_min_count=2,
    )
    assert numeric.fit_window_index == 0
    assert numeric.numeric_edges[0] == -np.inf
    assert numeric.numeric_edges[-1] == np.inf
    assert len(numeric.transform(pd.Series([100, -100, np.nan]))) == 3

    categorical = fit_frozen_bucketizer(
        pd.Series(["A", "A", "B", "rare"], name="category"),
        numeric_bins=4,
        categorical_min_count=2,
    )
    assert categorical.categorical_levels == ("A",)
    assert categorical.transform(pd.Series(["A", "B", "UNSEEN", np.nan])).tolist() == [
        "A",
        "OTHER",
        "OTHER",
        "MISSING",
    ]


def test_missingness_drift_and_validation_split_exclusion() -> None:
    data, candidates = _data()
    boundaries = equal_duration_boundaries(-1795, -1065, 4)
    data.loc[
        (data.recent_decision >= boundaries[3])
        & (data.index % 2 == 0),
        "f00",
    ] = np.nan
    candidates.loc[candidates.feature_name.eq("f01"), "split_assignment"] = "validation"
    evidence, _ = build_feature_stability_evidence(
        data, candidates, time_column="recent_decision", config=_config()
    )
    drift = evidence.set_index("feature_name").loc["f00"]
    validation = evidence.set_index("feature_name").loc["f01"]
    assert drift.max_missing_rate_difference > 0.05
    assert drift.exclusion_reason == "stability_threshold_failed"
    assert validation.exclusion_reason == "not_in_contrastive_training_split"


def test_ranking_is_deterministic_and_selects_exactly_23() -> None:
    evidence = _evidence_rows(25).sample(frac=1, random_state=7)
    first, _ = select_anchor_members(evidence, config=_config())
    second, _ = select_anchor_members(evidence, config=_config())
    assert first.feature_id.tolist() == second.feature_id.tolist()
    assert len(first) == 23
    assert first.anchor_rank.tolist() == list(range(1, 24))


def test_selection_fails_closed_when_fewer_than_23_qualify() -> None:
    with pytest.raises(RuntimeError, match="required=23, observed=22"):
        select_anchor_members(_evidence_rows(22), config=_config())


def test_aliases_exact_duplicates_and_validation_features_are_not_members() -> None:
    evidence = _evidence_rows(27)
    evidence.loc[0:1, "alias_group"] = "alias:one"
    evidence.loc[2:3, "exact_duplicate_group"] = "exact:one"
    evidence.loc[4, "contrastive_split"] = "validation"
    members, audit = select_anchor_members(evidence, config=_config())
    assert not (
        members.alias_group.fillna("").astype(str).loc[lambda values: values != ""].duplicated().any()
    )
    assert not (
        members.exact_duplicate_group.fillna("").astype(str).loc[lambda values: values != ""].duplicated().any()
    )
    assert "id-04" not in set(members.feature_id)
    assert "identity_equivalent_to_higher_ranked_member" in set(
        audit.selection_exclusion_reason
    )


def test_cross_type_identity_conflicts_cannot_coexist_in_anchor() -> None:
    evidence = _evidence_rows(25)
    evidence["identity_conflict_group"] = ""
    evidence.loc[0, "identity_conflict_group"] = "all_identity:shared"
    evidence.loc[1, "identity_conflict_group"] = "all_identity:shared"
    members, audit = select_anchor_members(evidence, config=_config())
    assert not {"id-00", "id-01"}.issubset(set(members.feature_id))
    assert audit.loc[
        audit.feature_id.eq("id-01"), "selection_exclusion_reason"
    ].iloc[0] == "identity_equivalent_to_higher_ranked_member"


def test_per_seed_anchor_normalizes_members_and_centroid() -> None:
    members = _evidence_rows(23)
    embeddings = pd.DataFrame(
        {
            "feature_id": members.feature_id,
            "feature_name": members.feature_name,
            "dataset": ["lendingclub_v2"] * 23,
            "split": ["train"] * 23,
            "joint_0000": np.arange(1, 24, dtype=float),
            "joint_0001": np.ones(23),
        }
    )
    anchor, manifest = build_seed_anchor(
        embeddings, members, seed=11, checkpoint_hash="checkpoint"
    )
    assert np.isclose(np.linalg.norm(anchor), 1.0, atol=1e-6)
    assert manifest["member_normalization"] == "l2_per_seed_space"
    assert manifest["anchor_hash"] == vector_hash(anchor)


def test_homecredit_seed_anchor_is_rejected() -> None:
    members = _evidence_rows(23)
    embeddings = pd.DataFrame(
        {
            "feature_id": members.feature_id,
            "dataset": ["homecredit"] * 23,
            "split": ["train"] * 23,
            "joint_0000": np.ones(23),
            "joint_0001": np.ones(23),
        }
    )
    with pytest.raises(ValueError, match="Home Credit"):
        build_seed_anchor(
            embeddings, members, seed=11, checkpoint_hash="checkpoint"
        )


def test_incorrect_anchor_metadata_and_hashes_are_rejected(tmp_path) -> None:
    config = _config()
    members = _evidence_rows(23)
    members_path = tmp_path / "members.csv"
    members.to_csv(members_path, index=False)
    evidence_path = tmp_path / "evidence.csv"
    members.to_csv(evidence_path, index=False)
    manifest = build_source_anchor_manifest(
        config=config,
        members=members,
        configuration_hash="config",
        data_manifest_hash="data",
        evidence_hash=sha256_file(evidence_path),
        external_dataset="homecredit",
        source_feature_universe_hash="universe",
        feature_split_hash="split",
        identity_evidence_hash="identity",
        raw_statistical_evidence_hash="raw",
        statistical_preprocessor_hash="preprocessor",
    )
    manifest["anchor_members_hash"] = sha256_file(members_path)
    validate_source_anchor_artifacts(
        config=config,
        members=members,
        manifest=manifest,
        training_feature_ids=set(members.feature_id),
        configuration_hash="config",
        data_manifest_hash="data",
        external_dataset="homecredit",
        source_feature_universe_hash="universe",
        feature_split_hash="split",
        identity_evidence_hash="identity",
        raw_statistical_evidence_hash="raw",
        statistical_preprocessor_hash="preprocessor",
        member_path=members_path,
        evidence_path=evidence_path,
    )
    bad = dict(manifest)
    bad["source_dataset"] = "homecredit"
    with pytest.raises(ValueError, match="source_dataset"):
        validate_source_anchor_artifacts(
            config=config,
            members=members,
            manifest=bad,
            training_feature_ids=set(members.feature_id),
            configuration_hash="config",
            data_manifest_hash="data",
            external_dataset="homecredit",
            source_feature_universe_hash="universe",
            feature_split_hash="split",
            identity_evidence_hash="identity",
            raw_statistical_evidence_hash="raw",
            statistical_preprocessor_hash="preprocessor",
            member_path=members_path,
            evidence_path=evidence_path,
        )
    evidence_path.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="evidence hash"):
        validate_source_anchor_artifacts(
            config=config,
            members=members,
            manifest=manifest,
            training_feature_ids=set(members.feature_id),
            configuration_hash="config",
            data_manifest_hash="data",
            external_dataset="homecredit",
            source_feature_universe_hash="universe",
            feature_split_hash="split",
            identity_evidence_hash="identity",
            raw_statistical_evidence_hash="raw",
            statistical_preprocessor_hash="preprocessor",
            member_path=members_path,
            evidence_path=evidence_path,
        )


def test_seed_anchor_hash_and_source_manifest_hash_are_enforced() -> None:
    vector = np.array([1.0, 0.0], dtype="float32")
    anchor_hash = vector_hash(vector)
    seed_manifest = {
        "source_dataset": "lendingclub_v2",
        "seed": 11,
        "checkpoint_hash": "checkpoint",
        "member_count": 23,
        "target_used": False,
        "oot_used": False,
        "external_data_used": False,
        "anchor_hash": anchor_hash,
    }
    source_manifest = {"anchor_hashes_by_seed": {"11": anchor_hash}}
    validate_seed_anchor(
        vector=vector,
        seed_manifest=seed_manifest,
        source_manifest=source_manifest,
        expected_seed=11,
        expected_checkpoint_hash="checkpoint",
    )
    source_manifest["anchor_hashes_by_seed"]["11"] = "wrong"
    with pytest.raises(ValueError, match="source-manifest"):
        validate_seed_anchor(
            vector=vector,
            seed_manifest=seed_manifest,
            source_manifest=source_manifest,
            expected_seed=11,
            expected_checkpoint_hash="checkpoint",
        )
