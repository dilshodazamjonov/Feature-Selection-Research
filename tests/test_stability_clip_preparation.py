from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pytest

import credit_risk_fs.clip.stability_preparation as stability_preparation

from credit_risk_fs.clip.exact_duplicates import feature_order_hash
from credit_risk_fs.clip.statistical_schema_v2 import (
    DESCRIPTOR_COLUMNS_V2,
    SCALED_DESCRIPTOR_COLUMNS_V2,
    UNSCALED_INDICATOR_COLUMNS_V2,
)
from credit_risk_fs.clip.stability_preparation import (
    DATASET_ID,
    GENERATED_RELATIVE_PATHS,
    DevMatrixReader,
    PreparationContext,
    StabilityPreparationError,
    _write_json,
    build_feature_universe_from_payloads,
    build_identity_equivalence,
    build_representation_split,
    build_semantic_metadata,
    build_stability_source_anchor,
    build_stability_pairs,
    build_validation_report,
    compute_raw_descriptors_from_frame,
    compute_temporal_stability,
    encode_text_embeddings,
    find_exact_dev_duplicates_chunked,
    fit_stability_preprocessor,
    load_work_checkpoint,
    render_feature_text,
    run_preparation,
    select_stability_anchor_members,
    verify_existing_output,
    write_work_checkpoint,
    write_sha256_manifest,
)
from credit_risk_fs.clip.text_encoder import MockFrozenTextEncoder


def _universe_payload(count: int) -> tuple[dict, dict, list[str]]:
    names = [f"d0__static__fixture_{index:04d}A" for index in range(count)]
    metadata = {
        "predictor_columns": names,
        "non_predictor_columns": ["case_id", "date_decision", "target"],
        "columns": [
            {"name": name, "arrow_type": "double"}
            for name in names
        ],
    }
    lineage = {
        "features": [
            {
                "output_feature": name,
                "source_family": "static",
                "source_feature": f"fixture_{index:04d}A",
                "aggregation": "identity_after_family_prefix",
                "logical_type": "numeric",
                "protocol_action": "include",
                "protocol_output_prefix": name,
            }
            for index, name in enumerate(names)
        ]
    }
    return metadata, lineage, names


def _fixture_universe(count: int = 20) -> pd.DataFrame:
    metadata, lineage, names = _universe_payload(count)
    universe, _ = build_feature_universe_from_payloads(
        metadata=metadata,
        lineage=lineage,
        dataset_id=DATASET_ID,
        expected_count=count,
        expected_universe_hash=feature_order_hash(names),
    )
    return universe


def _definitions(tmp_path: Path, count: int) -> Path:
    path = tmp_path / "feature_definitions.csv"
    pd.DataFrame(
        {
            "Variable": [f"fixture_{index:04d}A" for index in range(count)],
            "Description": [f"Official fixture meaning {index}." for index in range(count)],
        }
    ).to_csv(path, index=False)
    return path


def test_reconstructs_exact_1959_authenticated_feature_identities() -> None:
    metadata, lineage, names = _universe_payload(1959)
    universe, manifest = build_feature_universe_from_payloads(
        metadata=metadata,
        lineage=lineage,
        dataset_id=DATASET_ID,
        expected_count=1959,
        expected_universe_hash=feature_order_hash(names),
    )
    assert len(universe) == 1959
    assert universe.feature_name.nunique() == 1959
    assert universe.feature_id.nunique() == 1959
    assert universe.eligible_for_clip.all()
    assert manifest["ordered_feature_name_sha256"] == feature_order_hash(names)
    with pytest.raises(StabilityPreparationError, match="count or uniqueness"):
        build_feature_universe_from_payloads(
            metadata=metadata,
            lineage=lineage,
            dataset_id=DATASET_ID,
            expected_count=1958,
        )


def test_semantics_and_feature_text_v1_are_deterministic_and_traced(tmp_path: Path) -> None:
    universe = _fixture_universe(3)
    semantics = build_semantic_metadata(
        universe, feature_definitions_path=_definitions(tmp_path, 3)
    )
    rendered = render_feature_text(universe, semantics, source_manifest_hash="a" * 64)
    assert len(semantics) == 3
    assert semantics.description.str.contains("Official fixture meaning").all()
    assert set(semantics.description_source) == {
        "authenticated_feature_definitions_plus_lineage_operation"
    }
    assert set(semantics.semantic_group) == {"source_table::static"}
    assert set(rendered.template_version) == {"feature_text_v1"}
    assert rendered.rendered_text.str.startswith("Feature: ").all()
    assert rendered.rendered_text_sha256.str.fullmatch(r"[0-9a-f]{64}").all()
    rerendered = render_feature_text(universe, semantics, source_manifest_hash="a" * 64)
    pd.testing.assert_frame_equal(rendered, rerendered)


def test_guarded_reader_excludes_target_and_oot(tmp_path: Path) -> None:
    part = tmp_path / "part-00000.parquet"
    pd.DataFrame(
        {
            "date_decision": ["2019-01-01", "2020-02-25", "2020-02-26"],
            "d0__static__x": [1.0, 2.0, 999.0],
            "target": [0, 1, 1],
        }
    ).to_parquet(part, index=False)
    reader = DevMatrixReader(
        matrix_parts=[part],
        predictor_names=["d0__static__x"],
        date_column="date_decision",
        dev_start_inclusive="2019-01-01",
        dev_end_exclusive="2020-02-26",
        expected_dev_rows=2,
        row_batch_size=2,
    )
    observed = reader.read_frame(["d0__static__x"], include_date=True)
    assert observed["d0__static__x"].tolist() == [1.0, 2.0]
    assert observed.date_decision.max() == "2020-02-25"
    assert reader.audit.oot_feature_values_loaded is False
    assert reader.audit.target_values_loaded is False
    with pytest.raises(StabilityPreparationError, match="target or forbidden"):
        reader.read_frame(["target"])


def test_chunked_exact_duplicates_use_hash_only_for_candidates_then_verify(
    tmp_path: Path,
) -> None:
    part = tmp_path / "part-00000.parquet"
    pd.DataFrame(
        {
            "date_decision": ["2019-01-01", "2019-01-02", "2020-02-26"],
            "a": [1.0, np.nan, 999.0],
            "b": [1.0, np.nan, -999.0],
            "c": [1.0, 2.0, 999.0],
        }
    ).to_parquet(part, index=False)
    reader = DevMatrixReader(
        matrix_parts=[part],
        predictor_names=["a", "b", "c"],
        date_column="date_decision",
        dev_start_inclusive="2019-01-01",
        dev_end_exclusive="2020-02-26",
        expected_dev_rows=2,
        row_batch_size=2,
    )
    evidence = find_exact_dev_duplicates_chunked(
        reader, ["a", "b", "c"], feature_batch_size=2
    )
    assert len(evidence) == 1
    assert {evidence.iloc[0].feature_name_a, evidence.iloc[0].feature_name_b} == {"a", "b"}
    assert bool(evidence.iloc[0].actual_equality_verified)
    assert not bool(evidence.iloc[0].target_used)
    assert not bool(evidence.iloc[0].oot_used)


def test_equivalence_and_representation_split_are_group_safe_and_deterministic(
    tmp_path: Path,
) -> None:
    universe = _fixture_universe(20)
    semantics = build_semantic_metadata(
        universe, feature_definitions_path=_definitions(tmp_path, 20)
    )
    names = universe.feature_name.tolist()
    evidence = pd.DataFrame(
        {
            "feature_name_a": [names[0]],
            "feature_name_b": [names[1]],
            "evidence_source": ["fixture_aligned_DEV_values_and_missing_masks"],
            "actual_equality_verified": [True],
        }
    )
    equivalence = build_identity_equivalence(
        universe, exact_duplicate_evidence=evidence
    )
    assert equivalence.iloc[0].equivalence_group_id == equivalence.iloc[1].equivalence_group_id
    assert equivalence.iloc[0].group_size == 2
    split_a, manifest_a = build_representation_split(universe, semantics, equivalence)
    split_b, manifest_b = build_representation_split(universe, semantics, equivalence)
    pd.testing.assert_frame_equal(split_a, split_b)
    assert manifest_a["representation_split_hash"] == manifest_b["representation_split_hash"]
    assert split_a.groupby("equivalence_group_id").representation_split.nunique().max() == 1
    assert set(split_a.split_seed) == {42}


def test_hash_candidate_without_actual_equality_cannot_form_equivalence() -> None:
    universe = _fixture_universe(2)
    evidence = pd.DataFrame(
        {
            "feature_name_a": [universe.feature_name.iloc[0]],
            "feature_name_b": [universe.feature_name.iloc[1]],
            "evidence_source": ["hash_only"],
            "actual_equality_verified": [False],
        }
    )
    with pytest.raises(StabilityPreparationError, match="candidate hash"):
        build_identity_equivalence(universe, exact_duplicate_evidence=evidence)


def test_descriptor_order_and_train_only_robust_preprocessor_contract() -> None:
    universe = _fixture_universe(10)
    rng = np.random.default_rng(42)
    data = pd.DataFrame(
        {
            name: rng.normal(loc=index, scale=1.0, size=100)
            for index, name in enumerate(universe.feature_name)
        }
    )
    raw = compute_raw_descriptors_from_frame(data, universe)
    assert [column for column in DESCRIPTOR_COLUMNS_V2 if column in raw] == list(
        DESCRIPTOR_COLUMNS_V2
    )
    split = pd.DataFrame(
        {
            "feature_name": universe.feature_name,
            "representation_split": ["train"] * 8 + ["validation"] * 2,
        }
    )
    raw.loc[9, list(SCALED_DESCRIPTOR_COLUMNS_V2)] = 1e12
    split_hash = "b" * 64
    scaled, manifest = fit_stability_preprocessor(
        raw,
        split,
        feature_universe_hash=feature_order_hash(universe.feature_name.tolist()),
        representation_split_hash=split_hash,
    )
    assert manifest["fit_feature_count"] == 8
    assert manifest["validation_feature_identities_used_for_fit"] is False
    np.testing.assert_array_equal(
        raw[list(UNSCALED_INDICATOR_COLUMNS_V2)].to_numpy(float),
        scaled[list(UNSCALED_INDICATOR_COLUMNS_V2)].to_numpy(float),
    )
    continuous = scaled[list(SCALED_DESCRIPTOR_COLUMNS_V2)].to_numpy(float)
    assert np.isfinite(continuous).all()
    assert continuous.min() >= -8
    assert continuous.max() <= 8
    assert 8.0 in continuous[9]


def test_corrected_temporal_anchor_math_and_deterministic_selection() -> None:
    time = pd.Series(np.arange(40, dtype=float))
    values = pd.Series(np.tile(np.arange(10, dtype=float), 4), name="stable")
    evidence_row = compute_temporal_stability(
        values,
        time,
        boundaries=[0.0, 10.0, 20.0, 30.0, 40.0],
        min_non_missing_per_subwindow=5,
        numeric_bins=10,
        categorical_min_count=2,
        psi_epsilon=1e-6,
    )
    assert evidence_row["adjacent_window_psi_values"] == pytest.approx([0.0, 0.0, 0.0])
    assert evidence_row["max_missing_rate_difference"] == 0.0
    evidence = pd.DataFrame(
        [
            {
                "feature_id": f"{index:064x}",
                "feature_name": f"f{index}",
                "equivalence_group_id": f"g{index}",
                "eligibility_status": "eligible",
                "exclusion_reason": "",
                "max_adjacent_window_psi": index / 10000,
                "max_missing_rate_difference": 0.0,
            }
            for index in range(25)
        ]
    )
    members, audit = select_stability_anchor_members(
        evidence,
        member_count=23,
        max_adjacent_window_psi=0.10,
        max_missing_rate_difference=0.05,
    )
    assert len(members) == 23
    assert members.anchor_rank.tolist() == list(range(1, 24))
    assert audit.selection_status.eq("selected").sum() == 23


def test_anchor_calendar_windows_are_independent_of_pandas_datetime_unit(
    tmp_path: Path,
) -> None:
    universe = _fixture_universe(24)
    dates = pd.date_range("2019-01-01", periods=40, freq="D")
    stable_values = np.tile(np.arange(10, dtype=float), 4)
    matrix = pd.DataFrame(
        {
            "date_decision": dates.strftime("%Y-%m-%d"),
            **{name: stable_values for name in universe.feature_name},
        }
    )
    part = tmp_path / "part-00000.parquet"
    matrix.to_parquet(part, index=False)
    reader = DevMatrixReader(
        matrix_parts=[part],
        predictor_names=universe.feature_name.tolist(),
        date_column="date_decision",
        dev_start_inclusive="2019-01-01",
        dev_end_exclusive="2019-02-10",
        expected_dev_rows=40,
        row_batch_size=11,
    )
    equivalence = build_identity_equivalence(universe)
    representation = pd.DataFrame(
        {
            "feature_name": universe.feature_name,
            "equivalence_group_id": equivalence.equivalence_group_id,
            "representation_split": "train",
            "split_seed": 42,
        }
    )
    anchor, audit, manifest = build_stability_source_anchor(
        reader,
        universe,
        representation,
        equivalence,
        anchor_config={
            "member_count": 23,
            "max_adjacent_window_psi": 0.10,
            "max_missing_rate_difference": 0.05,
            "min_non_missing_per_subwindow": 5,
            "numeric_bins": 10,
            "categorical_min_count": 2,
            "psi_epsilon": 1e-6,
        },
        feature_batch_size=8,
        feature_universe_hash="a" * 64,
        representation_split_hash="b" * 64,
        raw_descriptors_hash="c" * 64,
    )
    assert len(anchor) == 23
    assert audit.qualifies_thresholds.sum() == 24
    assert manifest["subwindow_row_counts"] == [10, 10, 10, 10]
    assert manifest["numeric_day_boundaries"] == pytest.approx(
        [17897.0, 17907.0, 17917.0, 17927.0, 17937.0]
    )


def test_post_failure_stages_9_to_12_serialize_as_a_complete_package(
    tmp_path: Path,
) -> None:
    universe = _fixture_universe(24)
    semantics = build_semantic_metadata(
        universe, feature_definitions_path=_definitions(tmp_path, 24)
    )
    rendered = render_feature_text(universe, semantics, source_manifest_hash="a" * 64)
    equivalence = build_identity_equivalence(universe)
    representation = pd.DataFrame(
        {
            "feature_name": universe.feature_name,
            "equivalence_group_id": equivalence.equivalence_group_id,
            "representation_split": ["train"] * 23 + ["validation"],
            "split_seed": 42,
        }
    )
    dates = pd.date_range("2019-01-01", periods=40, freq="D")
    stable_values = np.tile(np.arange(10, dtype=float), 4)
    data = pd.DataFrame(
        {name: stable_values + index for index, name in enumerate(universe.feature_name)}
    )
    raw = compute_raw_descriptors_from_frame(data, universe)
    universe_hash = feature_order_hash(universe.feature_name.tolist())
    split_hash = "b" * 64
    scaled, preprocessor_manifest = fit_stability_preprocessor(
        raw,
        representation,
        feature_universe_hash=universe_hash,
        representation_split_hash=split_hash,
    )
    part = tmp_path / "post_failure_matrix.parquet"
    pd.concat(
        [pd.Series(dates.strftime("%Y-%m-%d"), name="date_decision"), data], axis=1
    ).to_parquet(part, index=False)
    reader = DevMatrixReader(
        matrix_parts=[part],
        predictor_names=universe.feature_name.tolist(),
        date_column="date_decision",
        dev_start_inclusive="2019-01-01",
        dev_end_exclusive="2019-02-10",
        expected_dev_rows=40,
        row_batch_size=13,
    )
    anchor, _, anchor_manifest = build_stability_source_anchor(
        reader,
        universe,
        representation,
        equivalence,
        anchor_config={
            "member_count": 23,
            "max_adjacent_window_psi": 0.10,
            "max_missing_rate_difference": 0.05,
            "min_non_missing_per_subwindow": 5,
            "numeric_bins": 10,
            "categorical_min_count": 2,
            "psi_epsilon": 1e-6,
        },
        feature_batch_size=8,
        feature_universe_hash=universe_hash,
        representation_split_hash=split_hash,
        raw_descriptors_hash="c" * 64,
    )
    encoder_config = {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "model_revision": "main",
        "embedding_dimension": 384,
        "normalize_embeddings": True,
        "batch_size": 8,
    }
    embeddings, embedding_manifest = encode_text_embeddings(
        rendered,
        universe,
        source_manifest_hash="a" * 64,
        encoder_config=encoder_config,
        feature_universe_hash=universe_hash,
        encoder=MockFrozenTextEncoder(
            model_name=encoder_config["model_name"],
            revision="main",
            embedding_dimension=384,
        ),
    )
    pairs = build_stability_pairs(
        universe,
        equivalence,
        representation,
        rendered,
        embeddings,
        raw,
        scaled,
        source_manifest_hash="a" * 64,
    )
    pair_path = tmp_path / "pairs.parquet"
    pairs.to_parquet(pair_path, index=False)
    assert len(pd.read_parquet(pair_path)) == 24

    config = {
        "dataset": {
            "expected_feature_count": 24,
            "forbidden_predictors": ["case_id", "date_decision", "target"],
        }
    }
    context = PreparationContext(
        repo_root=tmp_path,
        config_path=tmp_path / "fixture.json",
        config=config,
        configuration_hash="d" * 64,
    )
    report = build_validation_report(
        context=context,
        universe=universe,
        semantics=semantics,
        rendered_text=rendered,
        equivalence=equivalence,
        representation_split=representation,
        raw_descriptors=raw,
        scaled_descriptors=scaled,
        preprocessor_manifest=preprocessor_manifest,
        anchor=anchor,
        anchor_manifest=anchor_manifest,
        text_embeddings=embeddings,
        text_embedding_manifest=embedding_manifest,
        pairs=pairs,
        reader_audit=reader.audit.to_dict(),
    )
    assert report["overall_status"] == "PASS"
    report_path = _write_json(tmp_path / "validation_report.json", report)
    assert json.loads(report_path.read_text(encoding="utf-8"))["overall_status"] == "PASS"


def test_mock_embedding_schema_and_pair_table_join(tmp_path: Path) -> None:
    universe = _fixture_universe(3)
    semantics = build_semantic_metadata(
        universe, feature_definitions_path=_definitions(tmp_path, 3)
    )
    rendered = render_feature_text(universe, semantics, source_manifest_hash="a" * 64)
    encoder_config = {
        "model_name": "mock-frozen-text-encoder",
        "model_revision": "test",
        "embedding_dimension": 8,
        "normalize_embeddings": True,
        "batch_size": 2,
    }
    embeddings, manifest = encode_text_embeddings(
        rendered,
        universe,
        source_manifest_hash="a" * 64,
        encoder_config=encoder_config,
        feature_universe_hash=feature_order_hash(universe.feature_name.tolist()),
        encoder=MockFrozenTextEncoder(embedding_dimension=8),
    )
    assert manifest["embedding_dimension"] == 8
    embedding_columns = [
        column for column in embeddings if re.fullmatch(r"embedding_\d{4}", column)
    ]
    assert len(embedding_columns) == 8
    np.testing.assert_allclose(
        np.linalg.norm(embeddings[embedding_columns].to_numpy(float), axis=1),
        1.0,
        atol=1e-5,
    )
    equivalence = build_identity_equivalence(universe)
    representation = pd.DataFrame(
        {
            "feature_name": universe.feature_name,
            "equivalence_group_id": equivalence.equivalence_group_id,
            "representation_split": ["train", "train", "validation"],
            "split_seed": 42,
        }
    )
    data = pd.DataFrame({name: [1.0, 2.0, 3.0] for name in universe.feature_name})
    raw = compute_raw_descriptors_from_frame(data, universe)
    scaled = raw[["feature_id", "feature_name", *DESCRIPTOR_COLUMNS_V2]].copy()
    scaled["statistical_vector_sha256"] = ["c" * 64] * len(scaled)
    pairs = build_stability_pairs(
        universe,
        equivalence,
        representation,
        rendered,
        embeddings,
        raw,
        scaled,
        source_manifest_hash="a" * 64,
    )
    assert len(pairs) == 3
    assert pairs.feature_id.nunique() == 3
    assert len([column for column in pairs if re.fullmatch(r"embedding_\d{4}", column)]) == 8
    assert len([column for column in pairs if column.startswith("stat_")]) == 13
    assert "target" not in pairs


def test_sha_manifest_and_existing_output_verification_are_idempotent(tmp_path: Path) -> None:
    output = tmp_path / "clip_preparation_v1"
    for relative in GENERATED_RELATIVE_PATHS:
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture:{relative}\n", encoding="utf-8")
    (output / "methodology_lock.json").write_text(
        json.dumps({"configuration_hash": "a" * 64}), encoding="utf-8"
    )
    (output / "validation/validation_report.json").write_text(
        json.dumps({"overall_status": "PASS"}), encoding="utf-8"
    )
    roles = {relative: "fixture" for relative in GENERATED_RELATIVE_PATHS}
    first = write_sha256_manifest(output, roles)
    second = write_sha256_manifest(output, roles)
    pd.testing.assert_frame_equal(first, second)
    assert verify_existing_output(output, configuration_hash="a" * 64)
    (output / "metadata/feature_universe.csv").write_text("changed\n", encoding="utf-8")
    with pytest.raises(StabilityPreparationError, match="changed size|hash mismatch"):
        verify_existing_output(output, configuration_hash="a" * 64)


def test_work_checkpoint_is_hash_verified_and_resumable(tmp_path: Path) -> None:
    work = tmp_path / ".clip_preparation_v1.work"
    artifact = work / "statistics/statistical_descriptors_raw.csv"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("feature_name,missing_rate\nf,0.1\n", encoding="utf-8")
    checkpoint_path = write_work_checkpoint(
        work,
        configuration_hash="a" * 64,
        completed_stage=8,
        reader_audit={
            "scan_count": 490,
            "requested_predictor_column_count": 1959,
            "returned_row_counts": [1221743],
            "target_values_loaded": False,
            "oot_feature_values_loaded": False,
            "oot_labels_loaded": False,
        },
    )
    assert checkpoint_path.is_file()
    checkpoint = load_work_checkpoint(work, configuration_hash="a" * 64)
    assert checkpoint is not None
    assert checkpoint["completed_stage"] == 8
    assert checkpoint["reader_audit"]["requested_predictor_column_count"] == 1959
    artifact.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(StabilityPreparationError, match="changed|hash mismatch"):
        load_work_checkpoint(work, configuration_hash="a" * 64)


def test_orchestrator_rerun_reuses_stage_8_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    universe = _fixture_universe(24)
    definitions = _definitions(tmp_path, 24)
    dates = pd.date_range("2019-01-01", periods=40, freq="D")
    values = np.tile(np.arange(10, dtype=float), 4)
    part = tmp_path / "matrix.parquet"
    pd.DataFrame(
        {
            "date_decision": dates.strftime("%Y-%m-%d"),
            **{
                name: values + index
                for index, name in enumerate(universe.feature_name)
            },
        }
    ).to_parquet(part, index=False)
    output = tmp_path / "clip_preparation_v1"
    config = {
        "output_dir": str(output),
        "inputs": {
            "matrix_manifest": {"sha256": "a" * 64},
            "feature_definitions": {"path": str(definitions)},
        },
        "dataset": {
            "date_column": "date_decision",
            "dev_start_inclusive": "2019-01-01",
            "oot_start_exclusive_dev": "2019-02-10",
            "expected_dev_row_count": 40,
        },
        "statistical_view": {"row_batch_size": 13, "feature_batch_size": 8},
        "identity_equivalence": {"feature_batch_size": 8},
        "representation_split": {"seed": 42, "validation_fraction": 0.2},
        "source_anchor": {},
    }
    context = PreparationContext(
        repo_root=tmp_path,
        config_path=tmp_path / "fixture.json",
        config=config,
        configuration_hash="f" * 64,
    )
    calls = {"universe": 0, "raw": 0, "anchor": 0}
    original_raw = stability_preparation.compute_raw_descriptors

    def fake_universe(_context: PreparationContext):
        calls["universe"] += 1
        return universe.copy(), {
            "ordered_feature_name_sha256": feature_order_hash(
                universe.feature_name.tolist()
            )
        }

    def counted_raw(*args, **kwargs):
        calls["raw"] += 1
        return original_raw(*args, **kwargs)

    def blocked_anchor(*args, **kwargs):
        calls["anchor"] += 1
        raise StabilityPreparationError("forced post-stage-8 failure")

    monkeypatch.setattr(
        stability_preparation, "load_preparation_context", lambda *args, **kwargs: context
    )
    monkeypatch.setattr(
        stability_preparation,
        "validate_input_protocol",
        lambda _context: {
            "matrix_parts": [part],
            "input_provenance": [],
        },
    )
    monkeypatch.setattr(stability_preparation, "build_feature_universe", fake_universe)
    monkeypatch.setattr(stability_preparation, "compute_raw_descriptors", counted_raw)
    monkeypatch.setattr(
        stability_preparation, "build_stability_source_anchor", blocked_anchor
    )

    with pytest.raises(StabilityPreparationError, match="forced post-stage-8"):
        run_preparation("ignored.json", repo_root=tmp_path, progress=lambda _: None)
    assert calls == {"universe": 1, "raw": 1, "anchor": 1}
    checkpoint = load_work_checkpoint(
        output.with_name(".clip_preparation_v1.work"),
        configuration_hash="f" * 64,
    )
    assert checkpoint is not None and checkpoint["completed_stage"] == 8

    with pytest.raises(StabilityPreparationError, match="forced post-stage-8"):
        run_preparation("ignored.json", repo_root=tmp_path, progress=lambda _: None)
    assert calls == {"universe": 1, "raw": 1, "anchor": 2}
