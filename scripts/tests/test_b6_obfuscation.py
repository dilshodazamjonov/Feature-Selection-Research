from __future__ import annotations

import json
import socket
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts import b6_obfuscation as b6
from scripts import run_b6_obfuscation_ablation as runner


def _feature_names(count: int = b6.GLOBAL_FEATURE_COUNT) -> list[str]:
    return [f"ORIGINAL_FEATURE_{index:03d}" for index in range(1, count + 1)]


def _mapping() -> dict[str, str]:
    return b6.build_global_feature_mapping(_feature_names())


def _strict_response_format() -> dict[str, object]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "feature_ranking",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "selected_features": {
                        "type": "array",
                        "minItems": 100,
                        "maxItems": 100,
                        "items": {"type": "string"},
                    }
                },
                "required": ["selected_features"],
                "additionalProperties": False,
            },
        },
    }


def _safe_request(mapping: dict[str, str]) -> tuple[dict[str, object], list[dict[str, str]], list[str]]:
    candidate_ids = list(mapping.values())[:120]
    records = [
        {
            "feature_name": feature_id,
            "approved_definition": f"Permitted definition for {feature_id}",
            "lineage": f"source -> {feature_id}",
        }
        for feature_id in candidate_ids
    ]
    payload = b6.build_request_payload(
        messages=[
            {"role": "system", "content": "Rank definition-only candidate records."},
            {
                "role": "user",
                "content": json.dumps(records, ensure_ascii=False, separators=(",", ":")),
            },
        ],
        response_format=_strict_response_format(),
    )
    return payload, records, candidate_ids


def _prediction(ids: list[str], targets: list[int], scores: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stable_row_id": ids,
            "target": targets,
            "prediction_probability": scores,
        }
    )


def _valid_output_inputs() -> tuple[
    dict[str, list[str]], dict[str, str], list[dict[str, object]], list[dict[str, object]]
]:
    reverse = b6.reverse_feature_mapping(_mapping())
    ranked = list(reverse)[:100]
    rankings = {partition: ranked.copy() for partition in b6.PARTITIONS}
    performance = [
        {
            "dataset": "homecredit",
            "backbone": backbone,
            "K": k,
            "ho_auc": 0.70 + index * 0.05,
            "fold1_auc": 0.61 + index * 0.01,
            "fold2_auc": 0.62 + index * 0.01,
            "fold3_auc": 0.63 + index * 0.01,
            "fold4_auc": 0.64 + index * 0.01,
            "fold5_auc": 0.65 + index * 0.01,
        }
        for index, (backbone, k) in enumerate((('lr', 20), ('catboost', 40)))
    ]
    pairwise = []
    for backbone, auc_a, auc_b in (("lr", 0.71, 0.70), ("catboost", 0.78, 0.76)):
        pairwise.append(
            {
                "dataset": "homecredit",
                "backbone": backbone,
                "method_A": "Obfuscated Pure LLM",
                "method_B": "Pure LLM",
                "auc_A": auc_a,
                "auc_B": auc_b,
                "delta": auc_a - auc_b,
                "ci95_low": -0.01,
                "ci95_high": 0.03,
                "p_value": 0.2,
                "n_ho": 120053,
            }
        )
    return rankings, reverse, performance, pairwise


def test_global_mapping_is_deterministic_for_frozen_seed() -> None:
    names = _feature_names()
    assert b6.build_global_feature_mapping(names) == b6.build_global_feature_mapping(names)


def test_global_mapping_is_bijective_and_input_order_independent() -> None:
    names = _feature_names()
    mapping = b6.build_global_feature_mapping(names)
    reversed_input = b6.build_global_feature_mapping(reversed(names))
    assert mapping == reversed_input
    assert set(mapping) == set(names)
    assert len(set(mapping.values())) == 529
    assert set(mapping.values()) == {f"F{index:03d}" for index in range(1, 530)}


def test_one_global_mapping_gives_same_feature_id_in_every_partition() -> None:
    mapping = _mapping()
    feature = "ORIGINAL_FEATURE_217"
    partition_ids = {partition: mapping[feature] for partition in b6.PARTITIONS}
    assert len(set(partition_ids.values())) == 1


def test_ids_are_not_assigned_in_original_name_lexicographic_order() -> None:
    names = sorted(_feature_names())
    mapping = _mapping()
    assert [mapping[name] for name in names] != [f"F{index:03d}" for index in range(1, 530)]


def test_scrubbing_removes_names_from_name_description_and_lineage() -> None:
    mapping = {"AMT_CREDIT": "F001", "EXT_SOURCE_1": "F002"}
    records = [
        {
            "feature_name": "AMT_CREDIT",
            "description": "AMT_CREDIT plus external credit score EXT_SOURCE_1",
            "lineage": {"formula": "AMT_CREDIT / EXT_SOURCE_1"},
            "semantic_group": "external credit score",
        }
    ]
    scrubbed = b6.scrub_candidate_records(records, mapping)
    serialized = json.dumps(scrubbed)
    assert "AMT_CREDIT" not in serialized
    assert "EXT_SOURCE_1" not in serialized
    assert scrubbed[0]["feature_name"] == "F001"
    assert scrubbed[0]["semantic_group"] == "external credit score"
    assert records[0]["feature_name"] == "AMT_CREDIT"


def test_overlapping_original_names_are_scrubbed_in_one_safe_pass() -> None:
    mapping = {"AMT_CREDIT": "F001", "AMT_CREDIT_SUM": "F002"}
    scrubbed = b6.scrub_candidate_records(
        [{"feature_name": "AMT_CREDIT_SUM", "lineage": "AMT_CREDIT_SUM / AMT_CREDIT"}],
        mapping,
    )
    assert scrubbed == [{"feature_name": "F002", "lineage": "F002 / F001"}]


def test_fully_rendered_request_has_no_original_name_literal() -> None:
    mapping = _mapping()
    payload, records, candidate_ids = _safe_request(mapping)
    b6.assert_request_payload_safe(
        payload,
        original_feature_names=mapping,
        candidate_records=records,
        expected_candidate_ids=candidate_ids,
    )
    leaked = json.loads(json.dumps(payload))
    leaked["messages"][1]["content"] += " ORIGINAL_FEATURE_009"
    with pytest.raises(b6.B6ProtocolError, match="original-name literal"):
        b6.assert_request_payload_safe(
            leaked,
            original_feature_names=mapping,
            candidate_records=records,
            expected_candidate_ids=candidate_ids,
        )


def test_request_builder_cannot_serialize_reverse_mapping() -> None:
    mapping = _mapping()
    payload, records, candidate_ids = _safe_request(mapping)
    serialized = json.dumps(payload)
    assert "reverse_mapping" not in serialized
    assert "original_feature_name" not in serialized
    assert not any(name in serialized for name in mapping)
    b6.assert_request_payload_safe(
        payload,
        original_feature_names=mapping,
        candidate_records=records,
        expected_candidate_ids=candidate_ids,
    )


def test_request_safety_rejects_target_aware_fields_budget_mode_and_candidate_drift() -> None:
    mapping = _mapping()
    payload, records, candidate_ids = _safe_request(mapping)
    target_aware_records = [*records, {"feature_name": "F529", "validation_auc": 0.8}]
    with pytest.raises(b6.B6ProtocolError, match="forbidden evidence field"):
        b6.assert_request_payload_safe(
            payload,
            original_feature_names=mapping,
            candidate_records=target_aware_records,
            expected_candidate_ids=[*candidate_ids, "F529"],
        )
    with pytest.raises(b6.B6ProtocolError, match="evidence_mode"):
        b6.assert_request_payload_safe(
            payload,
            original_feature_names=mapping,
            candidate_records=records,
            expected_candidate_ids=candidate_ids,
            evidence_mode="screened",
        )
    with pytest.raises(b6.B6ProtocolError, match="ranking budget"):
        b6.assert_request_payload_safe(
            payload,
            original_feature_names=mapping,
            candidate_records=records,
            expected_candidate_ids=candidate_ids,
            ranking_budget=99,
        )
    with pytest.raises(b6.B6ProtocolError, match="IDs/order"):
        b6.assert_request_payload_safe(
            payload,
            original_feature_names=mapping,
            candidate_records=records,
            expected_candidate_ids=list(reversed(candidate_ids)),
        )


@pytest.mark.parametrize(
    "mutation",
    ["missing", "duplicate", "invented", "original_name", "wrong_length"],
)
def test_response_validation_rejects_invalid_outputs(mutation: str) -> None:
    mapping = _mapping()
    valid = list(mapping.values())[:100]
    response: dict[str, object] = {"status": "success", "selected_features": valid.copy()}
    if mutation == "missing":
        response.pop("selected_features")
    elif mutation == "duplicate":
        response["selected_features"] = valid[:-1] + [valid[0]]
    elif mutation == "invented":
        response["selected_features"] = valid[:-1] + ["F999"]
    elif mutation == "original_name":
        response["selected_features"] = valid[:-1] + [next(iter(mapping))]
    elif mutation == "wrong_length":
        response["selected_features"] = valid[:-1]
    with pytest.raises(b6.B6ProtocolError):
        b6.validate_ranking_response(
            response,
            candidate_ids=mapping.values(),
            original_feature_names=mapping,
        )


def test_response_validation_accepts_exactly_100_valid_distinct_ids() -> None:
    mapping = _mapping()
    valid = list(mapping.values())[:100]
    result = b6.validate_ranking_response(
        {"status": "success", "selected_features": valid},
        candidate_ids=mapping.values(),
        original_feature_names=mapping,
    )
    assert result == tuple(valid)


def test_prefixes_translate_exactly_20_and_40_original_features() -> None:
    reverse = b6.reverse_feature_mapping(_mapping())
    ranking = list(reverse)[:100]
    lr = b6.translate_ranking_prefix(ranking, reverse, 20)
    catboost = b6.translate_ranking_prefix(ranking, reverse, 40)
    assert len(lr) == 20
    assert len(catboost) == 40
    assert lr == catboost[:20]


def test_ho_access_is_blocked_until_subset_and_model_are_frozen() -> None:
    gate = b6.HOAccessGate()
    with pytest.raises(b6.B6ProtocolError, match="HO access"):
        gate.assert_access_allowed()
    gate.mark_subset_frozen()
    with pytest.raises(b6.B6ProtocolError, match="HO access"):
        gate.assert_access_allowed()
    gate.mark_model_frozen()
    gate.assert_access_allowed()


def test_paired_alignment_rejects_duplicates_labels_missing_and_wrong_production_count() -> None:
    valid_a = _prediction(["1", "2", "3", "4"], [0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    valid_b = _prediction(["4", "3", "2", "1"], [1, 1, 0, 0], [0.8, 0.7, 0.3, 0.2])

    duplicate = valid_a.copy()
    duplicate.loc[1, "stable_row_id"] = "1"
    with pytest.raises(ValueError, match="duplicated"):
        b6.align_paired_predictions(duplicate, valid_b, production=False)

    different_label = valid_b.copy()
    different_label.loc[different_label["stable_row_id"] == "1", "target"] = 1
    with pytest.raises(ValueError, match="targets disagree"):
        b6.align_paired_predictions(valid_a, different_label, production=False)

    missing = valid_b.iloc[:-1].copy()
    with pytest.raises(ValueError, match="identity sets differ"):
        b6.align_paired_predictions(valid_a, missing, production=False)

    with pytest.raises(b6.B6ProtocolError, match="120053"):
        b6.align_paired_predictions(valid_a, valid_b, production=True)


def test_pairwise_delta_is_exact_auc_a_minus_auc_b() -> None:
    a = _prediction(
        [str(index) for index in range(8)],
        [0, 0, 0, 0, 1, 1, 1, 1],
        [0.05, 0.20, 0.30, 0.60, 0.40, 0.70, 0.80, 0.95],
    )
    b = _prediction(
        [str(index) for index in range(8)],
        [0, 0, 0, 0, 1, 1, 1, 1],
        [0.10, 0.25, 0.35, 0.45, 0.50, 0.65, 0.75, 0.90],
    )
    aligned = b6.align_paired_predictions(a, b, production=False)
    row = b6.build_pairwise_row(
        backbone="lr",
        aligned=aligned,
        production=False,
        repetitions=30,
        minimum_valid=30,
    )
    assert row["delta"] == row["auc_A"] - row["auc_B"]


def test_paired_stratification_uses_shared_indices_and_preserves_class_counts() -> None:
    target = np.array([0, 0, 0, 0, 1, 1, 1])
    scores_a = np.arange(len(target), dtype=float)
    scores_b = scores_a + 100.0
    draws = b6.draw_paired_stratified_indices(target, repetitions=25)
    for sampled in draws:
        assert len(sampled) == len(target)
        assert int(target[sampled].sum()) == int(target.sum())
        assert np.array_equal(scores_b[sampled] - scores_a[sampled], np.full(len(target), 100.0))


def test_csv_schema_row_count_ordering_and_index_column_are_enforced() -> None:
    rankings, reverse, performance_rows, pairwise_rows = _valid_output_inputs()
    ranking = b6.rankings_frame(rankings, reverse)
    performance = b6.performance_frame(performance_rows)
    pairwise = b6.pairwise_frame(pairwise_rows)
    b6.validate_output_frames(ranking, performance, pairwise)
    assert len(ranking) == 600
    assert ranking.iloc[0]["partition"] == "fold1"
    assert ranking.iloc[-1]["partition"] == "full_dev"

    invalid = ranking.copy()
    invalid.insert(0, "Unnamed: 0", range(len(invalid)))
    with pytest.raises(b6.B6ProtocolError, match="columns"):
        b6.validate_output_frames(invalid, performance, pairwise)


def test_runner_refuses_existing_b6_directory_before_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "B6_artifacts"
    output.mkdir()
    preflight_called = False

    def forbidden_preflight(*_args: object, **_kwargs: object) -> None:
        nonlocal preflight_called
        preflight_called = True
        raise AssertionError("preflight should not be entered")

    monkeypatch.setattr(runner, "run_preflight", forbidden_preflight)
    assert runner.main(["--execute", "--output-dir", str(output)]) == 2
    assert not preflight_called
    assert list(output.iterdir()) == []


def test_mocked_end_to_end_publishes_exactly_three_csvs_without_external_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_network(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("network access is forbidden in B6 tests")

    def no_fit(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("real estimator fitting is forbidden in B6 tests")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    from sklearn.linear_model import LogisticRegression

    monkeypatch.setattr(LogisticRegression, "fit", no_fit)

    rankings, reverse, performance_rows, pairwise_rows = _valid_output_inputs()
    output = tmp_path / "B6_artifacts"
    published = b6.publish_mocked_ablation(
        output_dir=output,
        rankings=rankings,
        reverse_mapping=reverse,
        performance_rows=performance_rows,
        pairwise_rows=pairwise_rows,
    )
    assert published == output.resolve()
    assert sorted(path.name for path in output.iterdir()) == sorted(b6.OUTPUT_FILENAMES)
    assert not list(tmp_path.glob(".B6_artifacts.staging-*"))
    b6.validate_output_directory(output)


def test_b6_tests_and_mock_path_reference_no_full_data_or_openai_client() -> None:
    helper_source = Path(b6.__file__).read_text(encoding="utf-8")
    test_source = Path(__file__).read_text(encoding="utf-8")
    forbidden_client = "Open" + "AI("
    assert "application_train.csv" not in helper_source
    assert "previous_application.csv" not in helper_source
    assert forbidden_client not in helper_source
    assert forbidden_client not in test_source


def test_prospective_contract_uses_paired_controls_and_no_historical_inputs() -> None:
    assert runner.PROTOCOL_CONTRACT["comparison"] == (
        "prospective_named_control_vs_obfuscated_treatment"
    )
    assert runner.PROTOCOL_CONTRACT["historical_results_as_inputs"] is False
    assert runner.PROTOCOL_CONTRACT["total_ranking_calls"] == 12
    assert runner.PROTOCOL_CONTRACT["candidate_universe_per_call"] == 529
    assert runner.PROTOCOL_CONTRACT["claim_boundary"].startswith("single-draw")


def test_prospective_record_pack_changes_rendered_prompt_only_by_name_literals() -> None:
    features = _feature_names()
    dtypes = {feature: np.dtype("float64") for feature in features}
    mapping, named, opaque, freeze = runner._freeze_record_pack(features, dtypes)
    named_prompt = runner._selector().build_target_free_prompt(
        named, expected_features=features
    )
    opaque_ids = [mapping[feature] for feature in features]
    opaque_prompt = runner._selector().build_target_free_prompt(
        opaque, expected_features=opaque_ids
    )
    assert opaque_prompt == b6.scrub_text(named_prompt, mapping)
    assert not any(feature in opaque_prompt for feature in features)
    assert freeze["candidate_count"] == 529
    assert freeze["named_prompt_sha256"] != freeze["obfuscated_prompt_sha256"]


def test_prospective_ranking_phase_makes_twelve_guarded_mocked_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features = _feature_names()
    dtypes = {feature: np.dtype("float64") for feature in features}
    mapping, named, opaque, _ = runner._freeze_record_pack(features, dtypes)
    calls: list[dict[str, object]] = []

    class FakeCompletions:
        def create(self, **kwargs: object) -> SimpleNamespace:
            calls.append(dict(kwargs))
            prompt = str(kwargs["messages"][1]["content"])
            feature_block = prompt.split("Features:\n", 1)[1].split(
                "\n\nReturn ONLY valid JSON:", 1
            )[0]
            candidate_ids = [
                json.loads(line[2:])["name"]
                for line in feature_block.splitlines()
            ]
            content = json.dumps(
                {
                    "selected_features": candidate_ids[:100],
                    "reasoning_summary": "mocked",
                    "selection_principles": ["stability"],
                    "feature_reasons": {},
                }
            )
            return SimpleNamespace(
                id=f"mock-{len(calls)}",
                model=b6.MODEL_SNAPSHOT,
                usage=SimpleNamespace(
                    prompt_tokens=1, completion_tokens=1, total_tokens=2
                ),
                choices=[
                    SimpleNamespace(message=SimpleNamespace(content=content))
                ],
            )

    class FakeClient:
        def __init__(self, api_key: str) -> None:
            assert api_key == "test-key"
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    import openai

    monkeypatch.setattr(openai, "OpenAI", FakeClient)
    rankings, diagnostics = runner._make_rankings(
        named_records=named,
        opaque_records=opaque,
        candidate_features=features,
        mapping=mapping,
    )
    assert len(calls) == 12
    assert len(diagnostics) == 12
    assert set(rankings) == {"named", "obfuscated"}
    assert all(set(value) == set(b6.PARTITIONS) for value in rankings.values())
    assert all(len(result) == 100 for value in rankings.values() for result in value.values())


def test_prospective_cv_and_refit_use_24_mocked_fits_and_authorize_ho_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features = [f"FEATURE_{index:03d}" for index in range(100)]
    rows = 72
    ordered = pd.DataFrame(
        {
            feature: np.linspace(0.0, 1.0, rows) + index
            for index, feature in enumerate(features)
        }
    )
    ordered["__target__"] = np.arange(rows) % 2
    ordered["__time__"] = np.arange(rows)
    ordered["__stable_id__"] = [f"ID{index:03d}" for index in range(rows)]
    rankings = {
        condition: {partition: tuple(features) for partition in b6.PARTITIONS}
        for condition in runner.CONDITIONS
    }
    calls: list[tuple[str, bool]] = []

    def fake_fit_one(**kwargs: object) -> tuple[object, object, object, np.ndarray | None]:
        validation_target = kwargs["y_validation"]
        calls.append((str(kwargs["backbone"]), validation_target is None))
        scores = (
            None
            if validation_target is None
            else np.asarray(validation_target, dtype=float) * 0.8 + 0.1
        )
        return object(), object(), lambda *_args: np.array([]), scores

    monkeypatch.setattr(runner, "_fit_one", fake_fit_one)
    fold_aucs, frozen, gate = runner._fit_cv_and_full_dev(
        ordered_dev=ordered,
        rankings=rankings,
        params={"lr": {}, "catboost": {}},
    )
    assert len(calls) == 24
    assert sum(is_final for _, is_final in calls) == 4
    assert len(frozen) == 4
    assert all(len(fold_aucs[condition][backbone]) == 5 for condition in runner.CONDITIONS for backbone in b6.BACKBONES)
    gate.assert_access_allowed()
