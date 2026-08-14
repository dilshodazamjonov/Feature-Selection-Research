from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import credit_risk_fs.experiments.prompt_16_llm_supplement as supplement
from credit_risk_fs.data.homecredit_model_stability_2024.contract import canonical_sha256
from credit_risk_fs.selectors.llm_screening import LLMSelector
from credit_risk_fs.selectors.stable_core_llm_fill import StableCoreLLMFillSelector


ROOT = Path(__file__).resolve().parents[1]
AMENDMENT = (
    ROOT
    / "configs/protocols/homecredit_model_stability_2024_v2/"
    "prompt_16_llm_supplement_amendment.json"
)
V1_LOCK = (
    ROOT
    / "configs/protocols/homecredit_model_stability_2024_v1/"
    "third_dataset_protocol_lock.json"
)
MATRIX_ROOT = ROOT / "outputs/prompt_16_homecredit_model_stability_2024/matrix_v1"


def _load_cli():
    path = ROOT / "scripts/run_prompt_16_third_dataset.py"
    spec = importlib.util.spec_from_file_location("prompt16_supplement_cli_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic_metadata(count: int = 100) -> tuple[list[str], list[dict[str, object]]]:
    names = [f"f_{index:03d}" for index in range(count)]
    records: list[dict[str, object]] = []
    for name in names:
        fields = {
            "name": name,
            "source_family": "static",
            "source_table": "static",
            "original_feature": name,
            "depth": "0",
            "aggregation": "identity_after_family_prefix",
            "dtype": "float",
            "logical_type": "numeric",
            "approved_definition": f"Approved definition for {name}.",
        }
        rendered = "- " + json.dumps(
            fields, ensure_ascii=False, separators=(",", ":")
        )
        records.append(
            {
                **fields,
                "rendered_description": rendered,
                "description_sha256": hashlib.sha256(
                    rendered.encode("utf-8")
                ).hexdigest(),
            }
        )
    return names, records


class _DummyUsage:
    prompt_tokens = 100
    completion_tokens = 20
    total_tokens = 120


class _DummyResponse:
    def __init__(self, selected: list[str], *, model: str = supplement.FROZEN_LLM_MODEL):
        self.choices = [
            SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps(
                        {
                            "selected_features": selected,
                            "reasoning_summary": "synthetic strict response",
                            "selection_principles": ["stability", "coverage"],
                            "feature_reasons": {},
                        }
                    )
                )
            )
        ]
        self.usage = _DummyUsage()
        self.model = model
        self.id = "synthetic-response"


class _DummyCompletions:
    def __init__(self, responses: list[_DummyResponse]):
        self.responses = iter(responses)
        self.calls = 0

    def create(self, **_kwargs):
        self.calls += 1
        return next(self.responses)


def _strict_selector(selected_per_attempt: list[list[str]]) -> tuple[LLMSelector, _DummyCompletions]:
    selector = LLMSelector(
        description_csv_path="unused.json",
        cache_dir="unused",
        model=supplement.FROZEN_LLM_MODEL,
        temperature=0.0,
        max_features=100,
        ranking_budget=100,
        feature_budget=100,
        shared_pool_size=100,
        iv_filter_kwargs={},
    )
    completions = _DummyCompletions(
        [_DummyResponse(selected) for selected in selected_per_attempt]
    )
    selector._client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    return selector, completions


def test_v2_amendment_authenticates_exact_methods_registries_and_accounting():
    amendment, file_sha, internal_sha = supplement.load_supplemental_amendment(
        AMENDMENT
    )
    assert len(file_sha) == len(internal_sha) == 64
    assert [
        row["method_id"]
        for row in amendment["method_registry"]["added_methods"]
    ] == ["llm", "stable_core_llm_fill"]
    assert amendment["method_registry"]["unavailable_method"] == {
        "role": "semantic_mixed_voter",
        "status": "unavailable_due_to_unresolved_historical_provenance",
        "execution_cells": 0,
    }
    assert len(amendment["evaluation_registry"]["dev_evaluation_identities"]) == 20
    assert len(amendment["evaluation_registry"]["oot_evaluation_identities"]) == 4
    assert amendment["accounting"] == supplement.corrected_accounting()
    assert amendment["accounting"]["amended_dev_evaluations"] == 170
    assert amendment["accounting"]["amended_oot_evaluations"] == 34
    assert amendment["comparison_graph"]["semantic_mixed_voter_comparisons"] == []
    assert len(amendment["parameter_parity_audit"]) == 4
    assert {row["parity_verdict"] for row in amendment["parameter_parity_audit"]} == {
        "pass"
    }
    parity_path = (
        ROOT
        / "cleanup/audits/prompt_16_llm_scope_correction/parameter_parity_audit_v2.json"
    )
    parity = json.loads(parity_path.read_text(encoding="utf-8"))
    unsigned_parity = dict(parity)
    claimed_parity = unsigned_parity.pop("artifact_authentication_sha256")
    assert canonical_sha256(unsigned_parity) == claimed_parity
    assert parity["row_count"] == 4
    assert [(row["method_id"], row["model"]) for row in parity["rows"]] == [
        ("llm", "lr"),
        ("llm", "catboost"),
        ("stable_core_llm_fill", "lr"),
        ("stable_core_llm_fill", "catboost"),
    ]
    pointer_path = (
        ROOT
        / "configs/protocols/homecredit_model_stability_2024_v2/successor_pointer.json"
    )
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    unsigned_pointer = dict(pointer)
    claimed_pointer = unsigned_pointer.pop("artifact_authentication_sha256")
    assert canonical_sha256(unsigned_pointer) == claimed_pointer
    assert pointer["successor_amendment"]["file_sha256"] == file_sha
    assert pointer["oot_authorized"] is False


def test_real_target_free_description_and_prompt_hashes_match_the_freeze():
    amendment, _, _ = supplement.load_supplemental_amendment(AMENDMENT)
    bundle = supplement.build_description_and_prompt_freeze(
        matrix_root=MATRIX_ROOT,
        protocol_lock=V1_LOCK,
    )
    assert bundle["freeze"] == amendment["llm_provenance_freeze"]
    assert len(bundle["records"]) == len(bundle["predictors"]) == 1959
    assert canonical_sha256(bundle["predictors"]) == supplement.EXPECTED_UNIVERSE_SHA256
    forbidden_keys = {
        "target",
        "target_mean",
        "target_rate",
        "iv",
        "correlation",
        "mutual_information",
        "shap",
        "performance",
        "drift",
        "missing_rate",
    }
    for row in bundle["records"]:
        assert not (set(row) & forbidden_keys)
        assert hashlib.sha256(
            str(row["rendered_description"]).encode("utf-8")
        ).hexdigest() == row["description_sha256"]


def test_strict_target_free_ranking_accepts_exact_coverage_and_records_provenance():
    names, records = _synthetic_metadata()
    selector, completions = _strict_selector([names])
    attempts: list[dict[str, object]] = []
    result = selector.rank_target_free(
        records,
        expected_features=names,
        expected_response_model=supplement.FROZEN_LLM_MODEL,
        attempt_recorder=lambda record: attempts.append(dict(record)),
    )
    assert result["selected_features"] == names
    assert result["fallback_used"] is False
    assert result["candidate_coverage"]["unknown_features"] == 0
    assert result["candidate_coverage"]["duplicate_features"] == 0
    assert result["application_attempt"] == 1
    assert completions.calls == 1
    assert len(attempts) == 1 and attempts[0]["valid"] is True
    request = attempts[0]["request"]
    assert request["model"] == supplement.FROZEN_LLM_MODEL
    assert request["seed"] is None
    assert "OPENAI_API_KEY" not in json.dumps(request)


@pytest.mark.parametrize("failure", ["duplicate", "unknown", "missing"])
def test_strict_target_free_ranking_rejects_duplicate_unknown_and_missing(failure):
    names, records = _synthetic_metadata()
    invalid = list(names)
    if failure == "duplicate":
        invalid[-1] = invalid[0]
    elif failure == "unknown":
        invalid[-1] = "hallucinated_feature"
    else:
        invalid = invalid[:-1]
    selector, completions = _strict_selector([invalid, invalid, invalid])
    attempts: list[dict[str, object]] = []
    with pytest.raises(ValueError, match="strict target-free ranking contract"):
        selector.rank_target_free(
            records,
            expected_features=names,
            expected_response_model=supplement.FROZEN_LLM_MODEL,
            attempt_recorder=lambda record: attempts.append(dict(record)),
        )
    assert completions.calls == 3
    assert len(attempts) == 3
    assert all(record["valid"] is False for record in attempts)
    assert not hasattr(selector, "fallback_used_")


def test_target_free_ranking_cache_resume_is_byte_equivalent_and_makes_no_second_call(
    tmp_path: Path,
):
    names, records = _synthetic_metadata()
    first, first_calls = _strict_selector([names])
    prompt = first.build_target_free_prompt(records, expected_features=names)
    freeze = {
        "ordered_feature_universe_sha256": canonical_sha256(names),
        "ordered_rendered_descriptions_sha256": canonical_sha256(
            [row["rendered_description"] for row in records]
        ),
        "rendered_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "provider": "openai",
        "request_model": supplement.FROZEN_LLM_MODEL,
        "required_response_model": supplement.FROZEN_LLM_MODEL,
        "temperature": 0.0,
        "seed": None,
    }
    entry = {
        "authorization": {
            "_authorization_file_sha256": "a" * 64,
            "amendment": {"internal_sha256": "b" * 64},
        },
        "matrix_manifest_sha256": "c" * 64,
        "description_freeze": {
            "freeze": freeze,
            "predictors": names,
            "records": records,
            "prompt": prompt,
        },
        "plan": {"paths": {"matrix_root": str(tmp_path / "matrix")}},
    }
    ranking_a, manifest_a, payload_a = supplement.ensure_target_free_ranking(
        entry=entry,
        output_root=tmp_path / "supplement",
        selector_factory=lambda: first,
    )

    def forbidden_factory():
        pytest.fail("a sealed identity-matching ranking must be reused")

    ranking_b, manifest_b, payload_b = supplement.ensure_target_free_ranking(
        entry=entry,
        output_root=tmp_path / "supplement",
        selector_factory=forbidden_factory,
    )
    assert first_calls.calls == 1
    assert ranking_a == ranking_b == names
    assert manifest_a == manifest_b
    assert payload_a == payload_b
    assert (tmp_path / "supplement/llm_ranking/_SUCCESS").is_file()


def test_stable_core_supervised_components_are_fold_local_and_exactly_five():
    rng = np.random.RandomState(42)
    index = pd.Index(range(1000, 1040), name="fold_training_row")
    X = pd.DataFrame(
        rng.normal(size=(40, 8)).astype("float32"),
        index=index,
        columns=[f"f{i}" for i in range(8)],
    )
    y = pd.Series(([0, 1] * 20), index=index)
    selector = StableCoreLLMFillSelector(
        description_csv_path="unused",
        llm_shared_pool_size=8,
        final_feature_budget=3,
        bootstrap_iterations=5,
        bootstrap_fraction=0.8,
        stability_threshold=0.8,
        random_state=42,
        allow_unranked_padding=False,
    )
    selector.fit_with_authenticated_ranking(
        X,
        y,
        ranked_features=list(X.columns),
        ranking_manifest_sha256="d" * 64,
    )
    assert len(selector.bootstrap_trace_) == 5
    assert [row["random_state"] for row in selector.bootstrap_trace_] == [
        42,
        43,
        44,
        45,
        46,
    ]
    assert all(row["training_index_only"] for row in selector.bootstrap_trace_)
    assert all(row["sample_size"] == 32 for row in selector.bootstrap_trace_)
    assert len(selector.selected_features_) == 3
    assert set(selector.selected_features_).issubset(X.columns)


def test_supplemental_cli_has_one_all_fold_mode_and_old_oot_is_revoked():
    cli = _load_cli()
    args = cli.build_parser().parse_args(
        ["supplemental-dev", "--authorization", "authorization.json"]
    )
    assert args.operation == "supplemental-dev"
    assert not hasattr(args, "phase")
    assert not hasattr(args, "fold_id")
    source = inspect.getsource(cli._run_supervised)
    assert "former classical-only OOT command is revoked" in source
    worker_signature = inspect.signature(supplement.run_supplemental_dev_worker)
    assert "phase" not in worker_signature.parameters
    assert "oot_analysis_plan" not in worker_signature.parameters
    assert supplement.corrected_accounting()[
        "unavailable_semantic_mixed_voter_execution_cells"
    ] == 0


def test_classical_tree_identity_detects_any_byte_change(tmp_path: Path):
    root = tmp_path / "classical"
    (root / "fold_1/evaluations/cell_001").mkdir(parents=True)
    artifact = root / "fold_1/evaluations/cell_001/status.json"
    artifact.write_text("{}\n", encoding="utf-8")
    before = supplement.classical_tree_identity(root)
    artifact.write_text('{"changed":true}\n', encoding="utf-8")
    after = supplement.classical_tree_identity(root)
    assert before["tree_manifest_sha256"] != after["tree_manifest_sha256"]
    assert before["file_count"] == after["file_count"] == 1


def test_recursive_seal_is_atomic_and_writes_completion_marker_last(
    tmp_path: Path, monkeypatch
):
    root = tmp_path / "cell"
    root.mkdir()
    (root / "payload.txt").write_text("payload\n", encoding="utf-8")
    events: list[str] = []
    original_json = supplement.write_json_atomic
    original_text = supplement.write_text_atomic

    def observed_json(path, *args, **kwargs):
        events.append(Path(path).name)
        return original_json(path, *args, **kwargs)

    def observed_text(path, *args, **kwargs):
        events.append(Path(path).name)
        return original_text(path, *args, **kwargs)

    monkeypatch.setattr(supplement, "write_json_atomic", observed_json)
    monkeypatch.setattr(supplement, "write_text_atomic", observed_text)
    identity = {"cell": "synthetic"}
    supplement._seal_recursive_directory(root, identity)
    assert events[-2:] == ["manifest.json", "_SUCCESS"]
    assert supplement._load_recursive_sealed(root, identity) is not None
