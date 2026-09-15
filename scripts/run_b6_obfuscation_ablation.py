#!/usr/bin/env python
"""Prospectively frozen Home Credit B6 feature-name sensitivity experiment.

This is a paired, single-draw sensitivity analysis. The named control and the
obfuscated treatment are generated together from the same outcome-independent
529-feature record pack. Historical rankings, predictions, and manuscript
metrics are not computational inputs. The design varies the rendered exact
candidate-name literals while holding the retained descriptions and lineage
fixed; it does not prove causal isolation, absent memorisation, or absent
semantic recognition.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import inspect
import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
for import_root in (REPOSITORY_ROOT, SOURCE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scripts.b6_obfuscation import (  # noqa: E402
    BACKBONES,
    BOOTSTRAP_REPETITIONS,
    BOOTSTRAP_SEED,
    B6ProtocolError,
    CALL_ORDER_SEED,
    EVIDENCE_MODE,
    GLOBAL_FEATURE_COUNT,
    HOAccessGate,
    HO_ROW_COUNT,
    MAPPING_SEED,
    MODEL_SNAPSHOT,
    PARTITIONS,
    PREFIX_BY_BACKBONE,
    RANKING_BUDGET,
    TEMPERATURE,
    align_paired_predictions,
    assert_request_payload_safe,
    build_global_feature_mapping,
    build_homecredit_definition_records,
    build_pairwise_row,
    obfuscate_definition_records,
    pairwise_frame,
    performance_frame,
    publish_output_frames_atomic,
    rankings_frame,
    refuse_existing_output,
    reverse_feature_mapping,
    scrub_text,
    validate_ranking_response,
)


PROTOCOL_ID = "homecredit_b6_prospective_paired_name_channel_v1"
CONDITIONS = ("named", "obfuscated")
TOTAL_RANKING_CALLS = len(PARTITIONS) * len(CONDITIONS)
DESCRIPTION_PATH = Path("data/homecredit/metadata/columns_description.csv")
EXPECTED_PROMPT_SOURCE_SHA256 = (
    "4edfbae7518eb7f24142d153b8cae8da7d201957c1536a9e6c9ee4d2c35b5e11"
)
EXPECTED_SYSTEM_MESSAGE_SHA256 = (
    "ccdbdda09c8db90514fa7a41cc908130f15bbd048e69c33502e911a91919625f"
)
EXPECTED_RETRY_SUFFIX_SHA256 = (
    "59cc72d9bf1cc78f6d7334c0229c0872712bcb267f9cf6253f80750e8916f5a6"
)
EXPECTED_RESPONSE_FORMAT_SHA256 = (
    "7a11d8c2e26c69210a83a98098e2707bf90cd3c2e70c2e7e1df753fe0952b2e9"
)
EXPECTED_DEFINITION_BUILDER_SOURCE_SHA256 = (
    "48036e66edc264daaa7d63e84c92ed3317955c3e7b2acaa3e01db4d07564183a"
)
EXPECTED_OBFUSCATOR_SOURCE_SHA256 = (
    "b26257ea4291f90caf892792b4f5023436c3767e993cf15197732e45f7fd92e5"
)
EXPECTED_MAPPING_SOURCE_SHA256 = (
    "49ba8ebf1f8110629d43e82c620d1c08ad5069645e1b0a7b4bb1ebd759440d2f"
)

EXPECTED_LR_PARAMS = {
    "solver": "liblinear",
    "max_iter": 1000,
    "class_weight": "balanced",
    "random_state": 42,
}
EXPECTED_CATBOOST_PARAMS = {
    "depth": 10,
    "learning_rate": 0.01,
    "l2_leaf_reg": 95,
    "min_data_in_leaf": 290,
    "colsample_bylevel": 0.9,
    "random_strength": 0.125,
    "grow_policy": "Depthwise",
    "one_hot_max_size": 21,
    "leaf_estimation_method": "Newton",
    "bootstrap_type": "Bernoulli",
    "subsample": 0.55,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "auto_class_weights": "Balanced",
    "iterations": 1500,
    "early_stopping_rounds": 150,
    "verbose": 100,
    "random_state": 42,
    "allow_writing_files": False,
}

PROTOCOL_CONTRACT = {
    "protocol_id": PROTOCOL_ID,
    "dataset": "homecredit",
    "comparison": "prospective_named_control_vs_obfuscated_treatment",
    "historical_results_as_inputs": False,
    "conditions": CONDITIONS,
    "partitions": PARTITIONS,
    "ranking_calls_per_condition": len(PARTITIONS),
    "total_ranking_calls": TOTAL_RANKING_CALLS,
    "candidate_universe_per_call": GLOBAL_FEATURE_COUNT,
    "ranking_budget": RANKING_BUDGET,
    "prefix_by_backbone": PREFIX_BY_BACKBONE,
    "model_snapshot": MODEL_SNAPSHOT,
    "temperature": TEMPERATURE,
    "evidence_mode": EVIDENCE_MODE,
    "mapping_seed": MAPPING_SEED,
    "mapping_source_sha256": EXPECTED_MAPPING_SOURCE_SHA256,
    "definition_builder_source_sha256": EXPECTED_DEFINITION_BUILDER_SOURCE_SHA256,
    "obfuscator_source_sha256": EXPECTED_OBFUSCATOR_SOURCE_SHA256,
    "call_order_seed": CALL_ORDER_SEED,
    "model_seed": 42,
    "folds": {"count": 5, "gap_unique_time_groups": 1},
    "stable_tie_rule": "time_ascending_then_canonical_SK_ID_CURR_ascending",
    "ho_rows": HO_ROW_COUNT,
    "bootstrap": {
        "repetitions": BOOTSTRAP_REPETITIONS,
        "seed": BOOTSTRAP_SEED,
        "stratified": True,
        "shared_indices": True,
    },
    "claim_boundary": (
        "single-draw exact-name-channel sensitivity conditional on retained "
        "definitions, source concepts, and lineage"
    ),
}
PROTOCOL_SHA256 = hashlib.sha256(
    json.dumps(PROTOCOL_CONTRACT, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()


@dataclass
class PreflightReport:
    checks: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)

    def pass_check(self, code: str, detail: str) -> None:
        self.checks.append(f"PASS [{code}] {detail}")

    def note(self, code: str, detail: str) -> None:
        self.notes.append(f"NOTE [{code}] {detail}")

    def block(self, code: str, detail: str) -> None:
        self.blockers.append(f"BLOCK [{code}] {detail}")

    @property
    def ok(self) -> bool:
        return not self.blockers

    def render(self) -> str:
        return "\n".join(
            [
                f"B6 prospective protocol: {PROTOCOL_ID}",
                f"protocol_sha256={PROTOCOL_SHA256}",
                *self.checks,
                *self.notes,
                *self.blockers,
                f"RESULT: {'PASS' if self.ok else 'FAIL'} "
                f"({len(self.checks)} checks passed, {len(self.blockers)} blockers)",
            ]
        )


@dataclass
class FrozenPipeline:
    condition: str
    backbone: str
    selected_features: tuple[str, ...]
    preprocessor: Any
    model: Any
    predict_proba: Callable[[Any, Any], np.ndarray]


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise B6ProtocolError("PyYAML is required for B6") from exc
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise B6ProtocolError(f"expected a YAML mapping in {path}")
    return value


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    return _sha256_text(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
            default=str,
        )
    )


def _csv_header(path: Path) -> list[str]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return next(csv.reader(handle))
    except UnicodeDecodeError:
        with path.open("r", encoding="latin1", newline="") as handle:
            return next(csv.reader(handle))


def _inspect_protocol_and_models(root: Path, report: PreflightReport) -> None:
    base_path = root / "configs/base.yaml"
    protocol_path = root / "configs/protocols/credit_scoring_extension_v1.yaml"
    lr_path = root / "configs/models/lr.yaml"
    catboost_path = root / "configs/models/catboost.yaml"
    required = (base_path, protocol_path, lr_path, catboost_path)
    missing = [str(path.relative_to(root)) for path in required if not path.is_file()]
    if missing:
        report.block("frozen_configs", f"missing configuration files: {missing}")
        return
    base = _load_yaml(base_path)
    protocol = _load_yaml(protocol_path)
    lr = _load_yaml(lr_path).get("model_params", {}).get("lr")
    catboost = _load_yaml(catboost_path).get("model_params", {}).get("catboost")
    dataset = protocol.get("datasets", {}).get("homecredit", {})
    conditions_ok = (
        base.get("dataset_name") == "homecredit"
        and base.get("n_splits") == 5
        and base.get("cv_gap_groups") == 1
        and base.get("feature_budgets") == {"lr": 20, "catboost": 40}
        and base.get("model_params", {}).get("lr") == EXPECTED_LR_PARAMS
        and base.get("model_params", {}).get("catboost") == EXPECTED_CATBOOST_PARAMS
        and lr == EXPECTED_LR_PARAMS
        and catboost == EXPECTED_CATBOOST_PARAMS
        and dataset.get("candidate_universe", {}).get("count") == GLOBAL_FEATURE_COUNT
        and dataset.get("row_identifier") == "SK_ID_CURR"
        and dataset.get("target", {}).get("column") == "TARGET"
        and dataset.get("split", {}).get("dev", {}).get("rows") == 99_092
        and dataset.get("split", {}).get("oot", {}).get("rows") == HO_ROW_COUNT
    )
    if conditions_ok:
        report.pass_check(
            "frozen_configs",
            "exact LR/CatBoost parameters, five-fold gap-1 split, 529-feature "
            "universe, SK_ID_CURR/TARGET, DEV=99092 and HO=120053 are frozen",
        )
    else:
        report.block("frozen_configs", "configuration values differ from the B6 contract")
    report.pass_check(
        "prospective_design",
        f"two contemporaneous conditions, six calls each ({TOTAL_RANKING_CALLS} total), "
        "full-pool definitions-only input, deterministic condition order and no "
        "historical score/ranking input are frozen",
    )
    report.note(
        "historical_scope",
        "This estimates a new prospective named-versus-obfuscated contrast. It is not "
        "an exact reproduction or correction of the conflicting published run.",
    )


def _inspect_data_contract(root: Path, report: PreflightReport) -> None:
    raw_dir = root / "data/homecredit/raw"
    schema_path = root / "data/homecredit/metadata/raw_schema_snapshot.json"
    description_path = root / DESCRIPTION_PATH
    row_contract_path = root / "configs/protocols/row_alignment_contract_v1.json"
    expected_tables = {
        "application_train",
        "bureau",
        "bureau_balance",
        "credit_card_balance",
        "installments_payments",
        "POS_CASH_balance",
        "previous_application",
    }
    if not schema_path.is_file() or not description_path.is_file() or not row_contract_path.is_file():
        report.block("data_contract", "schema, description, or row-alignment contract is missing")
        return
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema_tables = {Path(str(item["name"])).stem for item in schema.get("files", [])}
    missing_tables = expected_tables - schema_tables
    missing_files = [name for name in expected_tables if not (raw_dir / f"{name}.csv").is_file()]
    if missing_files or "application_train" in missing_tables or "previous_application" in missing_tables:
        report.block(
            "data_contract",
            f"Home Credit source files/schema are incomplete: files={sorted(missing_files)}, "
            f"schema={sorted(missing_tables)}",
        )
        return
    application_header = set(_csv_header(raw_dir / "application_train.csv"))
    previous_header = set(_csv_header(raw_dir / "previous_application.csv"))
    description_header = {column.casefold() for column in _csv_header(description_path)}
    row_contract = json.loads(row_contract_path.read_text(encoding="utf-8"))[
        "datasets"
    ]["homecredit"]
    if (
        not missing_tables
        and {"SK_ID_CURR", "TARGET"} <= application_header
        and {"SK_ID_CURR", "DAYS_DECISION"} <= previous_header
        and {"table", "row", "description"} <= description_header
        and row_contract["dev"]["row_count"] == 99_092
        and row_contract["oot"]["row_count"] == HO_ROW_COUNT
        and row_contract["dev_oot_id_overlap_count"] == 0
    ):
        report.pass_check(
            "data_contract",
            "header-only inspection verifies all seven source tables, descriptions, "
            "time derivation, disjoint DEV/HO identities, and frozen row contracts "
            f"(schema_sha256={_sha256_text(schema_path.read_text(encoding='utf-8'))})",
        )
    else:
        report.block("data_contract", "Home Credit data/row contract differs from B6")


def _inspect_prompt_contract(report: PreflightReport) -> None:
    from credit_risk_fs.selectors.llm_screening import LLMSelector

    selector = LLMSelector(
        description_csv_path="unused",
        model=MODEL_SNAPSHOT,
        temperature=TEMPERATURE,
        ranking_budget=RANKING_BUDGET,
        max_features=RANKING_BUDGET,
        iv_filter_kwargs={},
    )
    observed = {
        "prompt_source": _sha256_text(inspect.getsource(LLMSelector.build_target_free_prompt)),
        "system_message": _sha256_text(LLMSelector.TARGET_FREE_SYSTEM_MESSAGE),
        "retry_suffix": _sha256_text(LLMSelector.TARGET_FREE_RETRY_SUFFIX),
        "response_format": _canonical_json_sha256(selector.target_free_response_format()),
        "definition_builder": _sha256_text(
            inspect.getsource(build_homecredit_definition_records)
        ),
        "obfuscator": _sha256_text(inspect.getsource(obfuscate_definition_records)),
        "mapping": _sha256_text(inspect.getsource(build_global_feature_mapping)),
    }
    expected = {
        "prompt_source": EXPECTED_PROMPT_SOURCE_SHA256,
        "system_message": EXPECTED_SYSTEM_MESSAGE_SHA256,
        "retry_suffix": EXPECTED_RETRY_SUFFIX_SHA256,
        "response_format": EXPECTED_RESPONSE_FORMAT_SHA256,
        "definition_builder": EXPECTED_DEFINITION_BUILDER_SOURCE_SHA256,
        "obfuscator": EXPECTED_OBFUSCATOR_SOURCE_SHA256,
        "mapping": EXPECTED_MAPPING_SOURCE_SHA256,
    }
    ranking_source = inspect.getsource(LLMSelector.rank_target_free)
    retry_contract = (
        "maximum_attempts != 3" in ranking_source
        and "client.chat.completions.create" in ranking_source
        and "if payload is not None" in ranking_source
        and "failed the strict target-free ranking contract" in ranking_source
    )
    if observed == expected and retry_contract:
        report.pass_check(
            "prompt_api_contract",
            "canonical target-free prompt, system message, retry suffix, strict JSON "
            "schema, exact snapshot, temperature zero, three attempts and no fallback "
            "match the prospective freeze",
        )
    else:
        report.block(
            "prompt_api_contract",
            f"prompt/API implementation drift detected; observed_hashes={observed}",
        )


def _inspect_runtime_contract(report: PreflightReport) -> None:
    try:
        from credit_risk_fs.evaluation.paired_inference import (
            BOOTSTRAP_REPETITIONS as canonical_repetitions,
            BOOTSTRAP_SEED as canonical_seed,
            paired_delong_test,
            paired_stratified_bootstrap,
        )
        from credit_risk_fs.experiments.lendingclub_identity import stable_chronological_order
        from credit_risk_fs.models._cv_utils import GroupedTimeSeriesSplit
        from credit_risk_fs.models.registry import get_model_bundle
        from credit_risk_fs.pipelines.common import (
            prepare_voting_pilot_dev_data,
            prepare_voting_research_oot_data,
        )
        from credit_risk_fs.preprocessing.encoding import Preprocessor

        symbols = (
            paired_delong_test,
            paired_stratified_bootstrap,
            stable_chronological_order,
            GroupedTimeSeriesSplit,
            get_model_bundle,
            prepare_voting_pilot_dev_data,
            prepare_voting_research_oot_data,
            Preprocessor,
        )
        if not all(callable(symbol) for symbol in symbols):
            raise TypeError("a canonical runtime symbol is not callable")
        if (canonical_repetitions, canonical_seed) != (
            BOOTSTRAP_REPETITIONS,
            BOOTSTRAP_SEED,
        ):
            raise ValueError("paired bootstrap constants differ")
    except Exception as exc:
        report.block("canonical_runtime", f"canonical runtime is unavailable: {exc}")
    else:
        report.pass_check(
            "canonical_runtime",
            "canonical loaders, stable time/ID ordering, grouped folds, preprocessing, "
            "model wrappers, alignment, paired bootstrap and DeLong are importable",
        )


def _inspect_credential(root: Path, report: PreflightReport) -> None:
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        report.block("api_credential", f"python-dotenv is unavailable: {exc}")
        return
    load_dotenv(root / ".env", override=False)
    if os.getenv("OPENAI_API_KEY"):
        report.pass_check(
            "api_credential", "OPENAI_API_KEY presence is confirmed; its value is not printed"
        )
    else:
        report.block("api_credential", "OPENAI_API_KEY is absent from environment/.env")


def run_preflight(root: Path, output_dir: Path) -> PreflightReport:
    """Complete the lightweight gate before loading data or constructing a client."""

    refuse_existing_output(output_dir)
    report = PreflightReport()
    root = root.resolve()
    if root != REPOSITORY_ROOT.resolve():
        report.block("repository_root", f"unexpected repository root: {root}")
        return report
    required_output = (root / "B6_artifacts").resolve()
    if output_dir.resolve() != required_output:
        report.block("output", f"output must be exactly {required_output}")
    else:
        report.pass_check("output", f"new output target is available: {required_output}")
    _inspect_protocol_and_models(root, report)
    _inspect_data_contract(root, report)
    _inspect_prompt_contract(report)
    _inspect_runtime_contract(report)
    _inspect_credential(root, report)
    return report


class _GuardedCompletions:
    def __init__(
        self,
        delegate: Any,
        *,
        condition: str,
        candidate_records: Sequence[Mapping[str, Any]],
        candidate_ids: Sequence[str],
        original_feature_names: Sequence[str],
    ) -> None:
        self._delegate = delegate
        self._condition = condition
        self._candidate_records = candidate_records
        self._candidate_ids = candidate_ids
        self._original_feature_names = original_feature_names

    def create(self, **kwargs: Any) -> Any:
        if kwargs.get("model") != MODEL_SNAPSHOT or kwargs.get("temperature") != TEMPERATURE:
            raise B6ProtocolError("model snapshot or temperature changed at the API boundary")
        assert_request_payload_safe(
            kwargs,
            original_feature_names=(
                self._original_feature_names if self._condition == "obfuscated" else ()
            ),
            candidate_records=self._candidate_records,
            expected_candidate_ids=self._candidate_ids,
            evidence_mode=EVIDENCE_MODE,
            ranking_budget=RANKING_BUDGET,
        )
        return self._delegate.create(**kwargs)


class _GuardedChat:
    def __init__(self, delegate: Any, **guard_kwargs: Any) -> None:
        self.completions = _GuardedCompletions(delegate.completions, **guard_kwargs)


class _GuardedClient:
    def __init__(self, delegate: Any, **guard_kwargs: Any) -> None:
        self.chat = _GuardedChat(delegate.chat, **guard_kwargs)


def _selector() -> Any:
    from credit_risk_fs.selectors.llm_screening import LLMSelector

    return LLMSelector(
        description_csv_path=str(REPOSITORY_ROOT / DESCRIPTION_PATH),
        cache_dir=str(Path(tempfile.gettempdir()) / "b6_no_cache"),
        model=MODEL_SNAPSHOT,
        temperature=TEMPERATURE,
        max_features=RANKING_BUDGET,
        ranking_budget=RANKING_BUDGET,
        feature_budget=RANKING_BUDGET,
        shared_pool_size=RANKING_BUDGET,
        prompt_version=PROTOCOL_ID,
        iv_filter_kwargs={},
    )


def _rank_condition(
    *,
    client: Any,
    condition: str,
    records: Sequence[Mapping[str, Any]],
    candidate_ids: Sequence[str],
    original_feature_names: Sequence[str],
) -> tuple[list[str], dict[str, Any]]:
    selector = _selector()
    selector._client = _GuardedClient(
        client,
        condition=condition,
        candidate_records=records,
        candidate_ids=candidate_ids,
        original_feature_names=original_feature_names,
    )
    attempts: list[dict[str, Any]] = []
    payload = selector.rank_target_free(
        records,
        expected_features=candidate_ids,
        expected_response_model=MODEL_SNAPSHOT,
        attempt_recorder=lambda record: attempts.append(dict(record)),
        maximum_attempts=3,
    )
    ranked = list(
        validate_ranking_response(
            payload,
            candidate_ids=candidate_ids,
            original_feature_names=(
                original_feature_names if condition == "obfuscated" else ()
            ),
        )
    )
    return ranked, {
        "application_attempt": int(payload["application_attempt"]),
        "prompt_sha256": str(payload["prompt_sha256"]),
        "response_id_sha256": _sha256_text(str(payload.get("response_id", ""))),
        "attempt_count": len(attempts),
    }


def _freeze_record_pack(
    candidate_features: Sequence[str], dtypes: Mapping[str, Any]
) -> tuple[dict[str, str], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    mapping = build_global_feature_mapping(candidate_features)
    named_records = build_homecredit_definition_records(
        candidate_features,
        dtypes=dtypes,
        description_csv_path=REPOSITORY_ROOT / DESCRIPTION_PATH,
    )
    opaque_records = obfuscate_definition_records(named_records, mapping)
    named_ids = list(candidate_features)
    opaque_ids = [mapping[name] for name in named_ids]
    named_prompt = _selector().build_target_free_prompt(
        named_records, expected_features=named_ids
    )
    opaque_prompt = _selector().build_target_free_prompt(
        opaque_records, expected_features=opaque_ids
    )
    if opaque_prompt != scrub_text(named_prompt, mapping):
        raise B6ProtocolError(
            "named and obfuscated prompts differ by more than literal candidate-name replacement"
        )
    freeze = {
        "protocol_sha256": PROTOCOL_SHA256,
        "candidate_count": len(named_ids),
        "candidate_order_sha256": _canonical_json_sha256(named_ids),
        "mapping_assignment_sha256": _canonical_json_sha256(
            [[name, mapping[name]] for name in sorted(mapping)]
        ),
        "named_records_sha256": _canonical_json_sha256(named_records),
        "obfuscated_records_sha256": _canonical_json_sha256(opaque_records),
        "named_prompt_sha256": _sha256_text(named_prompt),
        "obfuscated_prompt_sha256": _sha256_text(opaque_prompt),
    }
    return mapping, named_records, opaque_records, freeze


def _make_rankings(
    *,
    named_records: Sequence[Mapping[str, Any]],
    opaque_records: Sequence[Mapping[str, Any]],
    candidate_features: Sequence[str],
    mapping: Mapping[str, str],
) -> tuple[dict[str, dict[str, list[str]]], list[dict[str, Any]]]:
    from openai import OpenAI

    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise B6ProtocolError("OPENAI_API_KEY disappeared after preflight")
    client = OpenAI(api_key=key)
    records = {"named": named_records, "obfuscated": opaque_records}
    ids = {
        "named": list(candidate_features),
        "obfuscated": [mapping[name] for name in candidate_features],
    }
    rankings: dict[str, dict[str, list[str]]] = {condition: {} for condition in CONDITIONS}
    diagnostics: list[dict[str, Any]] = []
    generator = np.random.default_rng(CALL_ORDER_SEED)
    for partition in PARTITIONS:
        condition_order = [str(value) for value in generator.permutation(CONDITIONS)]
        for sequence, condition in enumerate(condition_order, start=1):
            print(
                f"Ranking {partition}: condition={condition}, paired_sequence={sequence}/2",
                flush=True,
            )
            ranking, diagnostic = _rank_condition(
                client=client,
                condition=condition,
                records=records[condition],
                candidate_ids=ids[condition],
                original_feature_names=list(candidate_features),
            )
            rankings[condition][partition] = ranking
            ranking_sha256 = _canonical_json_sha256(ranking)
            print(
                f"Accepted {partition}/{condition}: attempt="
                f"{diagnostic['application_attempt']}, ranking_sha256={ranking_sha256}",
                flush=True,
            )
            diagnostics.append(
                {
                    "partition": partition,
                    "condition": condition,
                    "paired_sequence": sequence,
                    "ranking_sha256": ranking_sha256,
                    **diagnostic,
                }
            )
    if any(set(value) != set(PARTITIONS) for value in rankings.values()):
        raise B6ProtocolError("ranking phase did not complete all twelve calls")
    return rankings, diagnostics


def _ordered_dev_frame(data: Any, candidate_features: Sequence[str]) -> pd.DataFrame:
    from credit_risk_fs.experiments.lendingclub_identity import stable_chronological_order

    if data.candidate_universe is None or tuple(data.candidate_universe) != tuple(
        candidate_features
    ):
        raise B6ProtocolError("loaded DEV candidate universe/order changed")
    frame = data.X.loc[:, candidate_features].copy()
    frame["__target__"] = data.y.to_numpy()
    frame["__time__"] = data.time_values.to_numpy()
    frame["__stable_id__"] = data.stable_row_ids.to_numpy()
    ordered = stable_chronological_order(
        frame, time_column="__time__", identity_column="__stable_id__"
    )
    if len(ordered) != 99_092 or ordered["__stable_id__"].duplicated().any():
        raise B6ProtocolError("ordered DEV identity contract failed")
    return ordered


def _model_parameters(root: Path) -> dict[str, dict[str, Any]]:
    return {
        "lr": dict(_load_yaml(root / "configs/models/lr.yaml")["model_params"]["lr"]),
        "catboost": dict(
            _load_yaml(root / "configs/models/catboost.yaml")["model_params"]["catboost"]
        ),
    }


def _fit_one(
    *,
    backbone: str,
    params: Mapping[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame | None,
    y_validation: pd.Series | None,
) -> tuple[Any, Any, Callable[[Any, Any], np.ndarray], np.ndarray | None]:
    from credit_risk_fs.models.registry import get_model_bundle
    from credit_risk_fs.preprocessing.encoding import Preprocessor

    preprocessor = Preprocessor()
    transformed_train = preprocessor.fit_transform(X_train)
    transformed_validation = (
        preprocessor.transform(X_validation) if X_validation is not None else None
    )
    get_model, train_model, predict_proba, _ = get_model_bundle(
        backbone, model_kwargs=dict(params)
    )
    model = train_model(
        get_model(),
        transformed_train,
        y_train,
        transformed_validation,
        y_validation,
    )
    validation_scores = (
        np.asarray(predict_proba(model, transformed_validation), dtype=float)
        if transformed_validation is not None
        else None
    )
    return preprocessor, model, predict_proba, validation_scores


def _original_rankings(
    rankings: Mapping[str, Mapping[str, Sequence[str]]],
    reverse_mapping: Mapping[str, str],
) -> dict[str, dict[str, tuple[str, ...]]]:
    translated: dict[str, dict[str, tuple[str, ...]]] = {
        condition: {} for condition in CONDITIONS
    }
    for partition in PARTITIONS:
        translated["named"][partition] = tuple(rankings["named"][partition])
        translated["obfuscated"][partition] = tuple(
            reverse_mapping[item] for item in rankings["obfuscated"][partition]
        )
    return translated


def _fit_cv_and_full_dev(
    *,
    ordered_dev: pd.DataFrame,
    rankings: Mapping[str, Mapping[str, Sequence[str]]],
    params: Mapping[str, Mapping[str, Any]],
) -> tuple[
    dict[str, dict[str, list[float]]],
    dict[tuple[str, str], FrozenPipeline],
    HOAccessGate,
]:
    from credit_risk_fs.models._cv_utils import GroupedTimeSeriesSplit

    features = ordered_dev.drop(columns=["__target__", "__time__", "__stable_id__"])
    target = ordered_dev["__target__"].astype("int8")
    folds = list(
        GroupedTimeSeriesSplit(n_splits=5, gap=1).split(
            ordered_dev["__time__"].to_numpy()
        )
    )
    if len(folds) != 5:
        raise B6ProtocolError(f"canonical splitter returned {len(folds)} folds; expected 5")
    fold_aucs: dict[str, dict[str, list[float]]] = {
        condition: {backbone: [] for backbone in BACKBONES} for condition in CONDITIONS
    }
    for fold_number, (train_index, validation_index) in enumerate(folds, start=1):
        partition = f"fold{fold_number}"
        for condition in CONDITIONS:
            for backbone in BACKBONES:
                k = PREFIX_BY_BACKBONE[backbone]
                selected = tuple(rankings[condition][partition][:k])
                if len(selected) != k or len(set(selected)) != k:
                    raise B6ProtocolError(
                        f"invalid {condition}/{backbone}/{partition} ranking prefix"
                    )
                print(
                    f"Fitting {partition}: condition={condition}, backbone={backbone}, K={k}",
                    flush=True,
                )
                _, _, _, scores = _fit_one(
                    backbone=backbone,
                    params=params[backbone],
                    X_train=features.iloc[train_index].loc[:, selected],
                    y_train=target.iloc[train_index],
                    X_validation=features.iloc[validation_index].loc[:, selected],
                    y_validation=target.iloc[validation_index],
                )
                if scores is None:
                    raise AssertionError("fold fitting did not produce validation scores")
                fold_aucs[condition][backbone].append(
                    float(roc_auc_score(target.iloc[validation_index], scores))
                )
                gc.collect()

    full_subsets = {
        (condition, backbone): tuple(
            rankings[condition]["full_dev"][: PREFIX_BY_BACKBONE[backbone]]
        )
        for condition in CONDITIONS
        for backbone in BACKBONES
    }
    gate = HOAccessGate()
    gate.mark_subset_frozen()
    frozen: dict[tuple[str, str], FrozenPipeline] = {}
    for condition in CONDITIONS:
        for backbone in BACKBONES:
            selected = full_subsets[(condition, backbone)]
            print(
                f"Refitting full DEV: condition={condition}, backbone={backbone}, "
                f"K={len(selected)}",
                flush=True,
            )
            preprocessor, model, predict_proba, _ = _fit_one(
                backbone=backbone,
                params=params[backbone],
                X_train=features.loc[:, selected],
                y_train=target,
                X_validation=None,
                y_validation=None,
            )
            frozen[(condition, backbone)] = FrozenPipeline(
                condition=condition,
                backbone=backbone,
                selected_features=selected,
                preprocessor=preprocessor,
                model=model,
                predict_proba=predict_proba,
            )
    if len(frozen) != len(CONDITIONS) * len(BACKBONES):
        raise B6ProtocolError("not all four full-DEV pipelines were frozen")
    gate.mark_model_frozen()
    gate.assert_access_allowed()
    return fold_aucs, frozen, gate


def _score_frozen_pipeline(pipeline: FrozenPipeline, oot_features: pd.DataFrame) -> np.ndarray:
    transformed = pipeline.preprocessor.transform(
        oot_features.loc[:, pipeline.selected_features]
    )
    scores = np.asarray(pipeline.predict_proba(pipeline.model, transformed), dtype=float)
    if len(scores) != HO_ROW_COUNT or not np.isfinite(scores).all():
        raise B6ProtocolError(
            f"{pipeline.condition}/{pipeline.backbone} produced an invalid HO score vector"
        )
    return scores


def _score_and_infer(
    *,
    root: Path,
    candidate_features: Sequence[str],
    frozen: Mapping[tuple[str, str], FrozenPipeline],
    ho_gate: HOAccessGate,
) -> list[dict[str, Any]]:
    from credit_risk_fs.pipelines.common import prepare_voting_research_oot_data

    ho_gate.assert_access_allowed()
    selected_union = {
        feature for pipeline in frozen.values() for feature in pipeline.selected_features
    }
    projection = [feature for feature in candidate_features if feature in selected_union]
    if set(projection) != selected_union:
        raise B6ProtocolError("frozen HO projection does not cover every selected feature")
    print(
        f"All full-DEV pipelines frozen; loading HO projection with {len(projection)} features",
        flush=True,
    )
    oot = prepare_voting_research_oot_data(
        root,
        dataset="homecredit",
        projected_candidate_features=projection,
        csv_chunk_rows=25_000,
    )
    if len(oot.X) != HO_ROW_COUNT or len(oot.stable_row_ids) != HO_ROW_COUNT:
        raise B6ProtocolError("canonical HO loader did not return exactly 120053 rows")
    pairwise_rows: list[dict[str, Any]] = []
    for backbone in BACKBONES:
        scores_a = _score_frozen_pipeline(frozen[("obfuscated", backbone)], oot.X)
        scores_b = _score_frozen_pipeline(frozen[("named", backbone)], oot.X)
        method_a = pd.DataFrame(
            {
                "stable_row_id": oot.stable_row_ids,
                "target": oot.y,
                "prediction_probability": scores_a,
            }
        )
        method_b = pd.DataFrame(
            {
                "stable_row_id": oot.stable_row_ids,
                "target": oot.y,
                "prediction_probability": scores_b,
            }
        )
        aligned = align_paired_predictions(method_a, method_b, production=True)
        pairwise_rows.append(
            build_pairwise_row(backbone=backbone, aligned=aligned, production=True)
        )
    return pairwise_rows


def execute_prospective_ablation(root: Path, output_dir: Path) -> Path:
    """Execute the prospectively frozen experiment after preflight succeeds."""

    from credit_risk_fs.pipelines.common import prepare_voting_pilot_dev_data

    np.random.seed(42)
    with tempfile.TemporaryDirectory(prefix="b6-prospective-"):
        print("Loading canonical DEV only; HO remains inaccessible", flush=True)
        dev = prepare_voting_pilot_dev_data(
            root,
            dataset="homecredit",
            csv_chunk_rows=25_000,
        )
        candidate_features = list(dev.candidate_universe or ())
        if (
            len(candidate_features) != GLOBAL_FEATURE_COUNT
            or len(set(candidate_features)) != GLOBAL_FEATURE_COUNT
        ):
            raise B6ProtocolError("runtime Home Credit universe is not 529 unique features")
        if list(dev.X.columns) != candidate_features:
            raise B6ProtocolError("DEV matrix and candidate-universe order differ")

        mapping, named_records, opaque_records, freeze = _freeze_record_pack(
            candidate_features, dev.X.dtypes.to_dict()
        )
        print(
            "Frozen definition/prompt pack: "
            f"named_prompt_sha256={freeze['named_prompt_sha256']}, "
            f"obfuscated_prompt_sha256={freeze['obfuscated_prompt_sha256']}",
            flush=True,
        )
        rankings, diagnostics = _make_rankings(
            named_records=named_records,
            opaque_records=opaque_records,
            candidate_features=candidate_features,
            mapping=mapping,
        )
        if len(diagnostics) != TOTAL_RANKING_CALLS:
            raise B6ProtocolError("ranking diagnostics do not contain exactly twelve calls")

        reverse = reverse_feature_mapping(mapping)
        original = _original_rankings(rankings, reverse)
        ordered_dev = _ordered_dev_frame(dev, candidate_features)
        fold_aucs, frozen, ho_gate = _fit_cv_and_full_dev(
            ordered_dev=ordered_dev,
            rankings=original,
            params=_model_parameters(root),
        )
        del ordered_dev, dev
        gc.collect()
        pairwise_rows = _score_and_infer(
            root=root,
            candidate_features=candidate_features,
            frozen=frozen,
            ho_gate=ho_gate,
        )
        pairwise_by_backbone = {row["backbone"]: row for row in pairwise_rows}
        performance_rows = []
        for backbone in BACKBONES:
            aucs = fold_aucs["obfuscated"][backbone]
            performance_rows.append(
                {
                    "dataset": "homecredit",
                    "backbone": backbone,
                    "K": PREFIX_BY_BACKBONE[backbone],
                    "ho_auc": pairwise_by_backbone[backbone]["auc_A"],
                    **{
                        f"fold{fold}_auc": auc
                        for fold, auc in enumerate(aucs, start=1)
                    },
                }
            )
        output = publish_output_frames_atomic(
            output_dir,
            rankings_frame(rankings["obfuscated"], reverse),
            performance_frame(performance_rows),
            pairwise_frame(pairwise_rows),
        )
    print(
        "Published a single-draw name-obfuscation sensitivity analysis; retained "
        "descriptions and lineage remain semantically recognizable.",
        flush=True,
    )
    for path in sorted(output.iterdir(), key=lambda item: item.name):
        print(path.resolve(), flush=True)
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("./B6_artifacts"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        refuse_existing_output(args.output_dir)
        report = run_preflight(REPOSITORY_ROOT, args.output_dir)
        print(report.render(), file=sys.stdout if report.ok else sys.stderr, flush=True)
        if not report.ok:
            return 2
        if args.preflight_only:
            return 0
        execute_prospective_ablation(REPOSITORY_ROOT, args.output_dir)
        return 0
    except Exception as exc:
        print(f"B6 failed safely: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
