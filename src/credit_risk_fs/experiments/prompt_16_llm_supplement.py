"""Authenticated, resumable Prompt-16 two-method supplemental DEV controller.

This module has one execution purpose: add the historically authenticated
``llm`` and ``stable_core_llm_fill`` methods to the five frozen DEV folds.  It
does not expose a phase argument and contains no OOT loader or OOT worker.
"""

from __future__ import annotations

from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Callable, Mapping, Sequence
import uuid

import numpy as np
import pandas as pd

from credit_risk_fs.data.homecredit_model_stability_2024.adapter import (
    validate_output_manifest,
)
from credit_risk_fs.data.homecredit_model_stability_2024.contract import (
    AdapterContract,
    canonical_sha256,
    file_sha256,
    load_adapter_contract,
)
from credit_risk_fs.experiments.atomic_io import (
    write_csv_atomic,
    write_json_atomic,
    write_parquet_atomic,
    write_text_atomic,
)
from credit_risk_fs.experiments.prompt_16_third_dataset import (
    NON_PREDICTORS,
    Prompt16ExecutionError,
    _archive_incomplete,
    _check_stop,
    _expected_scope,
    _fit_and_evaluate,
    _locked_alignment_summary,
    _publish_stage,
    _read_date_slice,
    _validate_scope_frame,
    load_execution_plan,
)
from credit_risk_fs.preprocessing.encoding import OriginalFeatureNumericEncoder
from credit_risk_fs.selectors.llm_screening import LLMSelector
from credit_risk_fs.selectors.stable_core_llm_fill import StableCoreLLMFillSelector


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "prompt_16_two_llm_method_supplement_v2"
AMENDMENT_SCHEMA_VERSION = "homecredit_model_stability_2024_amendment_v2"
AUTHORIZATION_SCHEMA_VERSION = "prompt_16_supplemental_dev_authorization_v2"
EXPECTED_BLOCKER_COMMIT = "b062362e999a9bec516996c6e3fad4fcb80d70dd"
EXPECTED_BLOCKER_SHA256 = (
    "51cd24903642a25cad1b7b1ebd541ec29a9478e5f5a957035ee490ee49d6729d"
)
EXPECTED_V1_FILE_SHA256 = (
    "e4b9f9f13286f15db0887c9dead09eb7e13f7912af786f2f2bc9c53d126b1860"
)
EXPECTED_V1_INTERNAL_SHA256 = (
    "638e1fa2aa54bf98b771206b56ac13f6a6b77e2093deb291b794081d1a475df6"
)
EXPECTED_MATRIX_MANIFEST_SHA256 = (
    "b5dc28de931e39a5a554c6ca2ff639e6af2705c106ecf1b0f077e5caafa02690"
)
EXPECTED_UNIVERSE_SHA256 = (
    "882e958aacfb0076ed7291ea8eee86e87b4d1b2d91ed8ad1d9ac7c896eb2681a"
)
EXPECTED_CLASSICAL_TREE_SHA256 = (
    "c956db2b2bb810805a1668916bf12c56f96ed994b03cd2a5c4acfde6fc6bd6ba"
)
EXPECTED_CLASSICAL_FILE_COUNT = 1371
EXPECTED_CLASSICAL_BYTE_COUNT = 351_363_036
FROZEN_LLM_MODEL = "gpt-4.1-mini-2025-04-14"
FROZEN_LLM_TEMPERATURE = 0.0
FROZEN_LLM_RANKING_BUDGET = 100
FROZEN_METHODS = ("llm", "stable_core_llm_fill")
FROZEN_MODELS = ("lr", "catboost")
FROZEN_FEATURE_BUDGETS = {"lr": 20, "catboost": 40}
FROZEN_FOLDS = (1, 2, 3, 4, 5)
FORBIDDEN_PROMPT_FIELD_TOKENS = (
    "target_rate",
    "target_mean",
    "information_value",
    "mutual_information",
    "shap_value",
    "model_importance",
    "dev_performance",
    "oot_performance",
    "missing_rate",
    "correlation",
    "drift_value",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Prompt16ExecutionError(f"unreadable JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise Prompt16ExecutionError(f"expected JSON object: {path}")
    return value


def _load_self_authenticated_json(
    path: str | Path,
    *,
    schema_version: str,
    status: str,
) -> tuple[dict[str, Any], str, str]:
    candidate = Path(path)
    payload = _json(candidate)
    if payload.get("schema_version") != schema_version:
        raise Prompt16ExecutionError(f"unsupported authenticated schema: {candidate}")
    if payload.get("status") != status:
        raise Prompt16ExecutionError(f"authenticated artifact status is not usable: {candidate}")
    claimed = payload.get("artifact_authentication_sha256")
    unsigned = dict(payload)
    unsigned.pop("artifact_authentication_sha256", None)
    observed = canonical_sha256(unsigned)
    if claimed != observed:
        raise Prompt16ExecutionError(f"internal authentication mismatch: {candidate}")
    return payload, file_sha256(candidate), observed


def supplemental_cells() -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for order, (method, model) in enumerate(
        (
            ("llm", "lr"),
            ("llm", "catboost"),
            ("stable_core_llm_fill", "lr"),
            ("stable_core_llm_fill", "catboost"),
        ),
        start=31,
    ):
        cells.append(
            {
                "configuration_order": order,
                "configuration_id": f"p16v2-c{order:03d}-{method}-{model}",
                "method_id": method,
                "implementation": (
                    "credit_risk_fs.selectors.llm_screening.LLMSelector"
                    if method == "llm"
                    else "credit_risk_fs.selectors.stable_core_llm_fill."
                    "StableCoreLLMFillSelector"
                ),
                "model": model,
                "requested_feature_budget": FROZEN_FEATURE_BUDGETS[model],
                "ranking_budget": FROZEN_LLM_RANKING_BUDGET,
                "seed": 42,
            }
        )
    return cells


def expanded_dev_evaluation_identities() -> list[dict[str, Any]]:
    return [
        {
            "evaluation_id": (
                f"p16v2-dev-fold-{fold_id}-c{cell['configuration_order']:03d}"
            ),
            "fold_id": fold_id,
            "configuration_order": cell["configuration_order"],
            "configuration_id": cell["configuration_id"],
            "method_id": cell["method_id"],
            "model": cell["model"],
            "requested_feature_budget": cell["requested_feature_budget"],
        }
        for fold_id in FROZEN_FOLDS
        for cell in supplemental_cells()
    ]


def expanded_oot_evaluation_identities() -> list[dict[str, Any]]:
    return [
        {
            "evaluation_id": f"p16v2-oot-c{cell['configuration_order']:03d}",
            "configuration_order": cell["configuration_order"],
            "configuration_id": cell["configuration_id"],
            "method_id": cell["method_id"],
            "model": cell["model"],
            "requested_feature_budget": cell["requested_feature_budget"],
        }
        for cell in supplemental_cells()
    ]


def corrected_accounting() -> dict[str, Any]:
    return {
        "target_free_llm_ranking_generations": 1,
        "target_free_llm_api_ranking_is_supervised_selector_fit": False,
        "global_llm_budget_truncation_states": 2,
        "added_registered_supervised_selector_fits": 10,
        "stable_core_outer_fits": 10,
        "stable_core_rf_mrmr_component_fits_per_outer_fit": 5,
        "added_internal_supervised_component_fits": 50,
        "preserved_classical_registered_selector_fits": 135,
        "amended_registered_supervised_selector_fits": 145,
        "added_dev_evaluations": 20,
        "preserved_classical_dev_evaluations": 150,
        "amended_dev_evaluations": 170,
        "added_full_dev_selector_refits": 2,
        "added_full_dev_internal_component_fits": 10,
        "preserved_classical_full_dev_selector_refits": 27,
        "amended_full_dev_selector_refits": 29,
        "added_oot_evaluations": 4,
        "preserved_classical_oot_evaluations": 30,
        "amended_oot_evaluations": 34,
        "unavailable_semantic_mixed_voter_execution_cells": 0,
    }


def load_supplemental_amendment(path: str | Path) -> tuple[dict[str, Any], str, str]:
    payload, file_digest, internal_digest = _load_self_authenticated_json(
        path,
        schema_version=AMENDMENT_SCHEMA_VERSION,
        status="prospective_authorized_before_supplemental_dev_and_oot",
    )
    if payload.get("parent_protocol", {}).get("file_sha256") != EXPECTED_V1_FILE_SHA256:
        raise Prompt16ExecutionError("v2 amendment parent protocol identity mismatch")
    if payload.get("blocker", {}).get("artifact_sha256") != EXPECTED_BLOCKER_SHA256:
        raise Prompt16ExecutionError("v2 amendment blocker identity mismatch")
    if payload.get("blocker", {}).get("commit") != EXPECTED_BLOCKER_COMMIT:
        raise Prompt16ExecutionError("v2 amendment blocker commit mismatch")
    methods = payload.get("method_registry", {}).get("added_methods")
    if not isinstance(methods, list) or [item.get("method_id") for item in methods] != list(
        FROZEN_METHODS
    ):
        raise Prompt16ExecutionError("v2 amendment method registry is not the exact two-method set")
    unavailable = payload.get("method_registry", {}).get("unavailable_method", {})
    if unavailable != {
        "role": "semantic_mixed_voter",
        "status": "unavailable_due_to_unresolved_historical_provenance",
        "execution_cells": 0,
    }:
        raise Prompt16ExecutionError("semantic/mixed voter limitation record changed")
    registry = payload.get("evaluation_registry", {})
    if registry.get("supplemental_cells") != supplemental_cells():
        raise Prompt16ExecutionError("supplemental configuration registry changed")
    if registry.get("dev_evaluation_identities") != expanded_dev_evaluation_identities():
        raise Prompt16ExecutionError("20-cell DEV identity registry changed")
    if registry.get("oot_evaluation_identities") != expanded_oot_evaluation_identities():
        raise Prompt16ExecutionError("four-cell OOT identity registry changed")
    if payload.get("accounting") != corrected_accounting():
        raise Prompt16ExecutionError("supplemental selector/evaluation accounting changed")
    return payload, file_digest, internal_digest


def load_supplemental_authorization(
    path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload, file_digest, internal_digest = _load_self_authenticated_json(
        path,
        schema_version=AUTHORIZATION_SCHEMA_VERSION,
        status="authorized_for_one_resumable_five_fold_supplemental_dev_command",
    )
    if payload.get("operation") != "supplemental_dev_only_no_oot_code_path":
        raise Prompt16ExecutionError("authorization operation boundary changed")
    amendment_path = PROJECT_ROOT / str(payload.get("amendment", {}).get("path", ""))
    amendment, amendment_file_sha, amendment_internal_sha = load_supplemental_amendment(
        amendment_path
    )
    expected_amendment = payload.get("amendment", {})
    if expected_amendment.get("file_sha256") != amendment_file_sha:
        raise Prompt16ExecutionError("authorization amendment file digest mismatch")
    if expected_amendment.get("internal_sha256") != amendment_internal_sha:
        raise Prompt16ExecutionError("authorization amendment internal digest mismatch")
    if payload.get("execution_accounting") != corrected_accounting():
        raise Prompt16ExecutionError("authorization accounting changed")
    if payload.get("supplemental_cells") != supplemental_cells():
        raise Prompt16ExecutionError("authorization cell registry changed")
    for item in payload.get("implementation_files", []):
        candidate = PROJECT_ROOT / str(item.get("path", ""))
        if not candidate.is_file() or file_sha256(candidate) != item.get("sha256"):
            raise Prompt16ExecutionError(
                f"authorized implementation file changed: {item.get('path')}"
            )
    payload["_authorization_file_sha256"] = file_digest
    payload["_authorization_internal_sha256"] = internal_digest
    payload["_amendment_payload"] = amendment
    return payload, amendment


def _normalize_definition(value: Any) -> str:
    return " ".join(str(value or "").split())


def render_target_free_feature_descriptions(
    *,
    predictors: Sequence[str],
    lineage_payload: Mapping[str, Any],
    metadata_payload: Mapping[str, Any],
    contract: AdapterContract,
) -> list[dict[str, Any]]:
    """Render one deterministic, outcome-independent line per predictor."""

    ordered = [str(value) for value in predictors]
    if len(ordered) != 1959 or len(ordered) != len(set(ordered)):
        raise Prompt16ExecutionError("description renderer requires 1,959 unique predictors")
    lineage = lineage_payload.get("features")
    if not isinstance(lineage, list) or [row.get("output_feature") for row in lineage] != ordered:
        raise Prompt16ExecutionError("lineage does not exactly cover the ordered predictor universe")
    dtype_by_name = {
        str(row["name"]): str(row["arrow_type"])
        for row in metadata_payload.get("columns", [])
        if isinstance(row, Mapping) and "name" in row and "arrow_type" in row
    }
    rules = {
        (table.family, rule.feature_name): rule
        for table in contract.tables
        for rule in table.feature_rules
    }
    table_depth = {table.family: table.depth for table in contract.tables}
    records: list[dict[str, Any]] = []
    for row in lineage:
        name = str(row["output_feature"])
        family = str(row["source_family"])
        original_value = row.get("source_feature")
        original = None if original_value is None else str(original_value)
        rule = None if original is None else rules.get((family, original))
        if original is not None and rule is None:
            raise Prompt16ExecutionError(f"approved definition is missing for {name}")
        if original is None:
            definition = f"Count of related rows in the {family} source table for the base case."
            original_rendered = "generated_row_count"
        else:
            definition = _normalize_definition(rule.description)
            if not definition:
                raise Prompt16ExecutionError(f"approved definition is blank for {name}")
            original_rendered = original
        fields = {
            "name": name,
            "source_family": family,
            "source_table": family,
            "original_feature": original_rendered,
            "depth": str(table_depth[family]),
            "aggregation": str(row["aggregation"]),
            "dtype": dtype_by_name.get(name, str(row["logical_type"])),
            "logical_type": str(row["logical_type"]),
            "approved_definition": definition,
        }
        rendered = "- " + json.dumps(
            fields,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        lower_rendered = rendered.lower()
        forbidden = [token for token in FORBIDDEN_PROMPT_FIELD_TOKENS if token in lower_rendered]
        if forbidden:
            raise Prompt16ExecutionError(
                f"rendered description contains a forbidden field token for {name}: {forbidden}"
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
    return records


def build_description_and_prompt_freeze(
    *,
    matrix_root: str | Path,
    protocol_lock: str | Path,
    metadata_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(matrix_root)
    metadata = dict(metadata_payload or _json(root / "metadata.json"))
    lineage = _json(root / "lineage.json")
    predictors = [str(value) for value in metadata.get("predictor_columns", [])]
    contract = load_adapter_contract(protocol_lock)
    records = render_target_free_feature_descriptions(
        predictors=predictors,
        lineage_payload=lineage,
        metadata_payload=metadata,
        contract=contract,
    )
    selector = LLMSelector(
        description_csv_path=str(root / "lineage.json"),
        cache_dir=str(root / "unused_target_free_cache"),
        model=FROZEN_LLM_MODEL,
        temperature=FROZEN_LLM_TEMPERATURE,
        max_features=FROZEN_LLM_RANKING_BUDGET,
        ranking_budget=FROZEN_LLM_RANKING_BUDGET,
        feature_budget=FROZEN_LLM_RANKING_BUDGET,
        shared_pool_size=FROZEN_LLM_RANKING_BUDGET,
        prompt_version="stability_expert_v4",
        iv_filter_kwargs={},
    )
    prompt = selector.build_target_free_prompt(records, expected_features=predictors)
    per_feature_hashes = [
        {"name": row["name"], "description_sha256": row["description_sha256"]}
        for row in records
    ]
    freeze = {
        "ordered_feature_count": len(predictors),
        "ordered_feature_universe_sha256": canonical_sha256(predictors),
        "lineage_file_sha256": file_sha256(root / "lineage.json"),
        "metadata_file_sha256": file_sha256(root / "metadata.json"),
        "description_contract": "prompt_16_target_free_adapter_lineage_v2",
        "description_count": len(records),
        "per_feature_description_hashes_sha256": canonical_sha256(per_feature_hashes),
        "ordered_rendered_descriptions_sha256": canonical_sha256(
            [row["rendered_description"] for row in records]
        ),
        "prompt_version": "stability_expert_v4",
        "prompt_chunk_count": 1,
        "prompt_chunk_order": [1],
        "prompt_merge_algorithm": "identity_single_chunk_no_cross_chunk_merge",
        "rendered_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "system_message_sha256": hashlib.sha256(
            LLMSelector.TARGET_FREE_SYSTEM_MESSAGE.encode("utf-8")
        ).hexdigest(),
        "retry_suffix_sha256": hashlib.sha256(
            LLMSelector.TARGET_FREE_RETRY_SUFFIX.encode("utf-8")
        ).hexdigest(),
        "provider": "openai",
        "endpoint": "chat.completions.create",
        "request_model": FROZEN_LLM_MODEL,
        "required_response_model": FROZEN_LLM_MODEL,
        "historical_request_alias": "gpt-4.1-mini",
        "historical_authenticated_response_model": FROZEN_LLM_MODEL,
        "temperature": FROZEN_LLM_TEMPERATURE,
        "seed": None,
        "seed_rule": "not_sent_by_authenticated_historical_chat_completions_call",
        "response_format": selector.target_free_response_format(),
        "application_parse_or_contract_attempts_maximum": 3,
        "sdk_transport_retry_rule": "OpenAI_Python_SDK_default_transport_retries",
        "parser": "strict_json_schema_then_exact_100_distinct_known_names",
        "fallback": "forbidden",
        "cache_reuse": "only_complete_recursively_hashed_identity_exact_success",
        "candidate_coverage": {
            "descriptions_required": 1959,
            "ranked_features_required": 100,
            "unknown_allowed": 0,
            "duplicates_allowed": 0,
            "missing_rank_positions_allowed": 0,
            "silent_fill_allowed": False,
        },
    }
    return {
        "predictors": predictors,
        "records": records,
        "prompt": prompt,
        "freeze": freeze,
    }


def classical_tree_identity(root: str | Path) -> dict[str, Any]:
    base = Path(root)
    if not base.is_dir():
        raise Prompt16ExecutionError("completed classical DEV root is missing")
    digest = hashlib.sha256()
    count = 0
    byte_count = 0
    for path in sorted(
        (item for item in base.rglob("*") if item.is_file()),
        key=lambda item: str(item.resolve()).lower(),
    ):
        relative = path.relative_to(base).as_posix()
        size = path.stat().st_size
        digest.update(
            f"{relative}\t{size}\t{file_sha256(path)}\n".encode("utf-8")
        )
        count += 1
        byte_count += size
    return {
        "tree_manifest_sha256": digest.hexdigest(),
        "file_count": count,
        "byte_count": byte_count,
        "serialization": (
            "relative POSIX path, TAB, byte length, TAB, lowercase SHA-256, LF; "
            "paths sorted by absolute path"
        ),
    }


def classical_evaluation_manifest_identity(root: str | Path) -> dict[str, Any]:
    """Hash all 150 classical evaluation manifests without opening metrics."""

    base = Path(root)
    all_rows: list[dict[str, str]] = []
    folds: dict[str, dict[str, Any]] = {}
    for fold_id in FROZEN_FOLDS:
        fold_rows = [
            {
                "path": path.relative_to(base).as_posix(),
                "sha256": file_sha256(path),
            }
            for path in sorted(
                (base / f"fold_{fold_id}" / "evaluations").glob(
                    "cell_*/manifest.json"
                ),
                key=lambda path: path.as_posix(),
            )
        ]
        folds[f"fold_{fold_id}"] = {
            "count": len(fold_rows),
            "manifest_registry_sha256": canonical_sha256(fold_rows),
        }
        all_rows.extend(fold_rows)
    return {
        "evaluation_manifest_count": len(all_rows),
        "evaluation_manifest_registry_sha256": canonical_sha256(all_rows),
        "fold_evaluation_manifest_registries": folds,
    }


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def active_prompt16_workers() -> list[dict[str, Any]]:
    """Return other live Prompt-16 Python workers using command-line identity."""

    import psutil

    current_pid = os.getpid()
    current_lineage_pids = {current_pid}
    try:
        current_process = psutil.Process(current_pid)
        while True:
            parent = current_process.parent()
            if parent is None:
                break
            parent_pid = int(parent.pid)
            if parent_pid in current_lineage_pids:
                break
            current_lineage_pids.add(parent_pid)
            current_process = parent
    except (psutil.Error, TypeError, ValueError):
        # The current PID remains excluded even if an ancestor exits while the
        # process table is being inspected.
        pass
    workers: list[dict[str, Any]] = []
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            pid = int(process.info["pid"])
            name = str(process.info.get("name") or "").lower()
            command = " ".join(process.info.get("cmdline") or [])
        except (psutil.Error, TypeError, ValueError):
            continue
        if pid in current_lineage_pids or not name.startswith("python"):
            continue
        normalized = command.lower().replace("\\", "/")
        if (
            "run_prompt_16_third_dataset.py" in normalized
            or "prompt_16_third_dataset:run_" in normalized
            or "prompt_16_llm_supplement:run_" in normalized
        ):
            workers.append({"pid": pid, "name": name, "command_line": command})
    return workers


def prompt16_execution_locks(plan: Mapping[str, Any]) -> list[str]:
    roots = {
        Path(plan["paths"]["matrix_root"]).parent,
        Path(plan["paths"]["pilot_root"]).parent,
        Path(plan["paths"]["dev_root"]).parent,
    }
    return sorted(
        str(path)
        for root in roots
        if root.exists()
        for path in root.rglob(".*.execution.lock")
        if path.is_file()
    )


def _authenticate_repository(authorization: Mapping[str, Any], authorization_path: Path) -> dict[str, Any]:
    if _git("branch", "--show-current") != "main":
        raise Prompt16ExecutionError("supplemental DEV requires branch main")
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise Prompt16ExecutionError("supplemental DEV requires a clean worktree")
    head = _git("rev-parse", "HEAD")
    implementation_commit = str(authorization.get("implementation_commit", ""))
    if _git("merge-base", "--is-ancestor", implementation_commit, head) != "":
        # ``git merge-base --is-ancestor`` succeeds with empty stdout.  The call
        # above already raises on a non-ancestor exit code.
        raise Prompt16ExecutionError("unexpected git ancestor command output")
    tracked = _git("ls-files", "--error-unmatch", authorization_path.relative_to(PROJECT_ROOT).as_posix())
    if tracked != authorization_path.relative_to(PROJECT_ROOT).as_posix():
        raise Prompt16ExecutionError("execution authorization is not tracked")
    return {
        "branch": "main",
        "head": head,
        "implementation_commit": implementation_commit,
        "implementation_commit_is_ancestor": True,
        "worktree_clean": True,
        "authorization_tracked": True,
    }


def _assert_no_oot_state(plan: Mapping[str, Any]) -> dict[str, Any]:
    paths = plan["paths"]
    oot_root = Path(paths["oot_root"])
    if oot_root.exists():
        raise Prompt16ExecutionError(
            f"third-dataset OOT contamination: frozen OOT root exists: {oot_root}"
        )
    result_root = oot_root.parent
    output_root = Path(paths["matrix_root"]).parent
    named: list[str] = []
    for base in (result_root, output_root):
        if not base.exists():
            continue
        for item in base.rglob("*"):
            if not item.exists():
                continue
            parts = [part.lower() for part in item.relative_to(base).parts]
            if any(
                part == "oot"
                or part.startswith("oot_")
                or part.startswith("oot-")
                for part in parts
            ):
                named.append(str(item))
    log_root = Path(paths["log_root"])
    if log_root.exists():
        named.extend(str(path) for path in log_root.glob("oot*"))
    if named:
        raise Prompt16ExecutionError(
            f"third-dataset OOT contamination paths exist: {sorted(set(named))[:20]}"
        )
    return {
        "oot_root": str(oot_root),
        "oot_root_exists": False,
        "oot_named_paths": [],
        "old_classical_only_oot_command": "revoked",
        "supplemental_controller_oot_capability": False,
    }


def authenticate_supplemental_entry(
    authorization_path: str | Path,
    *,
    require_repository_state: bool = True,
) -> dict[str, Any]:
    auth_path = Path(authorization_path).resolve()
    authorization, amendment = load_supplemental_authorization(auth_path)
    repository = (
        _authenticate_repository(authorization, auth_path)
        if require_repository_state
        else {"repository_check_skipped_for_synthetic_test": True}
    )
    plan_path = PROJECT_ROOT / str(authorization["execution_plan_path"])
    plan = load_execution_plan(plan_path)
    v1_path = PROJECT_ROOT / str(authorization["v1_protocol_path"])
    contract = load_adapter_contract(v1_path)
    if contract.lock_file_sha256 != EXPECTED_V1_FILE_SHA256:
        raise Prompt16ExecutionError("v1 protocol file changed")
    if contract.lock_internal_sha256 != EXPECTED_V1_INTERNAL_SHA256:
        raise Prompt16ExecutionError("v1 protocol internal identity changed")
    blocker_path = PROJECT_ROOT / str(authorization["blocker_path"])
    if file_sha256(blocker_path) != EXPECTED_BLOCKER_SHA256:
        raise Prompt16ExecutionError("preserved blocker artifact changed")
    v1_payload = _json(v1_path)
    v1_matrix = v1_payload["approved_protocol"]["method_and_evaluation_matrix"]
    amended_settings = amendment.get("frozen_classical_execution_settings", {})
    for key in ("seeds", "resource_controls", "final_model_settings"):
        if amended_settings.get(key) != v1_matrix.get(key):
            raise Prompt16ExecutionError(
                f"v2 amendment changed frozen classical execution setting: {key}"
            )
    frozen_split = amendment.get("frozen_data_and_fold_identities", {})
    v1_split = v1_payload["approved_protocol"]["split_and_fold_boundaries"]
    if frozen_split.get("fold_registry_sha256") != canonical_sha256(v1_split["folds"]):
        raise Prompt16ExecutionError("v2 amendment frozen fold registry changed")
    if frozen_split.get("full_dev", {}).get("ordered_case_id_sha256") != v1_split[
        "dev"
    ]["ordered_case_id_sha256"]:
        raise Prompt16ExecutionError("v2 amendment full-DEV identity changed")
    if frozen_split.get("oot", {}).get("ordered_case_id_sha256") != v1_split[
        "oot"
    ]["ordered_case_id_sha256"]:
        raise Prompt16ExecutionError("v2 amendment frozen OOT membership identity changed")

    matrix_root = Path(plan["paths"]["matrix_root"])
    matrix_manifest = validate_output_manifest(matrix_root)
    matrix_manifest_sha = file_sha256(matrix_root / "manifest.json")
    if matrix_manifest_sha != EXPECTED_MATRIX_MANIFEST_SHA256:
        raise Prompt16ExecutionError("completed matrix manifest changed")
    metadata = _json(matrix_root / "metadata.json")
    predictors = list(metadata.get("predictor_columns", []))
    if canonical_sha256(predictors) != EXPECTED_UNIVERSE_SHA256:
        raise Prompt16ExecutionError("ordered 1,959-feature universe changed")
    description_freeze = build_description_and_prompt_freeze(
        matrix_root=matrix_root,
        protocol_lock=v1_path,
        metadata_payload=metadata,
    )
    if description_freeze["freeze"] != amendment.get("llm_provenance_freeze"):
        raise Prompt16ExecutionError("rendered descriptions or prompt differ from v2 freeze")

    classical_root = Path(plan["paths"]["dev_root"])
    classical = classical_tree_identity(classical_root)
    if classical != {
        "tree_manifest_sha256": EXPECTED_CLASSICAL_TREE_SHA256,
        "file_count": EXPECTED_CLASSICAL_FILE_COUNT,
        "byte_count": EXPECTED_CLASSICAL_BYTE_COUNT,
        "serialization": classical["serialization"],
    }:
        raise Prompt16ExecutionError(f"completed classical DEV tree changed: {classical}")
    classical_evaluations = classical_evaluation_manifest_identity(classical_root)
    declared_classical = amendment.get("classical_preservation", {})
    for key in (
        "evaluation_manifest_count",
        "evaluation_manifest_registry_sha256",
        "fold_evaluation_manifest_registries",
    ):
        if classical_evaluations[key] != declared_classical.get(key):
            raise Prompt16ExecutionError(
                f"completed classical evaluation preservation identity changed: {key}"
            )
    oot = _assert_no_oot_state(plan)
    return {
        "authorization": authorization,
        "amendment": amendment,
        "plan": plan,
        "contract": contract,
        "repository": repository,
        "matrix_manifest": matrix_manifest,
        "matrix_manifest_sha256": matrix_manifest_sha,
        "matrix_metadata": metadata,
        "description_freeze": description_freeze,
        "classical_tree": classical,
        "classical_evaluation_manifests": classical_evaluations,
        "oot_state": oot,
        "authenticated_at_utc": _utc_now(),
    }


def _artifact_digest(path: Path, base: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(base).as_posix(),
        "byte_size": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _seal_recursive_directory(path: Path, identity: Mapping[str, Any]) -> dict[str, Any]:
    artifacts = [
        _artifact_digest(item, path)
        for item in sorted(
            (candidate for candidate in path.rglob("*") if candidate.is_file()),
            key=lambda candidate: candidate.relative_to(path).as_posix(),
        )
        if item.relative_to(path).as_posix() not in {"manifest.json", "_SUCCESS"}
        and not item.name.endswith(".partial")
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "identity": dict(identity),
        "artifacts": artifacts,
        "completed_at_utc": _utc_now(),
    }
    write_json_atomic(path / "manifest.json", manifest, overwrite=False)
    manifest_sha = file_sha256(path / "manifest.json")
    write_text_atomic(
        path / "_SUCCESS",
        json.dumps({"manifest_sha256": manifest_sha}, sort_keys=True) + "\n",
        overwrite=False,
    )
    return {**manifest, "manifest_sha256": manifest_sha}


def _load_recursive_sealed(
    path: Path, identity: Mapping[str, Any]
) -> dict[str, Any] | None:
    if not (path / "_SUCCESS").is_file():
        return None
    success = _json(path / "_SUCCESS")
    manifest_path = path / "manifest.json"
    if success.get("manifest_sha256") != file_sha256(manifest_path):
        raise Prompt16ExecutionError(f"completion marker mismatch: {path}")
    manifest = _json(manifest_path)
    if manifest.get("identity") != dict(identity):
        raise Prompt16ExecutionError(f"completed supplemental identity mismatch: {path}")
    declared: set[str] = set()
    for item in manifest.get("artifacts", []):
        relative = str(item["path"])
        declared.add(relative)
        artifact = path / relative
        if not artifact.is_file() or artifact.stat().st_size != int(item["byte_size"]):
            raise Prompt16ExecutionError(f"supplemental artifact size mismatch: {artifact}")
        if file_sha256(artifact) != item["sha256"]:
            raise Prompt16ExecutionError(f"supplemental artifact digest mismatch: {artifact}")
    observed = {
        candidate.relative_to(path).as_posix()
        for candidate in path.rglob("*")
        if candidate.is_file()
        and candidate.relative_to(path).as_posix() not in {"manifest.json", "_SUCCESS"}
        and not candidate.name.endswith(".partial")
    }
    if observed != declared:
        raise Prompt16ExecutionError(f"supplemental sealed inventory mismatch: {path}")
    return manifest


def _ranking_identity(entry: Mapping[str, Any]) -> dict[str, Any]:
    authorization = entry["authorization"]
    freeze = entry["description_freeze"]["freeze"]
    return {
        "operation": "target_free_shared_llm_ranking",
        "authorization_file_sha256": authorization["_authorization_file_sha256"],
        "amendment_internal_sha256": authorization["amendment"]["internal_sha256"],
        "matrix_manifest_sha256": entry["matrix_manifest_sha256"],
        "ordered_feature_universe_sha256": freeze["ordered_feature_universe_sha256"],
        "ordered_rendered_descriptions_sha256": freeze[
            "ordered_rendered_descriptions_sha256"
        ],
        "rendered_prompt_sha256": freeze["rendered_prompt_sha256"],
        "provider": freeze["provider"],
        "request_model": freeze["request_model"],
        "required_response_model": freeze["required_response_model"],
        "temperature": freeze["temperature"],
        "seed": freeze["seed"],
        "ranking_budget": FROZEN_LLM_RANKING_BUDGET,
        "scope": "outcome_independent_shared_across_all_dev_folds_and_later_full_dev",
    }


def ensure_target_free_ranking(
    *,
    entry: Mapping[str, Any],
    output_root: str | Path,
    stop_event: Any = None,
    stage_queue: Any = None,
    selector_factory: Callable[[], LLMSelector] | None = None,
) -> tuple[list[str], str, dict[str, Any]]:
    root = Path(output_root) / "llm_ranking"
    identity = _ranking_identity(entry)
    sealed = _load_recursive_sealed(root, identity)
    if sealed is not None:
        payload = _json(root / "ranking_payload.json")
        ranking = list(payload.get("selected_features", []))
        if (
            len(ranking) != FROZEN_LLM_RANKING_BUDGET
            or len(ranking) != len(set(ranking))
            or not set(ranking).issubset(
                set(entry["description_freeze"]["predictors"])
            )
            or payload.get("fallback_used") is not False
        ):
            raise Prompt16ExecutionError("sealed LLM ranking violates strict coverage")
        return ranking, file_sha256(root / "manifest.json"), payload

    _archive_incomplete(root, Path(output_root) / "archived_incomplete_attempts")
    root.mkdir(parents=True, exist_ok=False)
    _check_stop(stop_event)
    _publish_stage(
        stage_queue,
        "baseline_lightweight",
        "supplemental_dev:ranking:llm",
        component="outcome_independent_llm_ranking",
        method_id="llm",
        operation="external_api_ranking",
    )
    freeze_bundle = entry["description_freeze"]
    freeze = freeze_bundle["freeze"]
    write_json_atomic(root / "provenance_freeze.json", freeze, overwrite=False)
    write_json_atomic(
        root / "feature_descriptions.json",
        freeze_bundle["records"],
        overwrite=False,
    )
    write_csv_atomic(
        root / "feature_description_hashes.csv",
        pd.DataFrame(
            [
                {
                    "order": index,
                    "feature": row["name"],
                    "description_sha256": row["description_sha256"],
                }
                for index, row in enumerate(freeze_bundle["records"], start=1)
            ]
        ),
        required_columns=("order", "feature", "description_sha256"),
        ordered_row_identity_column="feature",
        overwrite=False,
    )
    write_text_atomic(root / "prompt.txt", freeze_bundle["prompt"], overwrite=False)

    selector = (
        selector_factory()
        if selector_factory is not None
        else LLMSelector(
            description_csv_path=str(Path(entry["plan"]["paths"]["matrix_root"]) / "lineage.json"),
            cache_dir=str(root),
            model=FROZEN_LLM_MODEL,
            temperature=FROZEN_LLM_TEMPERATURE,
            max_features=FROZEN_LLM_RANKING_BUDGET,
            ranking_budget=FROZEN_LLM_RANKING_BUDGET,
            feature_budget=FROZEN_LLM_RANKING_BUDGET,
            shared_pool_size=FROZEN_LLM_RANKING_BUDGET,
            prompt_version="stability_expert_v4",
            iv_filter_kwargs={},
        )
    )

    def record_attempt(record: Mapping[str, Any]) -> None:
        attempt = int(record["attempt"])
        attempt_root = root / "attempts" / f"attempt_{attempt:03d}"
        attempt_root.mkdir(parents=True, exist_ok=False)
        write_json_atomic(attempt_root / "request.json", record["request"], overwrite=False)
        write_json_atomic(attempt_root / "response.json", record["response"], overwrite=False)
        write_json_atomic(
            attempt_root / "status.json",
            {
                "attempt": attempt,
                "request_sha256": record["request_sha256"],
                "valid": bool(record["valid"]),
                "validation_error": record["validation_error"],
                "recorded_at_utc": _utc_now(),
            },
            overwrite=False,
        )

    started = time.perf_counter()
    try:
        payload = selector.rank_target_free(
            freeze_bundle["records"],
            expected_features=freeze_bundle["predictors"],
            expected_response_model=FROZEN_LLM_MODEL,
            attempt_recorder=record_attempt,
            maximum_attempts=3,
        )
    except Exception as exc:
        write_json_atomic(
            root / "failure.json",
            {
                "status": "failed_unsealed_safe_to_resume",
                "error": {"class": type(exc).__name__, "message": str(exc)},
                "elapsed_seconds": time.perf_counter() - started,
                "fallback_used": False,
            },
            overwrite=False,
        )
        raise
    payload.update(
        {
            "status": "complete",
            "method_id": "llm",
            "implementation": "credit_risk_fs.selectors.llm_screening.LLMSelector",
            "ranking_scope": "outcome_independent_shared_across_folds",
            "feature_universe_sha256": freeze["ordered_feature_universe_sha256"],
            "description_registry_sha256": freeze[
                "ordered_rendered_descriptions_sha256"
            ],
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    write_json_atomic(root / "ranking_payload.json", payload, overwrite=False)
    write_csv_atomic(
        root / "ranking.csv",
        pd.DataFrame(
            {
                "rank": range(1, FROZEN_LLM_RANKING_BUDGET + 1),
                "feature": payload["selected_features"],
            }
        ),
        required_columns=("rank", "feature"),
        ordered_row_identity_column="feature",
        overwrite=False,
    )
    write_json_atomic(
        root / "status.json",
        {
            "status": "complete",
            "ranked_features": FROZEN_LLM_RANKING_BUDGET,
            "fallback_used": False,
            "application_attempt": payload["application_attempt"],
        },
        overwrite=False,
    )
    sealed = _seal_recursive_directory(root, identity)
    return list(payload["selected_features"]), sealed["manifest_sha256"], payload


def _selection_identity(
    *,
    entry: Mapping[str, Any],
    ranking_manifest_sha256: str,
    method_id: str,
    model: str,
    fold_id: int | None,
) -> dict[str, Any]:
    return {
        "operation": (
            "global_target_free_budget_truncation"
            if method_id == "llm"
            else "fold_local_stable_core_supervised_fit"
        ),
        "authorization_file_sha256": entry["authorization"][
            "_authorization_file_sha256"
        ],
        "amendment_internal_sha256": entry["authorization"]["amendment"][
            "internal_sha256"
        ],
        "matrix_manifest_sha256": entry["matrix_manifest_sha256"],
        "ranking_manifest_sha256": ranking_manifest_sha256,
        "method_id": method_id,
        "model": model,
        "fold_id": fold_id,
        "feature_budget": FROZEN_FEATURE_BUDGETS[model],
        "seed": 42,
    }


def ensure_global_llm_selection_states(
    *,
    entry: Mapping[str, Any],
    output_root: str | Path,
    ranking: Sequence[str],
    ranking_manifest_sha256: str,
) -> dict[str, tuple[Path, str]]:
    outputs: dict[str, tuple[Path, str]] = {}
    for model in FROZEN_MODELS:
        path = Path(output_root) / "shared_selection_states" / f"llm_{model}"
        identity = _selection_identity(
            entry=entry,
            ranking_manifest_sha256=ranking_manifest_sha256,
            method_id="llm",
            model=model,
            fold_id=None,
        )
        sealed = _load_recursive_sealed(path, identity)
        if sealed is None:
            _archive_incomplete(path, Path(output_root) / "archived_incomplete_attempts")
            path.mkdir(parents=True, exist_ok=False)
            selected = list(ranking[: FROZEN_FEATURE_BUDGETS[model]])
            write_json_atomic(
                path / "selection.json",
                {
                    "status": "complete",
                    "method_id": "llm",
                    "implementation": "credit_risk_fs.selectors.llm_screening.LLMSelector",
                    "operation": "deterministic_truncation_of_authenticated_target_free_ranking",
                    "supervised_selector_fit": False,
                    "model": model,
                    "requested_feature_budget": FROZEN_FEATURE_BUDGETS[model],
                    "realized_support": len(selected),
                    "selected_features": selected,
                    "natural_support_unpadded": True,
                    "ranking_manifest_sha256": ranking_manifest_sha256,
                },
                overwrite=False,
            )
            write_csv_atomic(
                path / "selected_features.csv",
                pd.DataFrame(
                    {"rank": range(1, len(selected) + 1), "feature": selected}
                ),
                required_columns=("rank", "feature"),
                ordered_row_identity_column="feature",
                overwrite=False,
            )
            sealed = _seal_recursive_directory(path, identity)
        outputs[model] = (path, file_sha256(path / "manifest.json"))
    return outputs


def _load_fold_frames(
    *,
    entry: Mapping[str, Any],
    fold_id: int,
    stop_event: Any,
    stage_queue: Any,
    ram_ready_event: Any,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, Any]]:
    root = Path(entry["plan"]["paths"]["matrix_root"])
    manifest = entry["matrix_manifest"]
    metadata = entry["matrix_metadata"]
    predictors = list(metadata["predictor_columns"])
    protocol = _json(entry["plan"]["paths"]["protocol_lock"])
    train_expected, validation_expected = _expected_scope(protocol, "dev", fold_id)
    label = f"fold_{fold_id}"
    train = _read_date_slice(
        root,
        manifest,
        date_min=str(train_expected["date_min"]),
        date_max=str(train_expected["date_max"]),
        predictors=predictors,
        stop_event=stop_event,
        stage_queue=stage_queue,
        stage="dev_data_loading",
        fold_label=label + ":train",
        ram_ready_event=ram_ready_event,
    )
    validation = _read_date_slice(
        root,
        manifest,
        date_min=str(validation_expected["date_min"]),
        date_max=str(validation_expected["date_max"]),
        predictors=predictors,
        stop_event=stop_event,
        stage_queue=stage_queue,
        stage="dev_data_loading",
        fold_label=label + ":validation",
        ram_ready_event=ram_ready_event,
    )
    authentication = {
        "matrix_manifest_sha256": entry["matrix_manifest_sha256"],
        "train": _validate_scope_frame(train, train_expected, label + ":train"),
        "validation": _validate_scope_frame(
            validation, validation_expected, label + ":validation"
        ),
        "case_id_overlap": int(
            len(set(train["case_id"].tolist()) & set(validation["case_id"].tolist()))
        ),
    }
    if authentication["case_id_overlap"] != 0:
        raise Prompt16ExecutionError("training and held-out case IDs overlap")
    return train, validation, predictors, authentication


def _stable_selection_path(output_root: Path, fold_id: int, model: str) -> Path:
    return (
        output_root
        / f"fold_{fold_id}"
        / "selection_fits"
        / f"stable_core_llm_fill_{model}"
    )


def _fit_stable_core_state(
    *,
    entry: Mapping[str, Any],
    output_root: Path,
    fold_id: int,
    model: str,
    numeric_train: pd.DataFrame,
    y_train: pd.Series,
    scope_auth: Mapping[str, Any],
    ranking: Sequence[str],
    ranking_manifest_sha256: str,
    stop_event: Any,
    stage_queue: Any,
) -> tuple[Path, str]:
    path = _stable_selection_path(output_root, fold_id, model)
    identity = _selection_identity(
        entry=entry,
        ranking_manifest_sha256=ranking_manifest_sha256,
        method_id="stable_core_llm_fill",
        model=model,
        fold_id=fold_id,
    )
    sealed = _load_recursive_sealed(path, identity)
    if sealed is not None:
        return path, file_sha256(path / "manifest.json")
    _archive_incomplete(path, output_root / "archived_incomplete_attempts")
    path.mkdir(parents=True, exist_ok=False)
    _check_stop(stop_event)
    _publish_stage(
        stage_queue,
        "statistical_normalized_average_rank",
        f"supplemental_dev:fold_{fold_id}:stable_core_llm_fill_{model}",
        component="stable_core_llm_fill_selection",
        method_id="stable_core_llm_fill",
        model=model,
        operation="selector_fit",
    )
    started = time.perf_counter()
    selector = StableCoreLLMFillSelector(
        description_csv_path=str(
            Path(entry["plan"]["paths"]["matrix_root"]) / "lineage.json"
        ),
        cache_dir=str(output_root / "llm_ranking"),
        llm_model=FROZEN_LLM_MODEL,
        llm_temperature=FROZEN_LLM_TEMPERATURE,
        llm_max_features=FROZEN_LLM_RANKING_BUDGET,
        llm_shared_ranking_enabled=True,
        llm_config_hash=entry["description_freeze"]["freeze"][
            "rendered_prompt_sha256"
        ],
        llm_prompt_version="stability_expert_v3",
        llm_shared_pool_size=FROZEN_LLM_RANKING_BUDGET,
        final_feature_budget=FROZEN_FEATURE_BUDGETS[model],
        bootstrap_iterations=5,
        bootstrap_fraction=0.8,
        stability_threshold=0.8,
        random_state=42,
        iv_filter_kwargs={},
        allow_unranked_padding=False,
    )
    try:
        selector.fit_with_authenticated_ranking(
            numeric_train,
            y_train,
            ranked_features=list(ranking),
            ranking_manifest_sha256=ranking_manifest_sha256,
        )
        selected = list(selector.selected_features_ or [])
        if len(selected) != FROZEN_FEATURE_BUDGETS[model]:
            raise Prompt16ExecutionError(
                "stable-core hybrid did not realize its exact natural support"
            )
        if len(selected) != len(set(selected)):
            raise Prompt16ExecutionError("stable-core hybrid returned duplicate features")
        if not set(selected).issubset(set(numeric_train.columns)):
            raise Prompt16ExecutionError("stable-core hybrid escaped training universe")
        evidence = {
            "status": "complete",
            "method_id": "stable_core_llm_fill",
            "implementation": (
                "credit_risk_fs.selectors.stable_core_llm_fill."
                "StableCoreLLMFillSelector"
            ),
            "model": model,
            "fold_id": fold_id,
            "fit_scope": "dev_fold_training_only",
            "fold_train_alignment": scope_auth["train"]["observed"],
            "validation_or_oot_used_for_fit": False,
            "requested_feature_budget": FROZEN_FEATURE_BUDGETS[model],
            "realized_support": len(selected),
            "natural_support_unpadded": True,
            "selected_features": selected,
            "stable_core_features": list(selector.stable_core_features_ or []),
            "ranking_manifest_sha256": ranking_manifest_sha256,
            "bootstrap": {
                "iterations": 5,
                "fraction": 0.8,
                "stability_threshold": 0.8,
                "base_seed": 42,
                "component_implementation": (
                    "credit_risk_fs.selectors.mrmr."
                    "RandomForestRelevanceMRMRSelector"
                ),
                "component_method": "mrmr",
                "component_fit_count": 5,
                "trace": selector.bootstrap_trace_,
            },
            "fit_seconds": time.perf_counter() - started,
        }
        write_json_atomic(path / "selection.json", evidence, overwrite=False)
        write_csv_atomic(
            path / "selected_features.csv",
            pd.DataFrame(
                {"rank": range(1, len(selected) + 1), "feature": selected}
            ),
            required_columns=("rank", "feature"),
            ordered_row_identity_column="feature",
            overwrite=False,
        )
        frequency = selector.stable_core_frequency_
        if frequency is None:
            frequency = pd.DataFrame(
                columns=[
                    "feature_name",
                    "selection_count",
                    "mean_rank",
                    "selection_frequency",
                ]
            )
        write_csv_atomic(
            path / "stable_core_frequency.csv",
            frequency,
            required_columns=(
                "feature_name",
                "selection_count",
                "mean_rank",
                "selection_frequency",
            ),
            overwrite=False,
        )
    except Exception as exc:
        write_json_atomic(
            path / "failure.json",
            {
                "status": "failed_unsealed_safe_to_resume",
                "method_id": "stable_core_llm_fill",
                "model": model,
                "fold_id": fold_id,
                "fit_scope": "dev_fold_training_only",
                "fold_train_alignment": scope_auth["train"]["observed"],
                "error": {"class": type(exc).__name__, "message": str(exc)},
                "fit_seconds": time.perf_counter() - started,
            },
            overwrite=False,
        )
        raise
    finally:
        del selector
        gc.collect()
    sealed = _seal_recursive_directory(path, identity)
    return path, sealed["manifest_sha256"]


def _evaluation_identity(
    *,
    entry: Mapping[str, Any],
    fold_id: int,
    cell: Mapping[str, Any],
    selection_manifest_sha256: str,
) -> dict[str, Any]:
    return {
        "operation": "supplemental_dev_evaluation",
        "authorization_file_sha256": entry["authorization"][
            "_authorization_file_sha256"
        ],
        "amendment_internal_sha256": entry["authorization"]["amendment"][
            "internal_sha256"
        ],
        "matrix_manifest_sha256": entry["matrix_manifest_sha256"],
        "phase": "dev",
        "fold_id": fold_id,
        "configuration_order": int(cell["configuration_order"]),
        "cell_sha256": canonical_sha256(cell),
        "selection_manifest_sha256": selection_manifest_sha256,
    }


def _validate_completed_fold(
    *,
    fold_root: Path,
    entry: Mapping[str, Any],
    ranking_manifest_sha256: str,
    global_states: Mapping[str, tuple[Path, str]],
) -> dict[str, Any] | None:
    success_path = fold_root / "_SUCCESS"
    if not success_path.is_file():
        return None
    success = _json(success_path)
    manifest_path = fold_root / "fold_manifest.json"
    if success.get("fold_manifest_sha256") != file_sha256(manifest_path):
        raise Prompt16ExecutionError(f"supplemental fold completion marker mismatch: {fold_root}")
    manifest = _json(manifest_path)
    fold_id = int(manifest.get("fold_id", -1))
    if fold_id not in FROZEN_FOLDS:
        raise Prompt16ExecutionError("supplemental fold manifest has invalid fold id")
    for model in FROZEN_MODELS:
        stable_path = _stable_selection_path(fold_root.parent, fold_id, model)
        identity = _selection_identity(
            entry=entry,
            ranking_manifest_sha256=ranking_manifest_sha256,
            method_id="stable_core_llm_fill",
            model=model,
            fold_id=fold_id,
        )
        if _load_recursive_sealed(stable_path, identity) is None:
            raise Prompt16ExecutionError("supplemental fold stable-core fit is not sealed")
    for cell in supplemental_cells():
        selection_path, selection_sha = (
            global_states[str(cell["model"])]
            if cell["method_id"] == "llm"
            else (
                _stable_selection_path(fold_root.parent, fold_id, str(cell["model"])),
                file_sha256(
                    _stable_selection_path(
                        fold_root.parent, fold_id, str(cell["model"])
                    )
                    / "manifest.json"
                ),
            )
        )
        del selection_path
        evaluation_path = fold_root / "evaluations" / f"cell_{cell['configuration_order']:03d}"
        identity = _evaluation_identity(
            entry=entry,
            fold_id=fold_id,
            cell=cell,
            selection_manifest_sha256=selection_sha,
        )
        if _load_recursive_sealed(evaluation_path, identity) is None:
            raise Prompt16ExecutionError("supplemental fold evaluation is not sealed")
        status = _json(evaluation_path / "status.json")
        if status.get("status") != "complete":
            raise Prompt16ExecutionError("supplemental sealed evaluation is not complete")
    if manifest.get("completed_evaluations") != 4:
        raise Prompt16ExecutionError("supplemental fold does not account for four evaluations")
    return manifest


def run_supplemental_fold(
    *,
    entry: Mapping[str, Any],
    output_root: str | Path,
    fold_id: int,
    ranking: Sequence[str],
    ranking_manifest_sha256: str,
    global_states: Mapping[str, tuple[Path, str]],
    stop_event: Any = None,
    stage_queue: Any = None,
    ram_ready_event: Any = None,
) -> dict[str, Any]:
    if fold_id not in FROZEN_FOLDS:
        raise Prompt16ExecutionError("supplemental DEV fold id must be in 1..5")
    root = Path(output_root)
    fold_root = root / f"fold_{fold_id}"
    completed = _validate_completed_fold(
        fold_root=fold_root,
        entry=entry,
        ranking_manifest_sha256=ranking_manifest_sha256,
        global_states=global_states,
    )
    if completed is not None:
        return completed
    fold_root.mkdir(parents=True, exist_ok=True)
    archive_root = root / "archived_incomplete_attempts"
    train, validation, predictors, scope_auth = _load_fold_frames(
        entry=entry,
        fold_id=fold_id,
        stop_event=stop_event,
        stage_queue=stage_queue,
        ram_ready_event=ram_ready_event,
    )
    incomplete_stable: list[str] = []
    for model in FROZEN_MODELS:
        path = _stable_selection_path(root, fold_id, model)
        identity = _selection_identity(
            entry=entry,
            ranking_manifest_sha256=ranking_manifest_sha256,
            method_id="stable_core_llm_fill",
            model=model,
            fold_id=fold_id,
        )
        if _load_recursive_sealed(path, identity) is None:
            _archive_incomplete(path, archive_root)
            incomplete_stable.append(model)

    numeric_train: pd.DataFrame | None = None
    selection_target: pd.Series | None = None
    if incomplete_stable:
        _check_stop(stop_event)
        _publish_stage(
            stage_queue,
            "selection_encoding",
            f"supplemental_dev:fold_{fold_id}:selection_encoding",
            operation="selection_encoding",
        )
        started = time.perf_counter()
        selection_target = pd.Series(
            train["target"].to_numpy(dtype=np.int64, copy=True),
            index=train.index.copy(deep=True),
            name="target",
        )
        del validation
        for name in NON_PREDICTORS:
            if name in train:
                del train[name]
        if list(train.columns) != predictors:
            raise Prompt16ExecutionError("supplemental selector source order changed")
        gc.collect()
        encoder = OriginalFeatureNumericEncoder()
        encoder.fit(train)
        numeric_train = encoder.transform_releasing_source(train)
        if train.shape[1] != 0 or list(numeric_train.columns) != predictors:
            raise Prompt16ExecutionError("supplemental selector encoding changed universe")
        write_json_atomic(
            fold_root / "selector_encoding.json",
            {
                "implementation": (
                    "credit_risk_fs.preprocessing.encoding."
                    "OriginalFeatureNumericEncoder"
                ),
                "fit_scope": "dev_fold_training_only",
                "fold_id": fold_id,
                "training_rows": len(selection_target),
                "candidate_count": len(predictors),
                "numeric_column_count": len(encoder.numeric_columns_),
                "categorical_column_count": len(encoder.categorical_columns_),
                "training_only": True,
                "validation_released_before_supervised_fit": True,
                "shared_between_two_model_budget_hybrid_fits": True,
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        del encoder, train
        gc.collect()
        for model in incomplete_stable:
            _fit_stable_core_state(
                entry=entry,
                output_root=root,
                fold_id=fold_id,
                model=model,
                numeric_train=numeric_train,
                y_train=selection_target,
                scope_auth=scope_auth,
                ranking=ranking,
                ranking_manifest_sha256=ranking_manifest_sha256,
                stop_event=stop_event,
                stage_queue=stage_queue,
            )
        del numeric_train, selection_target
        numeric_train = None
        selection_target = None
        gc.collect()
        train, validation, reloaded_predictors, reloaded_scope = _load_fold_frames(
            entry=entry,
            fold_id=fold_id,
            stop_event=stop_event,
            stage_queue=stage_queue,
            ram_ready_event=ram_ready_event,
        )
        if reloaded_predictors != predictors or reloaded_scope != scope_auth:
            raise Prompt16ExecutionError("supplemental evaluation reload changed fold scope")

    completed_evaluations = 0
    for cell in supplemental_cells():
        _check_stop(stop_event)
        model = str(cell["model"])
        if cell["method_id"] == "llm":
            selection_path, selection_manifest_sha = global_states[model]
        else:
            selection_path = _stable_selection_path(root, fold_id, model)
            selection_manifest_sha = file_sha256(selection_path / "manifest.json")
        selection = _json(selection_path / "selection.json")
        selected = list(selection.get("selected_features", []))
        if len(selected) != int(cell["requested_feature_budget"]):
            raise Prompt16ExecutionError("supplemental selection support is not like-for-like")
        evaluation_path = fold_root / "evaluations" / f"cell_{cell['configuration_order']:03d}"
        identity = _evaluation_identity(
            entry=entry,
            fold_id=fold_id,
            cell=cell,
            selection_manifest_sha256=selection_manifest_sha,
        )
        sealed = _load_recursive_sealed(evaluation_path, identity)
        if sealed is not None:
            if _json(evaluation_path / "status.json").get("status") != "complete":
                raise Prompt16ExecutionError("sealed supplemental evaluation is not complete")
            completed_evaluations += 1
            continue
        _archive_incomplete(evaluation_path, archive_root)
        evaluation_path.mkdir(parents=True, exist_ok=False)
        _publish_stage(
            stage_queue,
            f"final_{model}",
            f"supplemental_dev:fold_{fold_id}:cell_{cell['configuration_order']:03d}",
            component=f"{cell['method_id']}_{model}_model_fit_and_evaluation",
            method_id=cell["method_id"],
            model=model,
            configuration_order=cell["configuration_order"],
            operation="fold_evaluation",
        )
        started = time.perf_counter()
        try:
            predictions, metrics, details = _fit_and_evaluate(
                cell=cell,
                selected=selected,
                train=train,
                validation=validation,
                predictors=predictors,
                matrix=entry["amendment"]["frozen_classical_execution_settings"],
                phase="dev",
                frozen_threshold=None,
            )
            prediction_auth = _locked_alignment_summary(
                predictions["case_id"].tolist(), predictions["target"].tolist()
            )
            expected_validation = scope_auth["validation"]["observed"]
            if (
                prediction_auth["ordered_case_id_sha256"]
                != expected_validation["ordered_case_id_sha256"]
                or prediction_auth["ordered_case_id_target_sha256"]
                != expected_validation["ordered_case_id_target_sha256"]
            ):
                raise Prompt16ExecutionError("supplemental prediction row alignment changed")
            write_parquet_atomic(
                evaluation_path / "predictions.parquet",
                predictions,
                required_columns=(
                    "case_id",
                    "target",
                    "score",
                    "decision_threshold",
                ),
                ordered_row_identity_column="case_id",
                overwrite=False,
            )
            write_json_atomic(evaluation_path / "metrics.json", metrics, overwrite=False)
            write_json_atomic(evaluation_path / "execution.json", details, overwrite=False)
            write_json_atomic(
                evaluation_path / "status.json",
                {
                    "status": "complete",
                    "evaluation_id": (
                        f"p16v2-dev-fold-{fold_id}-c{cell['configuration_order']:03d}"
                    ),
                    "fold_id": fold_id,
                    "configuration_order": cell["configuration_order"],
                    "cell": dict(cell),
                    "requested_feature_budget": cell["requested_feature_budget"],
                    "realized_support": len(selected),
                    "natural_support_like_for_like": True,
                    "prediction_alignment": prediction_auth,
                    "validation_target_used_for_fit": False,
                    "oot_opened": False,
                    "elapsed_seconds": time.perf_counter() - started,
                },
                overwrite=False,
            )
        except Exception as exc:
            write_json_atomic(
                evaluation_path / "failure.json",
                {
                    "status": "failed_unsealed_safe_to_resume",
                    "fold_id": fold_id,
                    "configuration_order": cell["configuration_order"],
                    "error": {"class": type(exc).__name__, "message": str(exc)},
                    "elapsed_seconds": time.perf_counter() - started,
                },
                overwrite=False,
            )
            raise
        _seal_recursive_directory(evaluation_path, identity)
        completed_evaluations += 1

    if completed_evaluations != 4:
        raise Prompt16ExecutionError("supplemental fold did not complete all four evaluations")
    accounting = {
        "schema_version": SCHEMA_VERSION,
        "phase": "supplemental_dev",
        "fold_id": fold_id,
        "target_free_ranking_generations_in_fold": 0,
        "global_llm_truncation_states_reused": 2,
        "registered_supervised_selector_fits": 2,
        "internal_stable_core_component_fits": 10,
        "expected_evaluations": 4,
        "completed_evaluations": 4,
        "oot_opened": False,
    }
    write_json_atomic(fold_root / "accounting.json", accounting)
    fold_manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "phase": "supplemental_dev",
        "fold_id": fold_id,
        "authorization_file_sha256": entry["authorization"][
            "_authorization_file_sha256"
        ],
        "amendment_internal_sha256": entry["authorization"]["amendment"][
            "internal_sha256"
        ],
        "matrix_manifest_sha256": entry["matrix_manifest_sha256"],
        "ranking_manifest_sha256": ranking_manifest_sha256,
        "scope_authentication": scope_auth,
        "completed_supervised_selector_fits": 2,
        "completed_internal_component_fits": 10,
        "completed_evaluations": 4,
        "oot_opened": False,
        "completed_at_utc": _utc_now(),
    }
    write_json_atomic(fold_root / "fold_manifest.json", fold_manifest)
    write_text_atomic(
        fold_root / "_SUCCESS",
        json.dumps(
            {"fold_manifest_sha256": file_sha256(fold_root / "fold_manifest.json")},
            sort_keys=True,
        )
        + "\n",
        overwrite=False,
    )
    del train, validation
    gc.collect()
    return fold_manifest


def _validate_controller_success(
    output_root: Path,
    entry: Mapping[str, Any],
) -> dict[str, Any] | None:
    if not (output_root / "_SUCCESS").is_file():
        return None
    success = _json(output_root / "_SUCCESS")
    manifest_path = output_root / "controller_manifest.json"
    if success.get("controller_manifest_sha256") != file_sha256(manifest_path):
        raise Prompt16ExecutionError("supplemental controller completion marker mismatch")
    manifest = _json(manifest_path)
    if manifest.get("status") != "complete" or manifest.get("completed_evaluations") != 20:
        raise Prompt16ExecutionError("supplemental controller manifest is incomplete")
    if manifest.get("authorization_file_sha256") != entry["authorization"].get(
        "_authorization_file_sha256"
    ):
        raise Prompt16ExecutionError("supplemental controller authorization changed")
    return manifest


def run_supplemental_dev_worker(
    *,
    authorization_path: str,
    stop_event: Any = None,
    stage_queue: Any = None,
    ram_ready_event: Any = None,
) -> dict[str, Any]:
    """Run or safely resume the one five-fold supplemental DEV operation."""

    entry = authenticate_supplemental_entry(authorization_path)
    output_root = Path(entry["authorization"]["paths"]["output_root"])
    completed = _validate_controller_success(output_root, entry)
    if completed is not None:
        return completed
    output_root.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        output_root / "controller_status.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "running_or_resuming",
            "operation": "supplemental_dev_only",
            "completed_folds": [
                fold_id
                for fold_id in FROZEN_FOLDS
                if (output_root / f"fold_{fold_id}" / "_SUCCESS").is_file()
            ],
            "oot_capability": False,
            "updated_at_utc": _utc_now(),
        },
    )
    ranking, ranking_manifest_sha, ranking_payload = ensure_target_free_ranking(
        entry=entry,
        output_root=output_root,
        stop_event=stop_event,
        stage_queue=stage_queue,
    )
    del ranking_payload
    global_states = ensure_global_llm_selection_states(
        entry=entry,
        output_root=output_root,
        ranking=ranking,
        ranking_manifest_sha256=ranking_manifest_sha,
    )
    fold_manifests: list[dict[str, Any]] = []
    for fold_id in FROZEN_FOLDS:
        _check_stop(stop_event)
        manifest = run_supplemental_fold(
            entry=entry,
            output_root=output_root,
            fold_id=fold_id,
            ranking=ranking,
            ranking_manifest_sha256=ranking_manifest_sha,
            global_states=global_states,
            stop_event=stop_event,
            stage_queue=stage_queue,
            ram_ready_event=ram_ready_event,
        )
        fold_manifests.append(manifest)
        write_json_atomic(
            output_root / "controller_status.json",
            {
                "schema_version": SCHEMA_VERSION,
                "status": "running_or_resuming",
                "operation": "supplemental_dev_only",
                "completed_folds": list(FROZEN_FOLDS[:fold_id]),
                "completed_supplemental_evaluations": fold_id * 4,
                "oot_capability": False,
                "updated_at_utc": _utc_now(),
            },
        )
    final_classical = classical_tree_identity(entry["plan"]["paths"]["dev_root"])
    if final_classical != entry["classical_tree"]:
        raise Prompt16ExecutionError("classical DEV tree changed during supplemental execution")
    no_oot = _assert_no_oot_state(entry["plan"])
    accounting = {
        **corrected_accounting(),
        "completed_target_free_llm_rankings": 1,
        "completed_global_llm_budget_truncation_states": 2,
        "completed_added_registered_supervised_selector_fits": 10,
        "completed_added_internal_supervised_component_fits": 50,
        "completed_added_dev_evaluations": 20,
        "authenticated_preserved_classical_dev_evaluations": 150,
        "authenticated_complete_amended_dev_evaluations": 170,
        "oot_opened": False,
    }
    write_json_atomic(output_root / "accounting.json", accounting)
    write_json_atomic(
        output_root / "classical_preservation.json",
        {
            "before": entry["classical_tree"],
            "after": final_classical,
            "byte_identical": True,
        },
    )
    write_json_atomic(output_root / "no_oot_state.json", no_oot)
    status = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "operation": "supplemental_dev_only",
        "completed_folds": list(FROZEN_FOLDS),
        "completed_supplemental_evaluations": 20,
        "authenticated_complete_amended_dev_evaluations": 170,
        "classical_tree_byte_identical": True,
        "oot_opened": False,
        "oot_capability": False,
        "completed_at_utc": _utc_now(),
    }
    write_json_atomic(output_root / "controller_status.json", status)
    controller_manifest = {
        **status,
        "authorization_file_sha256": entry["authorization"][
            "_authorization_file_sha256"
        ],
        "authorization_internal_sha256": entry["authorization"][
            "_authorization_internal_sha256"
        ],
        "amendment_internal_sha256": entry["authorization"]["amendment"][
            "internal_sha256"
        ],
        "matrix_manifest_sha256": entry["matrix_manifest_sha256"],
        "ranking_manifest_sha256": ranking_manifest_sha,
        "fold_manifest_sha256": {
            f"fold_{fold_id}": file_sha256(
                output_root / f"fold_{fold_id}" / "fold_manifest.json"
            )
            for fold_id in FROZEN_FOLDS
        },
        "classical_tree_sha256": final_classical["tree_manifest_sha256"],
        "accounting_sha256": file_sha256(output_root / "accounting.json"),
        "status_sha256": file_sha256(output_root / "controller_status.json"),
        "completed_evaluations": 20,
    }
    write_json_atomic(output_root / "controller_manifest.json", controller_manifest)
    write_text_atomic(
        output_root / "_SUCCESS",
        json.dumps(
            {
                "controller_manifest_sha256": file_sha256(
                    output_root / "controller_manifest.json"
                )
            },
            sort_keys=True,
        )
        + "\n",
        overwrite=False,
    )
    return controller_manifest


def record_supplemental_resource_stop(
    *,
    authorization_path: str | Path,
    supervisor_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    authorization, _ = load_supplemental_authorization(authorization_path)
    output_root = Path(authorization["paths"]["output_root"])
    stop_id = (
        f"resource_stop_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_"
        f"{uuid.uuid4().hex[:8]}"
    )
    path = output_root / "resource_stops" / stop_id
    path.mkdir(parents=True, exist_ok=False)
    write_json_atomic(
        path / "status.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "kind": "supplemental_resource_stop",
            "id": stop_id,
            "worker_partial_state": "preserved_unsealed_for_safe_resume",
            "supervisor": dict(supervisor_evidence),
            "recorded_at_utc": _utc_now(),
        },
        overwrite=False,
    )
    sealed = _seal_recursive_directory(
        path,
        {
            "operation": "supplemental_resource_stop_evidence",
            "authorization_file_sha256": authorization[
                "_authorization_file_sha256"
            ],
            "id": stop_id,
        },
    )
    return {
        "kind": "supplemental_resource_stop",
        "id": stop_id,
        "status": "complete",
        "manifest_sha256": sealed["manifest_sha256"],
    }


__all__ = [
    "AMENDMENT_SCHEMA_VERSION",
    "AUTHORIZATION_SCHEMA_VERSION",
    "FROZEN_LLM_MODEL",
    "SCHEMA_VERSION",
    "authenticate_supplemental_entry",
    "build_description_and_prompt_freeze",
    "active_prompt16_workers",
    "classical_evaluation_manifest_identity",
    "classical_tree_identity",
    "corrected_accounting",
    "ensure_target_free_ranking",
    "expanded_dev_evaluation_identities",
    "expanded_oot_evaluation_identities",
    "load_supplemental_amendment",
    "load_supplemental_authorization",
    "record_supplemental_resource_stop",
    "render_target_free_feature_descriptions",
    "run_supplemental_dev_worker",
    "run_supplemental_fold",
    "supplemental_cells",
    "prompt16_execution_locks",
]
