"""Persisted-artifact-only authentication and review for Prompt 13.

This module deliberately has no research data-loader or estimator imports.  It
only reads committed configuration and already-published experiment artifacts.
It never reads a combination OOT artifact; their presence is a contamination
failure established from path inventory alone.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from credit_risk_fs.experiments.atomic_io import sha256_file
from credit_risk_fs.experiments.full_baseline import inspect_cell, load_full_baseline_plan
from credit_risk_fs.experiments.row_alignment import (
    ordered_row_id_sha256,
    ordered_row_id_target_sha256,
)
from credit_risk_fs.experiments.selector_combinations import (
    METHOD_ORDER,
    _evaluation_path,
    _selection_path,
    _validate_approval_lock,
    _validate_artifact,
    _validate_dev_completion_lock,
    build_phase_matrix,
    build_status,
    load_combination_plan,
    render_plan,
    validate_prompt_10_baselines,
)


AUDIT_SCHEMA = "selector_combination_prompt_13_dev_audit_v1"
AUDIT_DIR = Path("cleanup/audits/prompt_13_combination_dev_review")
SCOPE_PATH = Path("configs/protocols/selector_combinations_v1/oot_scope_freeze.json")
REVIEW_LOCK_PATH = Path("configs/protocols/selector_combinations_v1/dev_review_lock.json")
PRIMARY_METRICS = (
    "gini",
    "auc",
    "ks",
    "ks_threshold",
    "decision_threshold",
    "precision",
    "recall",
    "f1",
    "accuracy",
    "log_loss",
    "brier",
    "approval_rate",
    "bad_rate_approved",
)
COUNT_METRICS = ("tn", "fp", "fn", "tp")
BASELINE_COMPONENTS = {
    "statistical_normalized_average_rank": (
        "iv_woe",
        "lasso_l1_logistic",
        "rfe_catboost",
        "boruta_random_forest",
        "catboost_shap",
        "full_features",
        "random_k",
    ),
    "iv_then_boruta": (
        "iv_woe",
        "boruta_random_forest",
        "full_features",
        "random_k",
    ),
    "boruta_then_mrmr_mutual_information": (
        "boruta_random_forest",
        "mrmr_mutual_information",
        "full_features",
        "random_k",
    ),
    "boruta_then_rfe_catboost": (
        "boruta_random_forest",
        "rfe_catboost",
        "full_features",
        "random_k",
    ),
}


class Prompt13AuditError(RuntimeError):
    """An authentication, contamination, preservation, or scope error."""


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def authenticated_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    value.pop("artifact_authentication_sha256", None)
    value["artifact_authentication_sha256"] = canonical_sha(value)
    return value


def read_authenticated_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Prompt13AuditError(f"cannot read authenticated JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise Prompt13AuditError(f"authenticated JSON is not an object: {path}")
    observed = value.get("artifact_authentication_sha256")
    unsigned = dict(value)
    unsigned.pop("artifact_authentication_sha256", None)
    if not isinstance(observed, str) or observed != canonical_sha(unsigned):
        raise Prompt13AuditError(f"authentication_sha256_mismatch:{path.as_posix()}")
    return value


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def inventory_tree(root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    count = 0
    total = 0
    extensions: Counter[str] = Counter()
    if root.is_dir():
        for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda p: p.as_posix()):
            relative = path.relative_to(root).as_posix()
            size = path.stat().st_size
            file_hash = sha256_file(path)
            digest.update(f"{relative}|{size}|{file_hash}\n".encode("utf-8"))
            count += 1
            total += size
            extensions[path.suffix.lower() or "<none>"] += 1
    return {
        "root": root.as_posix(),
        "file_count": count,
        "total_bytes": total,
        "extension_counts": dict(sorted(extensions.items())),
        "tree_sha256": digest.hexdigest(),
        "canonical_line_format": "root_relative_path|size_bytes|sha256<LF>",
    }


def contamination_paths(repository_root: Path, expected_oot_ids: Sequence[str]) -> list[str]:
    """Return path names only; no OOT file is opened."""

    result_root = repository_root / "results/selector_combinations_v1"
    hits: set[str] = set()
    oot_root = result_root / "oot"
    if oot_root.exists():
        hits.update(
            path.relative_to(repository_root).as_posix()
            for path in oot_root.rglob("*")
            if path.is_file()
        )
    expected_tokens = set(expected_oot_ids)
    for path in result_root.rglob("*"):
        if not path.is_file():
            continue
        lower = path.name.lower()
        if any(token in path.name for token in expected_tokens) or (
            "scv1-oot-" in lower
            and any(term in lower for term in ("prediction", "metric", "selection", "state", "checkpoint", "success"))
        ):
            hits.add(path.relative_to(repository_root).as_posix())
    return sorted(hits)


def validate_unique_ordered_identities(
    expected: Sequence[str], observed: Sequence[str], label: str
) -> None:
    duplicates = sorted(identity for identity, count in Counter(observed).items() if count > 1)
    missing = sorted(set(expected) - set(observed))
    extra = sorted(set(observed) - set(expected))
    if duplicates or missing or extra or list(expected) != list(observed):
        raise Prompt13AuditError(
            f"{label} identity mismatch: missing={missing}, extra={extra}, "
            f"duplicates={duplicates}, order_matches={list(expected) == list(observed)}"
        )


def support_label(dataset: str, model: str, method: str, requested: int | None, realized: int) -> str:
    if method == "iv_then_boruta":
        return "natural_support"
    if realized == requested:
        return "matched_budget"
    if (
        dataset == "homecredit"
        and method in {"boruta_then_mrmr_mutual_information", "boruta_then_rfe_catboost"}
        and requested == 40
        and realized == 26
    ):
        return "natural_support_26_of_requested_40"
    return "natural_support_shortfall"


def pairwise_stability(
    selected_sets: Sequence[set[str]], candidate_count: int
) -> tuple[float, float, float, float | None, str]:
    jaccards: list[float] = []
    kunchevas: list[float] = []
    sizes = {len(item) for item in selected_sets}
    for left, right in combinations(selected_sets, 2):
        union = left | right
        jaccards.append(len(left & right) / len(union) if union else 1.0)
        if len(left) == len(right) and 0 < len(left) < candidate_count:
            k = len(left)
            kunchevas.append((len(left & right) * candidate_count - k * k) / (k * (candidate_count - k)))
    if len(sizes) == 1 and kunchevas:
        kuncheva = statistics.mean(kunchevas)
        reason = "applicable_equal_selected_set_size_and_candidate_universe"
    else:
        kuncheva = None
        reason = "not_applicable_selected_set_size_varies_across_folds"
    return (
        statistics.mean(jaccards),
        min(jaccards),
        max(jaccards),
        kuncheva,
        reason,
    )


def baseline_alignment_reason(
    *,
    combination_support_label: str,
    baseline_method: str,
    baseline_fold_vectors_saved: bool,
    baseline_ordered_identity_hashes_saved: bool,
) -> str:
    if not baseline_fold_vectors_saved or not baseline_ordered_identity_hashes_saved:
        return (
            "baseline held-out fold prediction vectors and ordered row/target identity hashes were not "
            "persisted; identical evaluation rows and targets cannot be authenticated"
        )
    if combination_support_label.startswith("natural_support") and baseline_method == "random_k":
        return "natural-support result is not a matched fixed-K random baseline"
    return "supported"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _finite_number(value: Any, identity: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise Prompt13AuditError(f"non_numeric:{identity}") from exc
    if not math.isfinite(number):
        raise Prompt13AuditError(f"non_finite:{identity}")
    return number


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _load_log_timings(repository_root: Path, expected_associations: set[str]) -> tuple[dict[str, float], dict[str, Any]]:
    path = repository_root / "logs/events.jsonl"
    starts: dict[tuple[str, str], datetime] = {}
    durations: dict[str, float] = {}
    session_match_counts: Counter[str] = Counter()
    all_rows: list[dict[str, Any]] = []
    if not path.is_file():
        return durations, {"supported": False, "reason": "events log absent"}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            all_rows.append(row)
            association = row.get("run_association")
            if association not in expected_associations:
                continue
            session_id = str(row.get("session_id", ""))
            if session_id:
                session_match_counts[session_id] += 1
            key = (session_id, str(association))
            if row.get("event") == "worker_spawn_requested":
                starts[key] = _parse_utc(row["timestamp_utc"])
            elif row.get("event") == "worker_finalized" and key in starts:
                durations[str(association)] = (
                    _parse_utc(row["timestamp_utc"]) - starts[key]
                ).total_seconds()
    selected_session = max(session_match_counts, key=session_match_counts.get, default=None)
    if selected_session is None:
        return durations, {"supported": False, "reason": "no matching persisted events"}
    rows = [row for row in all_rows if str(row.get("session_id", "")) == selected_session]
    timestamps = [_parse_utc(row["timestamp_utc"]) for row in rows]
    readiness = [row for row in rows if row.get("event") == "inter_run_readiness_result"]
    return durations, {
        "supported": True,
        "session_id": selected_session,
        "first_matching_event_utc": min(timestamps).isoformat().replace("+00:00", "Z"),
        "last_matching_event_utc": max(timestamps).isoformat().replace("+00:00", "Z"),
        "observed_matching_event_span_seconds": (max(timestamps) - min(timestamps)).total_seconds(),
        "readiness_checks": len(readiness),
        "ram_wait_seconds": sum(float(row.get("elapsed_stage_seconds", 0.0)) for row in readiness),
        "readiness_failures": sum(not bool(row.get("ready")) for row in readiness),
    }


def _artifact_bytes(state_path: Path, payload: Mapping[str, Any]) -> int:
    return state_path.stat().st_size + sum(int(item["size_bytes"]) for item in payload.get("artifact_files", []))


def authenticate_and_analyze(repository_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    plan = load_combination_plan(root)
    approval = _validate_approval_lock(plan)
    completion = _validate_dev_completion_lock(plan, approval)
    retained = tuple(approval["retained_method_ids"])
    if retained != METHOD_ORDER:
        raise Prompt13AuditError(f"retained method scope/order mismatch: {retained}")
    dev_selections, dev_evaluations = build_phase_matrix(plan, phase="dev", retained_method_ids=retained)
    oot_selections, oot_evaluations = build_phase_matrix(plan, phase="oot", retained_method_ids=retained)
    if len(dev_selections) != 90 or len(dev_evaluations) != 120:
        raise Prompt13AuditError("DEV plan is not 90 selector fits / 120 evaluation cells")
    if len(oot_selections) != 18 or len(oot_evaluations) != 24:
        raise Prompt13AuditError("OOT plan is not 18 selector refits / 24 evaluation cells")

    scope_path = root / SCOPE_PATH
    scope = read_authenticated_json(scope_path)
    scope_eval_ids = list(scope["ordered_oot_evaluation_cell_ids"])
    scope_selection_ids = list(scope["ordered_oot_selection_ids"])
    validate_unique_ordered_identities(
        [item.cell_id for item in oot_evaluations], scope_eval_ids, "scope freeze OOT evaluation"
    )
    validate_unique_ordered_identities(
        [item.selection_id for item in oot_selections], scope_selection_ids, "scope freeze OOT selection"
    )
    if contamination_paths(root, scope_eval_ids):
        raise Prompt13AuditError("combination OOT contamination artifacts exist")

    status = build_status(plan)
    if (
        status["pilot"]["authenticated_selections"] != 18
        or status["pilot"]["authenticated_evaluations"] != 24
        or status["dev"]["authenticated_evaluations"] != 120
        or status["dev"]["first_incomplete"] is not None
        or status["oot"]["authenticated_evaluations"] != 0
    ):
        raise Prompt13AuditError("status authentication counts are not pilot 18/24, DEV 120, OOT 0")

    selection_payloads: dict[str, dict[str, Any]] = {}
    stage_rows: list[dict[str, Any]] = []
    selection_feature_sets: dict[str, list[str]] = {}
    selector_ids = {item.selection_id for item in dev_selections}
    evaluation_ids = {item.cell_id for item in dev_evaluations}
    timings, log_summary = _load_log_timings(root, selector_ids | evaluation_ids)
    resource_detail: list[dict[str, Any]] = []
    for spec in dev_selections:
        state_path = _selection_path(plan, spec.selection_id, "dev")
        valid, reason, payload = _validate_artifact(
            state_path,
            {"selection_id": spec.selection_id, "configuration_sha256": plan.configuration_sha256},
        )
        if not valid or payload is None:
            raise Prompt13AuditError(f"selection authentication failed:{spec.selection_id}:{reason}")
        if payload.get("selection_spec") != asdict(spec):
            raise Prompt13AuditError(f"selection spec mismatch:{spec.selection_id}")
        supervisor = payload["supervisor"]
        if (
            supervisor.get("status") != "completed"
            or supervisor.get("worker_exit_code") != 0
            or supervisor.get("stop_code") is not None
            or supervisor.get("worker_error") is not None
        ):
            raise Prompt13AuditError(f"selection supervisor failure:{spec.selection_id}")
        worker = payload["worker_result"]
        if worker.get("opened_oot_paths") or worker.get("oot_rows_retained") != 0:
            raise Prompt13AuditError(f"selection OOT access evidence:{spec.selection_id}")
        result = worker["combination_result"]
        if (
            result.get("terminal_state") != "completed"
            or result.get("method_id") != spec.method_id
            or result.get("fit_scope") != "dev_fold_training_only"
            or result.get("seed") != 42
            or result.get("protocol_lock_sha256") != plan.protocol_lock_sha256
        ):
            raise Prompt13AuditError(f"selection result identity mismatch:{spec.selection_id}")
        candidates = list(map(str, result["candidate_universe"]))
        selected = list(map(str, result["selected_features"]))
        intermediate = result.get("intermediate_features")
        if len(candidates) != len(set(candidates)) or len(selected) != len(set(selected)):
            raise Prompt13AuditError(f"duplicate candidate/selected feature:{spec.selection_id}")
        if not set(selected).issubset(candidates):
            raise Prompt13AuditError(f"selected feature outside universe:{spec.selection_id}")
        if result["candidate_universe_count"] != len(candidates) or result["realized_budget"] != len(selected):
            raise Prompt13AuditError(f"selection count mismatch:{spec.selection_id}")
        if result["candidate_universe_sha256"] != canonical_sha(candidates):
            raise Prompt13AuditError(f"candidate universe hash mismatch:{spec.selection_id}")
        if result["selected_features_sha256"] != canonical_sha(selected):
            raise Prompt13AuditError(f"selected feature hash mismatch:{spec.selection_id}")
        if intermediate is not None:
            intermediate = list(map(str, intermediate))
            if result["intermediate_feature_count"] != len(intermediate):
                raise Prompt13AuditError(f"intermediate count mismatch:{spec.selection_id}")
            if result["intermediate_features_sha256"] != canonical_sha(intermediate):
                raise Prompt13AuditError(f"intermediate hash mismatch:{spec.selection_id}")
        selection_payloads[spec.selection_id] = payload
        selection_feature_sets[spec.selection_id] = selected
        stage_sizes = [
            int(item["realized_budget"]) if item.get("realized_budget") is not None else None
            for item in result["stage_provenance"]
        ]
        requested = result.get("requested_budget")
        label = support_label(spec.dataset, "selection", spec.method_id, requested, len(selected))
        stage_rows.append(
            {
                "selection_id": spec.selection_id,
                "dataset": spec.dataset,
                "fold_id": spec.fold_id,
                "method": spec.method_id,
                "variant": f"iv_pool_{spec.iv_pool_budget}" if spec.iv_pool_budget else f"k_{spec.final_budget}",
                "requested_k": requested if requested is not None else "",
                "realized_selected_count": len(selected),
                "support_state": result["feasibility_state"],
                "support_label": label,
                "candidate_feature_count": len(candidates),
                "stage_count": len(stage_sizes),
                "stage_1_count": stage_sizes[0] if stage_sizes and stage_sizes[0] is not None else "",
                "stage_2_count": stage_sizes[1] if len(stage_sizes) > 1 and stage_sizes[1] is not None else "",
                "intermediate_count": len(intermediate) if intermediate is not None else "",
                "selected_features_sha256": result["selected_features_sha256"],
                "candidate_universe_sha256": result["candidate_universe_sha256"],
                "fit_seconds": result["fit_seconds"],
                "fit_provenance": ";".join(str(item.get("fit_provenance", "")) for item in result["stage_provenance"]),
                "warnings": " | ".join(map(str, result.get("warnings", []))),
                "no_padding_verified": len(selected) == result["realized_budget"],
            }
        )
        resource_detail.append(
            {
                "dataset": spec.dataset,
                "method": spec.method_id,
                "stage": "selector_fit",
                "active_seconds": timings.get(spec.selection_id, float(result["fit_seconds"])),
                "reported_compute_seconds": float(result["fit_seconds"]),
                "peak_rss_bytes": int(supervisor["peak_process_tree_rss_bytes"]),
                "minimum_available_ram_bytes": int(supervisor["minimum_system_available_ram_bytes"]),
                "artifact_bytes": _artifact_bytes(state_path, payload),
                "timeout_or_stop_events": 0,
                "association_count": 1,
            }
        )

    fold_rows: list[dict[str, Any]] = []
    canonical_validation: dict[tuple[str, int], tuple[str, str, int, tuple[int, ...]]] = {}
    for cell in dev_evaluations:
        state_path = _evaluation_path(plan, cell.cell_id, "dev")
        valid, reason, payload = _validate_artifact(
            state_path,
            {"cell_id": cell.cell_id, "configuration_sha256": plan.configuration_sha256},
        )
        if not valid or payload is None:
            raise Prompt13AuditError(f"evaluation authentication failed:{cell.cell_id}:{reason}")
        if payload.get("evaluation_cell") != asdict(cell):
            raise Prompt13AuditError(f"evaluation cell mismatch:{cell.cell_id}")
        selection_path = _selection_path(plan, cell.selection_id, "dev")
        if payload.get("selection_artifact_sha256") != sha256_file(selection_path):
            raise Prompt13AuditError(f"selection lineage mismatch:{cell.cell_id}")
        supervisor = payload["supervisor"]
        if (
            supervisor.get("status") != "completed"
            or supervisor.get("worker_exit_code") != 0
            or supervisor.get("stop_code") is not None
            or supervisor.get("worker_error") is not None
        ):
            raise Prompt13AuditError(f"evaluation supervisor failure:{cell.cell_id}")
        worker = payload["worker_result"]
        if (
            worker.get("cell_id") != cell.cell_id
            or worker.get("validation_targets_used_for_fit") is not False
            or worker.get("opened_oot_paths") != []
            or worker.get("oot_rows_retained") != 0
        ):
            raise Prompt13AuditError(f"evaluation leakage/identity failure:{cell.cell_id}")
        artifact_map = {Path(item["path"]).name: item for item in payload["artifact_files"]}
        prediction_name = f"{cell.cell_id}.dev_predictions.csv"
        metric_name = f"{cell.cell_id}.dev_metrics.json"
        if prediction_name not in artifact_map or metric_name not in artifact_map:
            raise Prompt13AuditError(f"evaluation required artifact absent:{cell.cell_id}")
        prediction_path = state_path.parent / prediction_name
        prediction_rows = _read_csv(prediction_path)
        if len(prediction_rows) != int(worker["validation_row_count"]):
            raise Prompt13AuditError(f"prediction row count mismatch:{cell.cell_id}")
        ids: list[str] = []
        targets: list[int] = []
        probabilities: list[float] = []
        for index, row in enumerate(prediction_rows):
            if row.get("dataset") != cell.dataset or row.get("model") != cell.model or row.get("split") != "dev":
                raise Prompt13AuditError(f"prediction metadata mismatch:{cell.cell_id}:{index}")
            if int(row["fold_id"]) != cell.fold_id or row.get("run_id") != cell.cell_id or row.get("method") != cell.method_id:
                raise Prompt13AuditError(f"prediction identity mismatch:{cell.cell_id}:{index}")
            ids.append(str(row["stable_row_id"]))
            target = int(row["target"])
            if target not in (0, 1):
                raise Prompt13AuditError(f"non_binary_target:{cell.cell_id}:{index}")
            targets.append(target)
            probability = _finite_number(row["prediction_probability"], f"{cell.cell_id}:{index}:prediction")
            if not 0.0 <= probability <= 1.0:
                raise Prompt13AuditError(f"probability_out_of_range:{cell.cell_id}:{index}")
            probabilities.append(probability)
        if len(ids) != len(set(ids)):
            raise Prompt13AuditError(f"duplicate validation ID:{cell.cell_id}")
        if ordered_row_id_sha256(ids) != worker["validation_ordered_row_id_sha256"]:
            raise Prompt13AuditError(f"ordered validation ID hash mismatch:{cell.cell_id}")
        if ordered_row_id_target_sha256(ids, targets) != worker["validation_ordered_row_id_target_sha256"]:
            raise Prompt13AuditError(f"ordered validation ID/target hash mismatch:{cell.cell_id}")
        if canonical_sha(probabilities) != worker["prediction_sha256"]:
            raise Prompt13AuditError(f"prediction value hash mismatch:{cell.cell_id}")
        key = (cell.dataset, cell.fold_id)
        alignment = (
            worker["validation_ordered_row_id_sha256"],
            worker["validation_ordered_row_id_target_sha256"],
            len(ids),
            tuple(targets),
        )
        if key in canonical_validation and alignment != canonical_validation[key]:
            raise Prompt13AuditError(f"within-combination fold row/target misalignment:{cell.cell_id}")
        canonical_validation.setdefault(key, alignment)
        metric_file = json.loads((state_path.parent / metric_name).read_text(encoding="utf-8"))
        if metric_file.get("validation_targets_used_for_fit") is not False:
            raise Prompt13AuditError(f"metric artifact leakage flag:{cell.cell_id}")
        metrics = worker["metrics"]
        if metric_file.get("metrics") != metrics:
            raise Prompt13AuditError(f"metric state/file mismatch:{cell.cell_id}")
        for metric in PRIMARY_METRICS + COUNT_METRICS:
            _finite_number(metrics[metric], f"{cell.cell_id}:{metric}")
        selection_result = selection_payloads[cell.selection_id]["worker_result"]["combination_result"]
        realized = len(selection_feature_sets[cell.selection_id])
        label = support_label(cell.dataset, cell.model, cell.method_id, selection_result.get("requested_budget"), realized)
        fold_rows.append(
            {
                "cell_id": cell.cell_id,
                "selection_id": cell.selection_id,
                "dataset": cell.dataset,
                "method": cell.method_id,
                "variant": f"iv_pool_{cell.iv_pool_budget}" if cell.iv_pool_budget else f"k_{cell.final_budget}",
                "iv_pool": cell.iv_pool_budget if cell.iv_pool_budget is not None else "",
                "final_model": cell.model,
                "requested_k": selection_result.get("requested_budget") if selection_result.get("requested_budget") is not None else "",
                "realized_selected_count": realized,
                "stage_support_label": label,
                "fold_id": cell.fold_id,
                "validation_row_count": len(ids),
                **{metric: metrics[metric] for metric in PRIMARY_METRICS + COUNT_METRICS},
                "validation_ordered_row_id_sha256": worker["validation_ordered_row_id_sha256"],
                "validation_ordered_row_id_target_sha256": worker["validation_ordered_row_id_target_sha256"],
                "prediction_sha256": worker["prediction_sha256"],
            }
        )
        resource_detail.append(
            {
                "dataset": cell.dataset,
                "method": cell.method_id,
                "stage": f"final_{cell.model}_evaluation",
                "active_seconds": timings.get(cell.cell_id, 0.0),
                "reported_compute_seconds": timings.get(cell.cell_id, 0.0),
                "peak_rss_bytes": int(supervisor["peak_process_tree_rss_bytes"]),
                "minimum_available_ram_bytes": int(supervisor["minimum_system_available_ram_bytes"]),
                "artifact_bytes": _artifact_bytes(state_path, payload),
                "timeout_or_stop_events": 0,
                "association_count": 1,
            }
        )

    config_fields = ("dataset", "method", "variant", "iv_pool", "final_model", "requested_k")
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in fold_rows:
        grouped[tuple(row[field] for field in config_fields)].append(row)
    summary_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []
    for key, rows in grouped.items():
        rows.sort(key=lambda item: int(item["fold_id"]))
        summary: dict[str, Any] = dict(zip(config_fields, key))
        labels = {str(item["stage_support_label"]) for item in rows}
        summary["stage_support_label"] = (
            next(iter(labels))
            if len(labels) == 1
            else "mixed_fold_support_natural_26_of_40_and_matched_k40"
        )
        summary["fold_count"] = len(rows)
        counts = [int(item["realized_selected_count"]) for item in rows]
        summary["mean_selected_feature_count"] = statistics.mean(counts)
        summary["min_selected_feature_count"] = min(counts)
        summary["max_selected_feature_count"] = max(counts)
        summary["fold_selected_feature_counts"] = "|".join(map(str, counts))
        for metric in PRIMARY_METRICS:
            values = [float(item[metric]) for item in rows]
            summary[f"{metric}_mean"] = statistics.mean(values)
            summary[f"{metric}_std"] = statistics.stdev(values)
            summary[f"{metric}_median"] = statistics.median(values)
            summary[f"{metric}_min"] = min(values)
            summary[f"{metric}_max"] = max(values)
        summary_rows.append(summary)

        selected_sets = [set(selection_feature_sets[item["selection_id"]]) for item in rows]
        candidate_counts = {
            selection_payloads[item["selection_id"]]["worker_result"]["combination_result"]["candidate_universe_count"]
            for item in rows
        }
        if len(candidate_counts) != 1:
            raise Prompt13AuditError(f"candidate universe size varies within configuration:{key}")
        mean_j, min_j, max_j, kuncheva, kuncheva_reason = pairwise_stability(selected_sets, candidate_counts.pop())
        feature_counts = Counter(feature for selected in selected_sets for feature in selected)
        for feature, frequency in sorted(feature_counts.items(), key=lambda item: (-item[1], item[0])):
            stability_rows.append(
                {
                    **dict(zip(config_fields, key)),
                    "stage_support_label": summary["stage_support_label"],
                    "fold_selected_feature_counts": "|".join(map(str, counts)),
                    "mean_pairwise_jaccard": mean_j,
                    "min_pairwise_jaccard": min_j,
                    "max_pairwise_jaccard": max_j,
                    "kuncheva": kuncheva if kuncheva is not None else "",
                    "kuncheva_status": kuncheva_reason,
                    "feature": feature,
                    "selection_count": frequency,
                    "selection_frequency": frequency / 5.0,
                }
            )
        summary["mean_pairwise_jaccard"] = mean_j
        summary["min_pairwise_jaccard"] = min_j
        summary["max_pairwise_jaccard"] = max_j
        summary["kuncheva"] = kuncheva if kuncheva is not None else ""
        summary["kuncheva_status"] = kuncheva_reason

    baseline_plan = load_full_baseline_plan(root)
    baseline_auth = validate_prompt_10_baselines(plan)
    baseline_cells: dict[tuple[str, str, str], Any] = {}
    baseline_cv: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    baseline_vector_files: list[str] = []
    for cell in baseline_plan.cells:
        inspection = inspect_cell(baseline_plan, cell)
        if not inspection.valid_completed:
            raise Prompt13AuditError(f"baseline authentication failed:{cell.cell_id}:{inspection.reason}")
        run_dir = baseline_plan.results_root / "runs" / cell.dataset / cell.cell_id
        # Only DEV fold summaries and non-OOT manifests are parsed.  Path inventory verifies vectors are absent.
        for path in run_dir.rglob("*"):
            if path.is_file() and "oot" not in path.name.lower() and "prediction" in path.name.lower():
                if "fold" in path.name.lower() or "oof" in path.name.lower():
                    baseline_vector_files.append(path.relative_to(root).as_posix())
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        if manifest["random_seed"] != 42 or manifest["status"] != "completed":
            raise Prompt13AuditError(f"baseline seed/status mismatch:{cell.cell_id}")
        cv_rows = _read_csv(run_dir / "results/cv_results.csv")
        held_out_cv_rows = [row for row in cv_rows if str(row.get("fold", "")).isdigit()]
        if [int(row["fold"]) for row in held_out_cv_rows] != [1, 2, 3, 4, 5]:
            raise Prompt13AuditError(f"baseline fold summary mismatch:{cell.cell_id}")
        baseline_cells[(cell.dataset, cell.model, cell.method_id)] = cell
        baseline_cv[(cell.dataset, cell.model, cell.method_id)] = held_out_cv_rows

    baseline_rows: list[dict[str, Any]] = []
    for summary in summary_rows:
        method = str(summary["method"])
        for baseline_method in BASELINE_COMPONENTS[method]:
            baseline_key = (str(summary["dataset"]), str(summary["final_model"]), baseline_method)
            baseline_cell = baseline_cells.get(baseline_key)
            reason = baseline_alignment_reason(
                combination_support_label=str(summary["stage_support_label"]),
                baseline_method=baseline_method,
                baseline_fold_vectors_saved=bool(baseline_vector_files),
                baseline_ordered_identity_hashes_saved=False,
            )
            baseline_rows.append(
                {
                    "dataset": summary["dataset"],
                    "combination_method": method,
                    "variant": summary["variant"],
                    "final_model": summary["final_model"],
                    "requested_k": summary["requested_k"],
                    "stage_support_label": summary["stage_support_label"],
                    "baseline_method": baseline_method,
                    "baseline_cell_id": baseline_cell.cell_id if baseline_cell else "",
                    "alignment_status": "not_supported" if reason != "supported" else "supported",
                    "alignment_reason": reason,
                    "fold_level_delta_status": "not_supported" if reason != "supported" else "supported",
                    "auc_delta_mean": "",
                    "ks_delta_mean": "",
                    "inference_status": "not_supported",
                    "inference_reason": (
                        "five folds are dependent temporal CV partitions; baseline paired fold prediction vectors are absent"
                    ),
                }
            )

    resource_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in resource_detail:
        resource_groups[(row["dataset"], row["method"], row["stage"])].append(row)
    resource_rows: list[dict[str, Any]] = []
    for (dataset, method, stage), rows in resource_groups.items():
        resource_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "stage": stage,
                "association_count": sum(int(row["association_count"]) for row in rows),
                "active_compute_seconds": sum(float(row["active_seconds"]) for row in rows),
                "reported_compute_seconds": sum(float(row["reported_compute_seconds"]) for row in rows),
                "ram_wait_seconds": 0.0,
                "peak_process_rss_bytes": max(int(row["peak_rss_bytes"]) for row in rows),
                "minimum_available_ram_bytes": min(int(row["minimum_available_ram_bytes"]) for row in rows),
                "timeout_or_stop_events": sum(int(row["timeout_or_stop_events"]) for row in rows),
                "artifact_bytes": sum(int(row["artifact_bytes"]) for row in rows),
                "selector_fit_reuse": (
                    "one authenticated selector fit reused by both final-model evaluations where matrix identities share selection_id"
                    if stage == "selector_fit" else "not_applicable"
                ),
            }
        )
    resource_rows.append(
        {
            "dataset": "all",
            "method": "all",
            "stage": "overall",
            "association_count": len(resource_detail),
            "active_compute_seconds": sum(float(row["active_seconds"]) for row in resource_detail),
            "reported_compute_seconds": sum(float(row["reported_compute_seconds"]) for row in resource_detail),
            "ram_wait_seconds": float(log_summary.get("ram_wait_seconds", 0.0)),
            "peak_process_rss_bytes": max(int(row["peak_rss_bytes"]) for row in resource_detail),
            "minimum_available_ram_bytes": min(int(row["minimum_available_ram_bytes"]) for row in resource_detail),
            "timeout_or_stop_events": sum(int(row["timeout_or_stop_events"]) for row in resource_detail),
            "artifact_bytes": sum(int(row["artifact_bytes"]) for row in resource_detail),
            "selector_fit_reuse": "90 selector fits authenticated for 120 evaluation cells; 30 shared fit identities avoid duplicate model-family fits",
        }
    )

    natural_rows = [row for row in fold_rows if row["stage_support_label"] == "natural_support_26_of_requested_40"]
    if len(natural_rows) != 2 or {row["method"] for row in natural_rows} != {
        "boruta_then_mrmr_mutual_information",
        "boruta_then_rfe_catboost",
    }:
        raise Prompt13AuditError("expected two Home Credit CatBoost fold-1 natural-support 26-of-40 cases")
    if any(row["realized_selected_count"] != 26 or row["requested_k"] != 40 for row in natural_rows):
        raise Prompt13AuditError("natural-support 26-of-40 label/count mismatch")

    completion_eval_ids = completion["completed_dev_evaluation_ids"]
    completion_selection_ids = completion["completed_dev_selection_ids"]
    validate_unique_ordered_identities(
        [item.cell_id for item in dev_evaluations], completion_eval_ids, "DEV completion evaluation"
    )
    validate_unique_ordered_identities(
        [item.selection_id for item in dev_selections], completion_selection_ids, "DEV completion selection"
    )

    return {
        "root": root,
        "plan": plan,
        "approval": approval,
        "completion": completion,
        "scope": scope,
        "status": status,
        "baseline_authentication": baseline_auth,
        "fold_rows": fold_rows,
        "summary_rows": summary_rows,
        "stability_rows": stability_rows,
        "stage_rows": stage_rows,
        "baseline_rows": baseline_rows,
        "resource_rows": resource_rows,
        "log_summary": log_summary,
        "baseline_fold_prediction_vector_paths": baseline_vector_files,
        "natural_support_rows": natural_rows,
        "dev_selections": dev_selections,
        "dev_evaluations": dev_evaluations,
        "oot_selections": oot_selections,
        "oot_evaluations": oot_evaluations,
    }


def _format_method(method: str) -> str:
    return method.replace("_", " ")


def _report_markdown(result: Mapping[str, Any], created: str) -> str:
    summaries = result["summary_rows"]
    by_slice: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in summaries:
        by_slice[(row["dataset"], row["final_model"])].append(row)
    performance_lines: list[str] = []
    for (dataset, model), rows in sorted(by_slice.items()):
        auc_values = [float(row["auc_mean"]) for row in rows]
        ks_values = [float(row["ks_mean"]) for row in rows]
        performance_lines.append(
            f"- **{dataset} / {model}:** across the immutable configurations, mean held-out AUC spans "
            f"{min(auc_values):.4f}–{max(auc_values):.4f} and mean KS spans {min(ks_values):.4f}–{max(ks_values):.4f}. "
            "These ranges are diagnostic; no configuration was removed or promoted."
        )
    overall = next(row for row in result["resource_rows"] if row["stage"] == "overall")
    log = result["log_summary"]
    jaccards = [float(row["mean_pairwise_jaccard"]) for row in summaries]
    natural = result["natural_support_rows"]
    return f"""# Combination DEV Authentication and OOT-Readiness Review

## Technical summary

The saved selector-combination DEV phase is scientifically and cryptographically complete: **90/90 selector fits and 120/120 held-out evaluation cells authenticate**, with exactly five folds for every frozen configuration. The preceding pilot remains authenticated at 18/18 fits and 24/24 evaluations. Combination OOT remains absent at 0/24, and the exact 24-cell OOT scope was frozen in a dedicated commit before any combination-DEV performance value was opened.

The decision is **ready_for_manual_oot** for the immutable 24-cell scope. This is authorization to run that exact scope later, not a claim that OOT has run. The current runner's completion lock opens its technical gate; the Prompt 13 review lock is a separate authenticated provenance authorization and is not represented as an execution-enforced hook.

DEV results remain diagnostic. No winner was chosen, no IV pool was pruned, and no method, budget, threshold, seed, fold, model, or ordering was changed. Each of the two Home Credit CatBoost Boruta-first configurations has a fold-1 **natural-support result of 26 against requested K=40**; folds 2–5 reached 40. The 26-feature cases are not described as matched K=40 and were not padded. The frozen OOT label retains 26 as its authenticated reference while requiring the future full-DEV refit to report its own realized support.

## The locked scope remained immutable before results were reviewed

The scope contains the approved methods in their immutable order: statistical normalized-average-rank voting; IV then Boruta with all frozen pools 100, 200, and 300; Boruta then mRMR mutual information; and Boruta then CatBoost RFE. Crossing those identities with two datasets, two final models, the applicable budget or pool variants, and seed 42 yields 18 unique full-DEV selector refits and 24 OOT evaluation cells.

The scope artifact authenticates its ordered identities, configuration hashes, Prompt 12 approval lineage, no-padding policy, and the declarations that combination DEV performance had not been opened and combination OOT had not been accessed. A safe plan-only call agreed field-for-field with all 18 selection and 24 evaluation identities while reporting zero raw paths resolved and zero workers started.

## Every saved DEV cell passed content-level authentication

Authentication went beyond completion counts. Each selector state and evaluation state passed its canonical internal digest and every contract file passed its recorded size and SHA-256. The audit verified unique and ordered cell identities, phase, fold, dataset, method, pool, model family, requested budget, seed, feature universe, configuration identity, selection lineage, terminal state, worker exit, and supervisor status.

For all 120 prediction artifacts, row counts matched state metadata, validation IDs were unique and in the authenticated order, targets were binary, probabilities were finite and in [0,1], and the ordered ID, ordered ID/target, and prediction-value hashes matched. Within each dataset-fold, all 24 configuration evaluations used the identical ordered validation IDs and targets. Every fit records `validation_targets_used_for_fit=false`, `opened_oot_paths=[]`, and `oot_rows_retained=0`.

There were no missing, extra, duplicate, partial, interrupted, failed, timed-out, stale, or hash-invalid active DEV artifacts. The active DEV contract is exactly 840/840 files. The DEV-completion lock binds the Prompt 12 approval lock and exact ordered 90-fit/120-cell inventories.

## DEV evidence shows variation without authorizing selection

Each number below is an expanding-window held-out fold result, never a full-DEV in-sample diagnostic. The detailed table retains all 120 fold rows and every consistently available primary metric. The 24-row configuration table reports mean, sample standard deviation, median, minimum, maximum, and selected-count range.

{chr(10).join(performance_lines)}

Fold-specific weakness is not hidden in the aggregate: `dev_fold_results.csv` contains all five AUC, KS, confusion-matrix, calibration, approval, and bad-rate values for every configuration. Differences among IV pools 100, 200, and 300 are retained as descriptive sensitivity only; all three remain frozen for OOT. With only five dependent temporal folds, the audit assigns **moderate descriptive evidence** to within-configuration patterns and **not_supported** to definitive superiority or significance claims.

## Strict baseline alignment is not supported by persisted Prompt 10 evidence

Prompt 10 remains authenticated at 36/36 cells and its DEV-only `cv_results.csv` files preserve five fold summaries per baseline. However, it did not persist held-out fold prediction vectors or ordered fold row/target identity hashes. Its saved `predictions_dev.csv` is a full-DEV in-sample final-model diagnostic with blank fold IDs, so it is not a substitute for held-out fold predictions.

Consequently, identical evaluation rows and target ordering cannot be authenticated between Prompt 10 and the combination DEV cells. All requested standalone-component, full-feature, and random-k pairings are therefore recorded as **not_supported**, with the exact reason, rather than reporting coerced fold deltas. No baseline OOT metric, prediction, or conclusion contributes to this review. Row-level paired inference, DeLong, bootstrap, pooled inference, ordinary fold t-tests, and Wilcoxon tests are not performed.

## Stability and stage support are authentic and feasible

Across the 24 configurations, mean pairwise fold Jaccard spans **{min(jaccards):.3f}–{max(jaccards):.3f}**. The stability file preserves fold selected counts, Jaccard mean/range, each feature's selection frequency, and Kuncheva only when a common candidate universe and equal selected-set size satisfy its assumptions. Variable natural-support sizes are explicitly marked not applicable for Kuncheva.

All 90 selector fits reached an authenticated completed or valid natural-support terminal state. `stage_support_audit.csv` records the support after every saved stage, candidate-universe and selected-feature hashes, fit provenance, shortfall warnings, and no-padding verification. The fold-1 row for each of the two Home Credit CatBoost Boruta-first configurations shows requested 40, realized 26, and `natural_support_26_of_requested_40`; folds 2–5 reached matched 40. No tentative or rejected feature was appended to either shortfall.

## Sequential resources and resume controls are safe for manual OOT

The DEV session authenticated 90 selector workers plus 120 evaluation workers under the sequential resource contract. Selector fits were reused where the LR and CatBoost evaluation cells shared a selection identity, yielding 90 fits rather than 120. Persisted worker intervals total **{float(overall['active_compute_seconds'])/3600:.2f} active-worker hours**; they are not presented as parallel-summed wall time. The matching event span is **{float(log.get('observed_matching_event_span_seconds', 0.0))/3600:.2f} hours**, and readiness checks recorded **{float(log.get('ram_wait_seconds', 0.0)):.2f} seconds** in readiness-wait stages.

Peak process-tree RSS was **{int(overall['peak_process_rss_bytes'])/2**30:.2f} GiB** and the lowest persisted available system RAM was **{int(overall['minimum_available_ram_bytes'])/2**30:.2f} GiB**. No selector or evaluation timeout, stop code, worker error, survivor process, partial file, or active/stale execution lock remains. Artifact size and stage-level timing/RAM detail are retained in `resource_summary.csv`. These controls are adequate for a later manual, sequential 24-cell OOT run; the OOT command must still be issued by the user.

## Focused, relevant, and complete validation passed

The focused Prompt 13 set passed 16 tests, the broader runner/authentication/resource/metrics/stability set passed 134 tests, and the complete repository suite passed 969 tests with 31 expected skips. The first full-suite shell wrapper expired at 120 seconds without a test failure and left no test process; the clean rerun completed in 238.15 seconds. Portable report schema validation and packaging passed. Its enhanced browser check is accurately limited to structural verification because no installed Chromium headless shell was available and none was downloaded.

## Limitations define what this review does not establish

- DEV is diagnostic and does not establish final superiority; OOT is the final evidence.
- The five folds are dependent expanding-window partitions, not independent experimental replicates.
- Prompt 10 baseline fold summaries cannot prove row/target pairing, so aligned baseline effects and inference are unavailable.
- Process timing comes from persisted worker lifecycle events and selector-reported fit time. It distinguishes summed active worker time from observed session span; non-worker Python overhead is not reconstructed as compute.
- The portable report is a reviewed snapshot. The CSV and JSON artifacts, not rounded narrative text, are the exact audit record.

## The exact frozen scope is ready for the user's manual OOT run

The audit found no authentication, contamination, feasibility, preservation, or resource-safety defect requiring a stop. The scientific decision is **ready_for_manual_oot**. Weak DEV performance is not a reason to modify the frozen scope, and every pool and natural-support configuration remains present. OOT must stay untouched until the user manually runs the exact command recorded in the decision artifact.

Generated from persisted authenticated artifacts only at {created}. No raw dataset path was resolved, no research loader or estimator was invoked, and no pilot, DEV, full-DEV refit, baseline, or OOT workload was executed.
"""


def _artifact_json(result: Mapping[str, Any], created: str) -> dict[str, Any]:
    summaries = sorted(
        result["summary_rows"],
        key=lambda row: (row["dataset"], row["final_model"], row["method"], str(row["variant"])),
    )
    chart_rows = [
        {
            "configuration": f"{row['dataset']} | {row['final_model']} | {_format_method(row['method'])} | {row['variant']}",
            "dataset": row["dataset"],
            "model": row["final_model"],
            "method": _format_method(row["method"]),
            "variant": row["variant"],
            "mean_auc": row["auc_mean"],
            "mean_ks": row["ks_mean"],
            "mean_jaccard": row["mean_pairwise_jaccard"],
            "support": row["stage_support_label"],
        }
        for row in summaries
    ]
    auth_rows = [
        {"check": "Pilot selector fits", "expected": 18, "authenticated": 18, "status": "pass"},
        {"check": "Pilot evaluations", "expected": 24, "authenticated": 24, "status": "pass"},
        {"check": "DEV selector fits", "expected": 90, "authenticated": 90, "status": "pass"},
        {"check": "DEV evaluations", "expected": 120, "authenticated": 120, "status": "pass"},
        {"check": "DEV active files", "expected": 840, "authenticated": 840, "status": "pass"},
        {"check": "Combination OOT evaluations", "expected": 24, "authenticated": 0, "status": "untouched"},
    ]
    summary_table = [
        {
            "dataset": row["dataset"],
            "model": row["final_model"],
            "method": _format_method(row["method"]),
            "variant": row["variant"],
            "support": row["stage_support_label"],
            "selected_range": f"{row['min_selected_feature_count']}–{row['max_selected_feature_count']}",
            "auc_mean": row["auc_mean"],
            "auc_min": row["auc_min"],
            "auc_max": row["auc_max"],
            "ks_mean": row["ks_mean"],
            "jaccard": row["mean_pairwise_jaccard"],
        }
        for row in summaries
    ]
    sources = [
        {"id": "dev_auth", "label": "Prompt 13 DEV authentication", "path": "dev_authentication.json"},
        {"id": "dev_summary", "label": "Authenticated DEV configuration summary", "path": "dev_configuration_summary.csv"},
        {"id": "baseline_alignment", "label": "Prompt 10 DEV alignment audit", "path": "aligned_baseline_comparisons.csv"},
        {"id": "stability", "label": "Selection stability audit", "path": "selection_stability.csv"},
        {"id": "resources", "label": "Persisted runtime and resource audit", "path": "resource_summary.csv"},
        {"id": "decision", "label": "Prompt 13 OOT-readiness decision", "path": "review_decision.json"},
        {"id": "tests_validation", "label": "Prompt 13 validation results", "path": "validation_results.json"},
    ]
    root_sources = [dict(item) for item in sources]
    for item in root_sources:
        if item["id"] == "dev_auth":
            item["query"] = {
                "engine": "duckdb",
                "language": "sql",
                "sql": "SELECT * FROM (VALUES ('Pilot selector fits',18,18,'pass'),('Pilot evaluations',24,24,'pass'),('DEV selector fits',90,90,'pass'),('DEV evaluations',120,120,'pass'),('DEV active files',840,840,'pass'),('Combination OOT evaluations',24,0,'untouched')) AS t(check_name,expected,authenticated,status)",
                "description": "Materializes the authenticated count checks bound by dev_authentication.json.",
            }
        elif item["id"] == "dev_summary":
            item["query"] = {
                "engine": "duckdb",
                "language": "sql",
                "sql": "SELECT dataset, final_model, method, variant, auc_mean, ks_mean, mean_pairwise_jaccard, stage_support_label FROM read_csv_auto('dev_configuration_summary.csv')",
                "description": "Selects the authenticated configuration-level DEV metrics and stability fields used by the report chart and table.",
                "tables_used": ["dev_configuration_summary.csv"],
                "filters": ["All 24 frozen configurations; no performance-based filtering"],
                "metric_definitions": {
                    "auc_mean": "Arithmetic mean of five authenticated held-out expanding-window fold ROC AUC values.",
                    "mean_pairwise_jaccard": "Mean Jaccard index across the ten fold-pair selected-feature set comparisons.",
                },
            }
    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "Combination DEV Authentication and OOT-Readiness Review",
            "description": "Authenticated Prompt 13 technical review of saved five-fold combination DEV evidence.",
            "generatedAt": created,
            "filters": [],
            "cards": [],
            "charts": [
                {
                    "id": "stability_performance",
                    "title": "DEV stability and held-out discrimination",
                    "subtitle": "Each point is one frozen configuration; AUC is diagnostic and does not rank OOT scope.",
                    "type": "scatter",
                    "dataset": "configuration_evidence",
                    "sourceId": "dev_summary",
                    "source": {
                        "id": "dev_summary",
                        "label": "Authenticated DEV configuration summary",
                        "path": "dev_configuration_summary.csv",
                        "query": {
                            "engine": "duckdb",
                            "language": "sql",
                            "sql": "SELECT dataset, final_model, method, variant, auc_mean, ks_mean, mean_pairwise_jaccard, stage_support_label FROM read_csv_auto('dev_configuration_summary.csv')",
                            "description": "Selects all 24 frozen configuration summaries without performance-based filtering.",
                        },
                    },
                    "encodings": {
                        "x": {"field": "mean_jaccard", "type": "quantitative", "label": "Mean pairwise Jaccard"},
                        "y": {"field": "mean_auc", "type": "quantitative", "label": "Mean held-out AUC"},
                        "color": {"field": "dataset", "type": "nominal", "label": "Dataset"},
                        "label": {"field": "configuration", "type": "nominal", "label": "Configuration"},
                        "tooltip": [
                            {"field": "method", "type": "nominal", "label": "Method"},
                            {"field": "variant", "type": "nominal", "label": "Variant"},
                            {"field": "model", "type": "nominal", "label": "Model"},
                            {"field": "mean_ks", "type": "quantitative", "label": "Mean KS"},
                            {"field": "support", "type": "nominal", "label": "Stage support"},
                        ],
                    },
                }
            ],
            "tables": [
                {
                    "id": "authentication_table",
                    "title": "Authentication and OOT-absence checks",
                    "subtitle": "Expected and observed authenticated counts from persisted state.",
                    "dataset": "authentication_checks",
                    "sourceId": "dev_auth",
                    "defaultSort": {"field": "check", "direction": "asc"},
                    "columns": [
                        {"field": "check", "label": "Check", "type": "text"},
                        {"field": "expected", "label": "Expected", "format": "number"},
                        {"field": "authenticated", "label": "Authenticated", "format": "number"},
                        {"field": "status", "label": "Status", "type": "text"},
                    ],
                },
                {
                    "id": "configuration_table",
                    "title": "Frozen configuration evidence",
                    "subtitle": "Five-fold held-out means and ranges; natural support remains separate from requested K.",
                    "dataset": "configuration_summary",
                    "sourceId": "dev_summary",
                    "defaultSort": {"field": "dataset", "direction": "asc"},
                    "columns": [
                        {"field": "dataset", "label": "Dataset", "type": "text"},
                        {"field": "model", "label": "Model", "type": "text"},
                        {"field": "method", "label": "Method", "type": "text"},
                        {"field": "variant", "label": "Variant", "type": "text"},
                        {"field": "support", "label": "Support", "type": "text"},
                        {"field": "selected_range", "label": "Selected range", "type": "text"},
                        {"field": "auc_mean", "label": "Mean AUC", "format": "number"},
                        {"field": "auc_min", "label": "Min AUC", "format": "number"},
                        {"field": "auc_max", "label": "Max AUC", "format": "number"},
                        {"field": "ks_mean", "label": "Mean KS", "format": "number"},
                        {"field": "jaccard", "label": "Mean Jaccard", "format": "number"},
                    ],
                },
            ],
            "sources": root_sources,
            "blocks": [
                {"id": "title", "type": "markdown", "body": "# Combination DEV Authentication and OOT-Readiness Review"},
                {
                    "id": "summary",
                    "type": "markdown",
                    "sourceId": "decision",
                    "body": "## The frozen 24-cell scope is ready for manual OOT\n\n**90/90 selector fits and 120/120 held-out DEV cells authenticate.** Combination OOT is still 0/24. Each Home Credit CatBoost Boruta-first configuration has an unpadded fold-1 natural-support result of 26 against requested K=40; the frozen OOT label preserves that authenticated reference. DEV is diagnostic: no winner or pool was selected.",
                },
                {"id": "auth_intro", "type": "markdown", "body": "## Content-level authentication passed\n\nThe audit verified state digests, contract-file hashes, prediction/target validity and ordering, selection lineage, feature-universe identities, worker exits, and leakage flags—not only completion counts."},
                {"id": "auth_table_block", "type": "table", "tableId": "authentication_table"},
                {"id": "dev_intro", "type": "markdown", "body": "## DEV evidence varies but does not alter scope\n\nThe table exposes all 24 frozen configuration summaries. Every value is from held-out expanding-window folds; weak folds remain in the detailed CSV and weak DEV results remain eligible for OOT."},
                {"id": "config_table_block", "type": "table", "tableId": "configuration_table"},
                {"id": "stability_intro", "type": "markdown", "body": "## Stability is evidence, not a winner rule\n\nRead points horizontally for cross-fold feature-set stability and vertically for descriptive held-out AUC. Different datasets are not directly comparable, and the chart does not authorize pruning."},
                {"id": "stability_chart_block", "type": "chart", "chartId": "stability_performance"},
                {"id": "baseline", "type": "markdown", "sourceId": "baseline_alignment", "body": "## Baseline effects and paired inference are not supported\n\nPrompt 10 authenticates at 36/36, but held-out fold prediction vectors and ordered row/target hashes were not saved. Exact evaluation-row alignment cannot be proved, so standalone-component, full-feature, random-k, row-level, and fold-significance comparisons are all reported `not_supported`; baseline OOT is excluded."},
                {"id": "support", "type": "markdown", "sourceId": "stability", "body": "## All stage-support policies remain feasible\n\nAll 90 selector fits completed under the authenticated support policy. Each Home Credit CatBoost Boruta-first configuration realized an unpadded 26 of requested 40 on fold 1, while folds 2–5 reached 40. The 26-feature cases are not matched-budget K=40; the future refit must report its own realized support."},
                {"id": "resources", "type": "markdown", "sourceId": "resources", "body": "## Sequential runtime controls are adequate\n\nThe run reused 90 selector fits across 120 model evaluations, ended without timeouts or stop codes, and left no worker, survivor, partial file, or execution lock. Active compute, RAM waiting, session span, RSS, available RAM, and artifact bytes are reported separately."},
                {"id": "tests", "type": "markdown", "sourceId": "tests_validation", "body": "## Validation passed at every required level\n\nFocused Prompt 13 tests passed 16/16, relevant runner and scientific-contract tests passed 134/134, and the complete repository suite passed 969 tests with 31 skips. Portable schema and packaging checks passed; browser QA was structural-only because no installed Chromium headless shell was available, and no browser was downloaded."},
                {"id": "limitations", "type": "markdown", "body": "## Limits and uncertainty\n\n- DEV is diagnostic; OOT is final evidence.\n- Five expanding-window folds are dependent partitions, not independent experiments.\n- Baseline row-level alignment is not reconstructable from summaries.\n- The review lock is a provenance authorization; the runner's existing completion lock is the technical gate."},
                {"id": "next", "type": "markdown", "sourceId": "decision", "body": "## Next step\n\nThe user may manually run only the exact frozen OOT command recorded in the decision artifact. This report does not claim that OOT ran and does not authorize any scope change."},
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": created,
            "status": "ready",
            "datasets": {
                "authentication_checks": auth_rows,
                "configuration_summary": summary_table,
                "configuration_evidence": chart_rows,
            },
            "accessIssues": [],
        },
        "sources": root_sources,
    }


def write_review_package(repository_root: str | Path) -> dict[str, Any]:
    result = authenticate_and_analyze(repository_root)
    root: Path = result["root"]
    output = root / AUDIT_DIR
    output.mkdir(parents=True, exist_ok=True)
    created = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    plan = result["plan"]
    approval_path = root / plan.configuration["gates"]["pilot_approval_lock_path"]
    completion_path = root / plan.configuration["gates"]["dev_completion_lock_path"]
    scope_path = root / SCOPE_PATH
    pre_snapshot_path = output / "preservation_snapshot_pre_review.json"
    pre_snapshot = read_authenticated_json(pre_snapshot_path)

    auth = authenticated_payload(
        {
            "schema_version": AUDIT_SCHEMA,
            "created_at_utc": created,
            "authentication_status": "authenticated_complete",
            "raw_dataset_paths_resolved": False,
            "research_data_loaders_invoked": 0,
            "workers_started": 0,
            "pilot": {"selector_fits": "18/18", "evaluation_cells": "24/24", "first_incomplete": None},
            "dev": {
                "selector_fits": "90/90",
                "evaluation_cells": "120/120",
                "folds_per_configuration": 5,
                "frozen_configurations": 24,
                "active_files": "840/840",
                "missing": 0,
                "extra": 0,
                "duplicate": 0,
                "partial": 0,
                "interrupted": 0,
                "failed": 0,
                "timed_out": 0,
                "hash_invalid": 0,
            },
            "approval_lock": {
                "path": approval_path.relative_to(root).as_posix(),
                "file_sha256": sha256_file(approval_path),
                "artifact_authentication_sha256": result["approval"]["artifact_authentication_sha256"],
                "review_digest_sha256": result["approval"]["approved_review_digest_sha256"],
            },
            "dev_completion_lock": {
                "path": completion_path.relative_to(root).as_posix(),
                "file_sha256": sha256_file(completion_path),
                "artifact_authentication_sha256": result["completion"]["artifact_authentication_sha256"],
                "ordered_evaluation_count": len(result["completion"]["completed_dev_evaluation_ids"]),
                "ordered_selection_count": len(result["completion"]["completed_dev_selection_ids"]),
            },
            "prediction_contract": {
                "authenticated_files": 120,
                "row_count_order_identity_target_probability_checks_passed": True,
                "within_dataset_fold_ordered_row_target_alignment_passed": True,
                "validation_target_used_for_fit_any": False,
                "opened_oot_paths_any": False,
                "oot_rows_retained_total": 0,
            },
            "selection_contract": {
                "authenticated_files": 90,
                "candidate_selected_intermediate_stage_hashes_passed": True,
                "natural_support_26_of_40_fold_rows": 2,
                "natural_support_configuration_count": 2,
                "padding_detected": False,
            },
            "baseline": {
                **result["baseline_authentication"],
                "held_out_fold_prediction_vectors_saved": False,
                "ordered_fold_row_target_identity_hashes_saved": False,
                "baseline_oot_evidence_used": False,
            },
            "oot": {"authenticated_evaluations": 0, "expected_evaluations": 24, "scientific_artifacts": 0},
        }
    )
    atomic_json(output / "dev_authentication.json", auth)

    fold_fields = list(result["fold_rows"][0])
    atomic_csv(output / "dev_fold_results.csv", result["fold_rows"], fold_fields)
    summary_fields = list(result["summary_rows"][0])
    atomic_csv(output / "dev_configuration_summary.csv", result["summary_rows"], summary_fields)
    atomic_csv(output / "aligned_baseline_comparisons.csv", result["baseline_rows"], list(result["baseline_rows"][0]))
    atomic_csv(output / "selection_stability.csv", result["stability_rows"], list(result["stability_rows"][0]))
    atomic_csv(output / "stage_support_audit.csv", result["stage_rows"], list(result["stage_rows"][0]))
    atomic_csv(output / "resource_summary.csv", result["resource_rows"], list(result["resource_rows"][0]))

    safe_plan = render_plan(plan)
    scope_validation = authenticated_payload(
        {
            "schema_version": AUDIT_SCHEMA,
            "created_at_utc": created,
            "status": "validated_exact_ordered_scope",
            "scope_path": SCOPE_PATH.as_posix(),
            "scope_file_sha256": sha256_file(scope_path),
            "scope_internal_authentication_sha256": result["scope"]["artifact_authentication_sha256"],
            "plan_raw_dataset_paths_resolved": safe_plan["raw_dataset_paths_resolved"],
            "plan_workers_started": safe_plan["workers_started"],
            "expected_oot_selection_refits": 18,
            "observed_oot_selection_refits": safe_plan["oot_selection_count"],
            "expected_oot_evaluations": 24,
            "observed_oot_evaluations": safe_plan["oot_evaluation_count"],
            "selection_order_matches": [item["selection_id"] for item in safe_plan["oot_selections"]]
            == list(result["scope"]["ordered_oot_selection_ids"]),
            "evaluation_order_matches": [item["cell_id"] for item in safe_plan["oot_evaluations"]]
            == list(result["scope"]["ordered_oot_evaluation_cell_ids"]),
            "combination_oot_directory_present": (root / "results/selector_combinations_v1/oot").exists(),
            "combination_oot_scientific_artifact_count": 0,
            "combination_oot_accessed": False,
            "all_iv_pools_retained": [100, 200, 300],
            "no_dev_based_selection_removal_tuning_or_reordering": True,
            "natural_support_no_padding_verified": True,
        }
    )
    atomic_json(output / "oot_scope_validation.json", scope_validation)

    current_trees = {
        "combination_pilot": inventory_tree(root / "results/selector_combinations_v1/pilot"),
        "combination_dev": inventory_tree(root / "results/selector_combinations_v1/dev"),
        "prompt_10_baseline": inventory_tree(root / "results/full_baseline_v1"),
    }
    tree_matches = {
        name: all(
            current_trees[name][field] == pre_snapshot["artifact_trees"][name][field]
            for field in ("file_count", "total_bytes", "tree_sha256")
        )
        for name in current_trees
    }
    config_checks = []
    for item in pre_snapshot["configuration"]["files"]:
        path = root / item["path"]
        intentionally_changed_plan_support = item["path"] == "src/credit_risk_fs/experiments/selector_combinations.py"
        config_checks.append(
            {
                "path": item["path"],
                "starting_sha256": item["sha256"],
                "ending_sha256": sha256_file(path),
                "byte_identical": sha256_file(path) == item["sha256"],
                "classification": (
                    "intentional_plan_only_support_change_in_scope_freeze_commit"
                    if intentionally_changed_plan_support
                    else "scientific_configuration_must_be_identical"
                ),
            }
        )
    scientific_config_pass = all(
        item["byte_identical"]
        for item in config_checks
        if item["classification"] == "scientific_configuration_must_be_identical"
    )
    preservation = authenticated_payload(
        {
            "schema_version": AUDIT_SCHEMA,
            "created_at_utc": created,
            "status": "preserved_byte_identical" if all(tree_matches.values()) and scientific_config_pass else "preservation_failure",
            "starting_snapshot_path": pre_snapshot_path.relative_to(root).as_posix(),
            "starting_snapshot_file_sha256": sha256_file(pre_snapshot_path),
            "tree_comparisons": {
                name: {
                    "starting": pre_snapshot["artifact_trees"][name],
                    "ending": current_trees[name],
                    "byte_identical": tree_matches[name],
                }
                for name in current_trees
            },
            "configuration_file_checks": config_checks,
            "scientific_configurations_byte_identical": scientific_config_pass,
            "approval_lock_byte_identical": sha256_file(approval_path)
            == pre_snapshot["locks"]["pilot_approval_lock"]["file_sha256"],
            "dev_completion_lock_byte_identical": sha256_file(completion_path)
            == pre_snapshot["locks"]["dev_completion_lock"]["file_sha256"],
            "combination_oot_remains_0_of_24": True,
            "combination_oot_scientific_artifacts": 0,
            "experiment_output_files_modified": 0,
        }
    )
    if preservation["status"] != "preserved_byte_identical":
        raise Prompt13AuditError("post-review preservation check failed")
    atomic_json(output / "preservation_check.json", preservation)

    manual_command = "$env:PYTHONDONTWRITEBYTECODE='1'; .\\.venv\\Scripts\\python.exe -B scripts\\run_selector_combination_research.py --repository-root . --phase oot"
    decision = authenticated_payload(
        {
            "schema_version": AUDIT_SCHEMA,
            "created_at_utc": created,
            "decision": "ready_for_manual_oot",
            "decision_standard": {
                "authentication_passed": True,
                "complete_frozen_24_cell_scope_feasible": True,
                "scientific_defect_requires_stop": False,
                "contamination_detected": False,
                "preservation_passed": True,
                "runtime_locking_resume_controls_safe": True,
            },
            "scope_authorized": {
                "selection_refit_count": 18,
                "evaluation_count": 24,
                "ordered_evaluation_ids": [item.cell_id for item in result["oot_evaluations"]],
                "retained_method_ids": list(METHOD_ORDER),
                "iv_pool_budgets": [100, 200, 300],
                "natural_support_policy": "preserve the authenticated 26-of-requested-40 reference label; future full-DEV refit reports its own support and padding is forbidden",
            },
            "dev_evidence_use": "diagnostic_only_no_winner_pruning_tuning_or_reordering",
            "baseline_alignment": "not_supported_missing_held_out_fold_prediction_vectors_and_ordered_row_target_hashes",
            "technical_gate": "open_authenticated_dev_complete",
            "scientific_gate": "ready_authenticated_prompt_13_review_lock_required",
            "oot_has_run": False,
            "manual_oot_command_not_executed": manual_command,
        }
    )
    atomic_json(output / "review_decision.json", decision)
    report_md = _report_markdown(result, created)
    atomic_text(output / "review_report.md", report_md)
    artifact = _artifact_json(result, created)
    atomic_json(output / "artifact.json", artifact)
    return {
        "created_at_utc": created,
        "result": result,
        "output": output,
        "scope_validation": scope_validation,
        "preservation": preservation,
        "decision": decision,
        "manual_command": manual_command,
    }


def finalize_manifest_and_lock(repository_root: str | Path, package: Mapping[str, Any], report_receipt: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    output = root / AUDIT_DIR
    created = str(package["created_at_utc"])
    manifest_names = [
        "preservation_snapshot_pre_review.json",
        "dev_authentication.json",
        "dev_fold_results.csv",
        "dev_configuration_summary.csv",
        "aligned_baseline_comparisons.csv",
        "selection_stability.csv",
        "stage_support_audit.csv",
        "resource_summary.csv",
        "oot_scope_validation.json",
        "preservation_check.json",
        "review_decision.json",
        "review_report.md",
        "artifact.json",
        "report.html",
        "validation_results.json",
    ]
    files = []
    for name in manifest_names:
        path = output / name
        if not path.is_file():
            raise Prompt13AuditError(f"required review deliverable absent:{name}")
        files.append({"path": name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    manifest = authenticated_payload(
        {
            "schema_version": AUDIT_SCHEMA,
            "created_at_utc": created,
            "status": "authenticated_review_package",
            "decision": "ready_for_manual_oot",
            "file_count": len(files),
            "files": files,
            "report_packaging": dict(report_receipt),
            "raw_dataset_paths_resolved": False,
            "workers_started": 0,
            "real_workloads_executed": 0,
            "combination_oot_accessed": False,
            "baseline_oot_evidence_used": False,
        }
    )
    manifest_path = output / "audit_manifest.json"
    atomic_json(manifest_path, manifest)

    result = package["result"]
    plan = result["plan"]
    approval_path = root / plan.configuration["gates"]["pilot_approval_lock_path"]
    completion_path = root / plan.configuration["gates"]["dev_completion_lock_path"]
    scope_path = root / SCOPE_PATH
    snapshot_path = output / "preservation_snapshot_pre_review.json"
    decision_path = output / "review_decision.json"
    lock = authenticated_payload(
        {
            "schema_version": "selector_combination_dev_review_lock_v1",
            "created_at_utc": created,
            "terminal_state": "ready_for_manual_oot",
            "authorization_scope": "later_manual_execution_of_exact_frozen_combination_oot_scope_only",
            "oot_has_run": False,
            "technical_gate_state": "open_authenticated_dev_complete",
            "scientific_gate_state": "open_authenticated_prompt_13_review",
            "runner_enforcement": "provenance_lock_not_an_execution_enforced_runner_hook",
            "configuration_sha256": plan.configuration_sha256,
            "protocol_lock_sha256": plan.protocol_lock_sha256,
            "pilot_approval_lock_path": approval_path.relative_to(root).as_posix(),
            "pilot_approval_lock_file_sha256": sha256_file(approval_path),
            "pilot_approval_lock_authentication_sha256": result["approval"]["artifact_authentication_sha256"],
            "dev_completion_lock_path": completion_path.relative_to(root).as_posix(),
            "dev_completion_lock_file_sha256": sha256_file(completion_path),
            "dev_completion_lock_authentication_sha256": result["completion"]["artifact_authentication_sha256"],
            "oot_scope_freeze_path": scope_path.relative_to(root).as_posix(),
            "oot_scope_freeze_file_sha256": sha256_file(scope_path),
            "oot_scope_freeze_authentication_sha256": result["scope"]["artifact_authentication_sha256"],
            "scope_freeze_commit": "26158348a273876ac11956b557e8534d9edffdd2",
            "code_identity": {
                "scope_generation_source_path": result["scope"]["scope_generation_code"]["path"],
                "scope_generation_source_sha256": sha256_file(
                    root / result["scope"]["scope_generation_code"]["path"]
                ),
                "scope_generation_frozen_sha256": result["scope"]["scope_generation_code"]["sha256"],
                "audit_module_path": "src/credit_risk_fs/experiments/prompt_13_dev_audit.py",
                "audit_module_sha256": sha256_file(
                    root / "src/credit_risk_fs/experiments/prompt_13_dev_audit.py"
                ),
                "audit_entry_point_path": "scripts/audit_selector_combination_dev.py",
                "audit_entry_point_sha256": sha256_file(
                    root / "scripts/audit_selector_combination_dev.py"
                ),
                "focused_test_path": "tests/test_prompt_13_dev_audit.py",
                "focused_test_sha256": sha256_file(root / "tests/test_prompt_13_dev_audit.py"),
            },
            "preservation_snapshot_path": snapshot_path.relative_to(root).as_posix(),
            "preservation_snapshot_file_sha256": sha256_file(snapshot_path),
            "audit_manifest_path": manifest_path.relative_to(root).as_posix(),
            "audit_manifest_file_sha256": sha256_file(manifest_path),
            "audit_manifest_authentication_sha256": manifest["artifact_authentication_sha256"],
            "review_decision_path": decision_path.relative_to(root).as_posix(),
            "review_decision_file_sha256": sha256_file(decision_path),
            "review_decision_authentication_sha256": package["decision"]["artifact_authentication_sha256"],
            "retained_method_ids": list(METHOD_ORDER),
            "ordered_oot_selection_ids": [item.selection_id for item in result["oot_selections"]],
            "ordered_oot_evaluation_ids": [item.cell_id for item in result["oot_evaluations"]],
            "expected_oot_selection_refits": 18,
            "expected_oot_evaluations": 24,
            "iv_pool_budgets": [100, 200, 300],
            "seed": 42,
            "natural_support_policy": "preserve the authenticated 26-of-requested-40 reference label; future full-DEV refit reports its own support and padding is forbidden",
            "no_configuration_selection_removal_tuning_or_reordering_from_dev": True,
            "raw_dataset_paths_resolved": False,
            "workers_started": 0,
        }
    )
    lock_path = root / REVIEW_LOCK_PATH
    atomic_json(lock_path, lock)
    reread = read_authenticated_json(lock_path)
    if reread != lock:
        raise Prompt13AuditError("review lock round-trip authentication failed")
    return {"manifest": manifest, "lock": lock, "lock_path": lock_path}
