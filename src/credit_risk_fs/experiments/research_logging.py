"""Process-safe human run logs with a separate machine-readable audit stream."""

from __future__ import annotations

import json
import logging
import multiprocessing
import os
import queue
import re
import sys
import threading
import traceback as traceback_module
import uuid
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any, Iterator, Mapping, TextIO


DEFAULT_RESEARCH_LOG = Path("logs/runs.log")
DEFAULT_AUDIT_LOG_NAME = "events.jsonl"
DEFAULT_DEBUG_LOG_NAME = "debug.log"
LOG_SCHEMA_VERSION = "research_run_log_v1"
DEFAULT_QUEUE_CAPACITY = 1024
PRIORITY_QUEUE_TIMEOUT_SECONDS = 0.25
LISTENER_SHUTDOWN_TIMEOUT_SECONDS = 2.0
STAGE_HEARTBEAT_INTERVAL_SECONDS = 30.0
_MAX_STRING_LENGTH = 16_384
_MAX_SEQUENCE_ITEMS = 64
_MAX_MAPPING_ITEMS = 64
_GIB = 1024**3
_NO_DEBUG_TRACEBACK_EXCEPTIONS = {"KeyboardInterrupt", "ManualResearchStop"}

_STAGE_LABELS = {
    "plan_construction": "Research plan",
    "release_authentication": "Release authentication",
    "workflow_preflight": "Repository preflight",
    "inter_run_readiness": "Resource readiness check",
    "dev_global_barrier": "DEV validation",
    "workflow_finalization": "Research finalization",
    "dev_data_loading": "DEV data loading",
    "target_extraction": "Target validation",
    "feature_filtering_sanitization": "Feature filtering",
    "row_boundary_selection": "Fold boundary selection",
    "selection_encoding": "Selection encoding",
    "voter_rf_corr_mrmr": "RF relevance + mRMR",
    "reference_rf_corr_mrmr": "RF relevance + mRMR",
    "voter_boruta": "Boruta",
    "rank_aggregation": "Rank voting",
    "rfe_encoding": "RFE encoding",
    "rfe": "RFE",
    "selected_projection_reload": "Selected-feature loading",
    "final_preprocessing": "Model preprocessing",
    "final_model_fit": "Model fit",
    "final_prediction": "DEV prediction",
    "fold_artifact_writing": "Fold artifact writing",
    "fold_checkpoint_finalization": "Fold checkpoint",
    "dev_oof_aggregation": "DEV prediction aggregation",
    "dev_evaluation": "DEV evaluation",
    "dev_artifact_writing": "DEV artifact writing",
    "dev_checkpoint_finalization": "DEV checkpoint",
    "full_dev_data_loading": "Full-DEV data loading",
    "full_dev_target_extraction": "Full-DEV target validation",
    "full_dev_feature_filtering_sanitization": "Full-DEV feature filtering",
    "full_dev_selected_projection_reload": "Full-DEV selected-feature loading",
    "full_dev_selected_feature_validation": "Selected-feature validation",
    "locked_oot_data_loading": "Locked OOT data loading",
    "oot_target_extraction": "OOT target validation",
    "oot_feature_filtering_sanitization": "OOT feature filtering",
    "full_dev_preprocessing": "Full-DEV preprocessing",
    "full_dev_model_fit": "Full-DEV model fit",
    "full_dev_prediction": "Full-DEV prediction",
    "oot_prediction": "OOT prediction",
    "oot_artifact_writing": "OOT artifact writing",
    "oot_evaluation": "OOT evaluation",
    "oot_checkpoint_finalization": "OOT checkpoint",
    "checkpoint_resume_validation": "Checkpoint validation",
    "data_loading": "Data loading",
    "pilot_dev_data_loading": "Pilot DEV data loading",
    "pilot_fold_projection": "Pilot fold-1 projection",
    "pilot_selection_encoding": "Pilot selection encoding",
    "pilot_catboost_shap": "CatBoost-SHAP pilot",
    "pilot_boruta_random_forest": "Boruta pilot",
    "pilot_rfe_catboost": "CatBoost RFE pilot",
}

_HUMAN_SILENT_EVENTS = {
    "logging_shutdown",
    "worker_logging_configured",
    "worker_spawn_requested",
    "worker_spawned",
    "worker_started",
    "worker_completed",
    "worker_failed",
    "worker_interrupted",
    # The session/run terminal event renders the single concise failure or
    # manual-interrupt line after cleanup and checkpoint handling completes.
    "stage_failed",
    "stage_interrupted",
    "component_started",
    "component_completed",
    "resource_artifact_written",
    "execution_lock_created",
    "execution_lock_released",
}


_CONTEXT: ContextVar[dict[str, Any]] = ContextVar("research_log_context", default={})
_ACTIVE_EMITTER: Any | None = None
_ACTIVE_SESSION: "ResearchLogSession | None" = None


def _timestamp_utc() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _safe_value(value: Any, *, depth: int = 0) -> Any:
    """Keep log fields scalar and bounded; never retain scientific payloads."""

    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, Path):
        value = value.as_posix()
    if isinstance(value, str):
        if len(value) <= _MAX_STRING_LENGTH:
            return value
        return value[:_MAX_STRING_LENGTH] + "...[truncated]"
    if depth >= 3:
        return f"<{type(value).__name__} omitted>"
    if isinstance(value, Mapping):
        items = list(value.items())[:_MAX_MAPPING_ITEMS]
        result = {
            str(key): _safe_value(item, depth=depth + 1) for key, item in items
        }
        if len(value) > _MAX_MAPPING_ITEMS:
            result["_truncated_items"] = len(value) - _MAX_MAPPING_ITEMS
        return result
    if isinstance(value, (list, tuple, set, frozenset)):
        values = list(value)
        result = [
            _safe_value(item, depth=depth + 1)
            for item in values[:_MAX_SEQUENCE_ITEMS]
        ]
        if len(values) > _MAX_SEQUENCE_ITEMS:
            result.append(f"<{len(values) - _MAX_SEQUENCE_ITEMS} items omitted>")
        return result
    return f"<{type(value).__name__} omitted>"


def _record(
    event: str,
    *,
    level: str,
    message: str,
    session_id: str,
    fields: Mapping[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": LOG_SCHEMA_VERSION,
        "timestamp_utc": _timestamp_utc(),
        "level": str(level).upper(),
        "pid": os.getpid(),
        "session_id": str(session_id),
        "event": str(event),
        "message": _safe_value(str(message)),
    }
    for key, value in fields.items():
        if key in payload:
            continue
        payload[str(key)] = _safe_value(value)
    return payload


def _human_timestamp(value: Any) -> str:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        parsed = parsed.astimezone(timezone.utc)
    except (TypeError, ValueError):
        parsed = datetime.now(timezone.utc)
    return parsed.strftime("%Y-%m-%d %H:%M:%S UTC")


def _clean_message(value: Any) -> str:
    message = " ".join(str(value or "").split())
    message = re.sub(r"^CONTROLLED_STOP\s+[A-Z0-9_]+:\s*", "", message)
    return message[:500]


def _run_number(run_id: Any) -> str | None:
    match = re.search(r"(?:^|-)cdv1-(\d{3})(?:-|$)", f"-{run_id}")
    return match.group(1) if match else None


def _dataset_label(dataset: Any, run_id: Any) -> str | None:
    value = str(dataset or "").lower()
    combined = f"{value} {run_id}".lower()
    if "lendingclub" in combined:
        return "LendingClub"
    if "homecredit" in combined:
        return "Home Credit"
    return str(dataset).replace("_", " ").title() if dataset else None


def _model_label(model: Any, run_id: Any) -> str | None:
    value = str(model or "").lower()
    combined = f"{value} {run_id}".lower()
    if "catboost" in combined:
        return "CatBoost"
    if value == "lr" or re.search(r"-lr-s\d+$", combined):
        return "Logistic Regression"
    return str(model).replace("_", " ").title() if model else None


def _method_label(payload: Mapping[str, Any]) -> str | None:
    run_id = str(payload.get("run_id") or "")
    match = re.search(r"-voting-k(\d+)-", run_id)
    if match:
        return f"Voting K={match.group(1)}"
    if "-reference-rf-corr-mrmr-" in run_id:
        return "Reference RF+Corr+mRMR"
    selector = str(payload.get("selector") or "")
    if selector == "rank_voting_v1":
        return "Rank voting"
    if selector:
        return selector.replace("_", " ").title()
    return None


def _fold_label(payload: Mapping[str, Any]) -> str | None:
    fold = payload.get("fold_id")
    try:
        value = int(fold)
    except (TypeError, ValueError):
        return None
    return f"Fold {value}/5" if 1 <= value <= 5 else None


def _stage_label(payload: Mapping[str, Any]) -> str:
    stage = str(payload.get("stage") or "")
    if stage in {"final_model_fit", "full_dev_model_fit"}:
        model = _model_label(payload.get("model"), payload.get("run_id"))
        if model:
            return f"{model} model fit"
    if stage in _STAGE_LABELS:
        return _STAGE_LABELS[stage]
    component = str(payload.get("component") or "")
    if component and component not in {"model_fit", "checkpoint_manager"}:
        return component.replace("_", " ").title()
    return stage.replace("_", " ").title() if stage else "Research stage"


def _duration(value: Any) -> str | None:
    try:
        seconds = max(0, int(round(float(value))))
    except (TypeError, ValueError):
        return None
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def _gib(value: Any) -> str | None:
    try:
        return f"{float(value) / _GIB:.1f} GiB"
    except (TypeError, ValueError):
        return None


def _stop_reason(code: Any, message: Any = "") -> str:
    normalized = str(code or "").lower()
    if normalized in {"ram_system_headroom", "ram_process_limit"}:
        return "RAM safety limit reached"
    if normalized == "gpu_process_limit":
        return "GPU memory safety limit reached"
    if normalized in {"disk_results_limit", "disk_temp_limit"}:
        return "disk safety limit reached"
    if normalized == "manual_interrupt":
        return "interrupted manually"
    if normalized == "wall_clock_limit":
        return "wall-clock safety limit reached"
    cleaned = _clean_message(message)
    return cleaned or normalized.replace("_", " ") or "controlled stop"


def _human_line(
    payload: Mapping[str, Any], *, debug_log_path: str
) -> str | None:
    event = str(payload.get("event") or "")
    level = str(payload.get("level") or "INFO").upper()
    if event in _HUMAN_SILENT_EVENTS:
        return None
    if event == "python_log" and level not in {"WARNING", "ERROR", "CRITICAL"}:
        return None

    action = "INFO"
    parts: list[str] = []
    run_number = _run_number(payload.get("run_id"))
    run_label = f"Run {run_number}" if run_number else None
    pilot_cell = str(payload.get("pilot_cell") or "")
    pilot_label = f"Pilot {pilot_cell[:2]}" if pilot_cell[:2].isdigit() else None
    fold_label = _fold_label(payload)

    if event == "logging_initialized":
        parts = ["Logging ready", str(payload.get("log_path") or "logs/runs.log")]
    elif event == "session_started":
        action, parts = "START", ["Research session started"]
    elif event == "configuration_authenticated":
        parts = ["Research plan and configuration hashes authenticated"]
    elif event == "release_authenticated":
        commit = str(payload.get("git_commit") or "")[:8]
        parts = [f"Release {commit} authenticated" if commit else "Release authenticated"]
    elif event == "plan_resume_decision":
        parts = [_clean_message(payload.get("message"))]
    elif event == "run_execution_started":
        action = "START"
        parts = [
            item
            for item in (
                run_label,
                pilot_label,
                _dataset_label(payload.get("dataset"), payload.get("run_id")),
                _method_label(payload),
                _model_label(payload.get("model"), payload.get("run_id")),
            )
            if item
        ]
    elif event in {"run_resume_decision", "run_resume_authenticated"}:
        parts = [
            item
            for item in (run_label, pilot_label, _clean_message(payload.get("message")))
            if item
        ]
    elif event == "stage_started":
        parts = [item for item in (fold_label, f"{_stage_label(payload)} started") if item]
    elif event == "stage_heartbeat":
        action = "ACTIVE"
        parts = [item for item in (fold_label, f"{_stage_label(payload)} running") if item]
        elapsed = _duration(
            payload.get("elapsed_stage_seconds", payload.get("elapsed_supervisor_seconds"))
        )
        ram = _gib(
            payload.get(
                "worker_rss_bytes",
                payload.get("process_tree_rss_bytes", payload.get("parent_rss_bytes")),
            )
        )
        available = _gib(payload.get("system_available_ram_bytes"))
        if elapsed:
            parts.append(f"Elapsed {elapsed}")
        if ram:
            parts.append(f"RAM {ram}")
        if available:
            parts.append(f"Available {available}")
    elif event in {"ram_wait_started", "ram_wait_periodic"}:
        action = "WAIT"
        dataset = _dataset_label(payload.get("dataset"), payload.get("run_id"))
        stage = str(payload.get("stage") or "")
        if stage in {
            "initialized",
            "data_loading",
            "dev_data_loading",
            "pilot_dev_data_loading",
            "full_dev_data_loading",
            "locked_oot_data_loading",
        }:
            activity = f"{dataset or 'Dataset'} loading paused"
        else:
            activity = f"{dataset + ' ' if dataset else ''}{_stage_label(payload)} paused"
        parts = [activity]
        if event == "ram_wait_periodic":
            try:
                waiting_seconds = float(payload.get("waiting_seconds"))
            except (TypeError, ValueError):
                waiting_seconds = -1
            waiting = (
                f"{int(waiting_seconds // 60)}m"
                if waiting_seconds >= 60 and waiting_seconds % 60 < 1e-9
                else _duration(payload.get("waiting_seconds"))
            )
            if waiting:
                parts.append(f"Waiting {waiting}")
        ram = _gib(payload.get("process_tree_rss_bytes"))
        available = _gib(payload.get("system_available_ram_bytes"))
        if ram:
            parts.append(f"Process RAM {ram}")
        if available:
            parts.append(f"Available {available}")
    elif event == "ram_resumed":
        action = "RESUME"
        available = _gib(payload.get("system_available_ram_bytes"))
        parts = [
            f"Available RAM stable at {available}"
            if available
            else "Available RAM recovered stably"
        ]
        dataset = _dataset_label(payload.get("dataset"), payload.get("run_id"))
        stage = str(payload.get("stage") or "")
        if stage in {
            "initialized",
            "data_loading",
            "dev_data_loading",
            "pilot_dev_data_loading",
            "full_dev_data_loading",
            "locked_oot_data_loading",
        }:
            parts.append(f"Continuing {dataset or 'dataset'} loading")
        else:
            parts.append(
                f"Continuing {dataset + ' ' if dataset else ''}{_stage_label(payload)}"
            )
    elif event == "stage_completed":
        action = "DONE"
        parts = [item for item in (fold_label, f"{_stage_label(payload)} completed") if item]
        elapsed = _duration(payload.get("elapsed_stage_seconds"))
        if elapsed:
            parts[-1] += f" in {elapsed}"
    elif event == "stage_aborted":
        action = "STOP"
        reason = _stop_reason(payload.get("stop_code"), payload.get("message"))
        parts = [item for item in (fold_label, f"{_stage_label(payload)} stopped: {reason}") if item]
    elif event == "resource_warning":
        action = "WARN"
        parts = [_stop_reason(payload.get("warning_code"), payload.get("message"))]
    elif event == "supervisor_lifecycle":
        state = str(payload.get("lifecycle_state") or "")
        if state == "RESOURCE_STOP_LATCHED":
            action, parts = "STOP", [_stop_reason(payload.get("stop_code"), payload.get("message"))]
        elif state == "WALL_CLOCK_STOP_LATCHED":
            action, parts = "STOP", [_stop_reason(payload.get("stop_code"), payload.get("message"))]
        elif state == "COOPERATIVE_STOP_REQUESTED":
            action, parts = "INFO", ["Graceful worker stop requested"]
        elif state == "GRACE_PERIOD":
            action, parts = "INFO", ["Waiting for graceful worker exit"]
        elif state == "TERMINATE_PROCESS_TREE":
            action, parts = "WARN", ["Terminating the research worker"]
        elif state == "FORCE_KILL_REMAINDERS":
            action, parts = "WARN", ["Force-stopping remaining worker processes"]
        elif state == "EXIT_CONFIRMED":
            action, parts = "DONE", ["Research worker cleanup confirmed"]
        elif state == "WORKER_TREE_TERMINATION_FAILED":
            action, parts = "ERROR", ["Research worker cleanup could not be confirmed"]
        else:
            return None
    elif event == "checkpoint_transition":
        parts = [item for item in (run_label, f"Checkpoint saved after {_stage_label(payload)}") if item]
    elif event == "checkpoint_validation_started":
        parts = [item for item in (run_label, "Checkpoint validation started") if item]
    elif event == "checkpoint_validation_completed":
        action, parts = "DONE", [item for item in (run_label, "Checkpoint validation completed") if item]
    elif event == "run_finalized":
        status = str(payload.get("status") or "")
        if status in {"completed", "dev_complete"}:
            action, parts = "DONE", [
                item
                for item in (
                    run_label,
                    pilot_label,
                    f"Run {status.replace('_', ' ')}",
                )
                if item
            ]
        elif status in {
            "aborted_resource_limit",
            "resource_aborted",
            "timed_out",
            "manually_interrupted",
            "interrupted",
        }:
            action, parts = "STOP", [
                item
                for item in (
                    run_label,
                    pilot_label,
                    f"Run stopped: {_stop_reason(payload.get('stop_code'))}",
                )
                if item
            ]
        else:
            action, parts = "ERROR", [
                item for item in (run_label, pilot_label, "Run failed") if item
            ]
            parts.append(f"Details: {debug_log_path}")
        elapsed = _duration(payload.get("runtime_seconds"))
        ram = _gib(payload.get("peak_process_tree_rss_bytes"))
        available = _gib(payload.get("minimum_system_available_ram_bytes"))
        if elapsed:
            parts.append(f"Elapsed {elapsed}")
        if ram:
            parts.append(f"Peak RAM {ram}")
        if available:
            parts.append(f"Minimum available {available}")
    elif event == "session_completed":
        action, parts = "DONE", [_clean_message(payload.get("message")) or "Research session completed"]
    elif event == "session_controlled_stop":
        action, parts = "STOP", [_stop_reason(payload.get("stop_code"), payload.get("message"))]
    elif event == "session_interrupted":
        action, parts = "STOP", [_clean_message(payload.get("message")) or "Research run interrupted manually"]
    elif event == "session_failed":
        action = "ERROR"
        parts = [_clean_message(payload.get("message")) or "Unexpected research error", f"Details: {debug_log_path}"]
    elif event == "logging_backpressure":
        action, parts = "WARN", ["Some internal audit events were dropped under queue pressure"]
    elif event == "python_log":
        action = "ERROR" if level in {"ERROR", "CRITICAL"} else "WARN"
        parts = [_clean_message(payload.get("message"))]
        if payload.get("exception_class") and action == "ERROR":
            parts.append(f"Details: {debug_log_path}")
    elif event == "session_finalized":
        action, parts = "WARN", [_clean_message(payload.get("message"))]
    elif level in {"WARNING", "ERROR", "CRITICAL"}:
        action = "ERROR" if level in {"ERROR", "CRITICAL"} else "WARN"
        parts = [_clean_message(payload.get("message"))]
    else:
        return None

    rendered = " | ".join(part for part in parts if part)
    action_spacing = "   " if action == "WAIT" else " " * max(1, 6 - len(action))
    return (
        f"[{_human_timestamp(payload.get('timestamp_utc'))}] "
        f"{action}{action_spacing}| {rendered}"
    )


@contextmanager
def suppress_third_party_output() -> Iterator[None]:
    """Silence raw library progress while canonical parent heartbeats remain visible."""

    with open(os.devnull, "w", encoding="utf-8") as sink:
        with redirect_stdout(sink), redirect_stderr(sink):
            yield


class _ParentEmitter:
    def __init__(self, session: "ResearchLogSession") -> None:
        self.session = session

    def emit(self, payload: Mapping[str, Any], *, priority: bool) -> bool:
        del priority
        self.session._write(payload)
        return True


class _WorkerEmitter:
    def __init__(
        self,
        target_queue: Any,
        *,
        routine_drop_counter: Any,
        priority_drop_counter: Any,
    ) -> None:
        self.target_queue = target_queue
        self.routine_drop_counter = routine_drop_counter
        self.priority_drop_counter = priority_drop_counter

    def emit(self, payload: Mapping[str, Any], *, priority: bool) -> bool:
        try:
            if priority:
                self.target_queue.put(
                    dict(payload), timeout=PRIORITY_QUEUE_TIMEOUT_SECONDS
                )
            else:
                self.target_queue.put_nowait(dict(payload))
            return True
        except (queue.Full, BrokenPipeError, EOFError, OSError, ValueError):
            counter = (
                self.priority_drop_counter if priority else self.routine_drop_counter
            )
            lock_getter = getattr(counter, "get_lock", None)
            if callable(lock_getter):
                with lock_getter():
                    counter.value = int(counter.value) + 1
            else:
                counter.value = int(counter.value) + 1
            return False


class _PythonLoggingBridge(logging.Handler):
    """Mirror ordinary Python logging into the audit stream and human warnings."""

    _research_logging_bridge = True

    def emit(self, record: logging.LogRecord) -> None:
        try:
            fields: dict[str, Any] = {"logger": record.name}
            if record.exc_info:
                fields["exception_class"] = record.exc_info[0].__name__
                fields["traceback"] = "".join(
                    traceback_module.format_exception(*record.exc_info)
                )
            emit_research_event(
                "python_log",
                level=record.levelname,
                message=record.getMessage(),
                priority=record.levelno >= logging.ERROR,
                **fields,
            )
        except Exception:
            self.handleError(record)


def research_logging_active() -> bool:
    return _ACTIVE_EMITTER is not None


def emit_research_event(
    event: str,
    *,
    level: str = "INFO",
    message: str = "",
    priority: bool = False,
    **fields: Any,
) -> bool:
    """Emit one bounded, scalar-only record when a research session is active."""

    emitter = _ACTIVE_EMITTER
    if emitter is None:
        return False
    context = dict(_CONTEXT.get())
    context.update(fields)
    session_id = str(context.pop("session_id", "unknown"))
    payload = _record(
        event,
        level=level,
        message=message,
        session_id=session_id,
        fields=context,
    )
    return bool(emitter.emit(payload, priority=priority))


def emit_contextual_research_event(
    event: str,
    *contexts: Mapping[str, Any],
    level: str = "INFO",
    message: str = "",
    priority: bool = False,
    **fields: Any,
) -> bool:
    """Merge contextual fields with explicit precedence before event emission."""

    merged: dict[str, Any] = {}
    for context in contexts:
        merged.update(context)
    merged.update(fields)
    for reserved in ("event", "level", "message", "priority"):
        merged.pop(reserved, None)
    return emit_research_event(
        event,
        level=level,
        message=message,
        priority=priority,
        **merged,
    )


@contextmanager
def bind_research_context(**fields: Any) -> Iterator[None]:
    context = dict(_CONTEXT.get())
    context.update(fields)
    token = _CONTEXT.set(context)
    try:
        yield
    finally:
        _CONTEXT.reset(token)


@contextmanager
def logged_stage(
    stage: str,
    *,
    message: str,
    component: str | None = None,
    heartbeat_interval_seconds: float | None = STAGE_HEARTBEAT_INTERVAL_SECONDS,
    **fields: Any,
) -> Iterator[None]:
    """Log one synchronous stage without changing the wrapped computation."""

    started = monotonic()
    context = {"stage": stage, **fields}
    if component is not None:
        context["component"] = component
    emit_contextual_research_event(
        "stage_started", context, message=message, priority=True
    )
    heartbeat_stop = threading.Event()
    heartbeat_thread: threading.Thread | None = None
    heartbeat_interval = (
        None
        if heartbeat_interval_seconds is None
        else max(0.01, float(heartbeat_interval_seconds))
    )
    captured_context = dict(_CONTEXT.get())

    if heartbeat_interval is not None:
        def heartbeat_loop() -> None:
            while not heartbeat_stop.wait(heartbeat_interval):
                memory: dict[str, Any] = {}
                try:
                    import psutil

                    memory = {
                        "parent_rss_bytes": int(
                            psutil.Process(os.getpid()).memory_info().rss
                        ),
                        "system_available_ram_bytes": int(
                            psutil.virtual_memory().available
                        ),
                    }
                except (ImportError, OSError):
                    pass
                with bind_research_context(**captured_context):
                    emit_contextual_research_event(
                        "stage_heartbeat",
                        context,
                        memory,
                        message=f"{message} remains active",
                        elapsed_stage_seconds=monotonic() - started,
                    )

        heartbeat_thread = threading.Thread(
            target=heartbeat_loop,
            name=f"research-stage-heartbeat-{stage}",
            daemon=False,
        )
        heartbeat_thread.start()

    def stop_heartbeat() -> None:
        heartbeat_stop.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=1.0)

    try:
        yield
    except KeyboardInterrupt:
        stop_heartbeat()
        emit_contextual_research_event(
            "stage_interrupted",
            context,
            level="ERROR",
            message=f"{message} interrupted",
            priority=True,
            elapsed_stage_seconds=monotonic() - started,
            exception_class="KeyboardInterrupt",
        )
        raise
    except BaseException as exc:
        stop_heartbeat()
        emit_contextual_research_event(
            "stage_failed",
            context,
            level="ERROR",
            message=f"{message} failed: {type(exc).__name__}: {exc}",
            priority=True,
            elapsed_stage_seconds=monotonic() - started,
            exception_class=type(exc).__name__,
            traceback="".join(
                traceback_module.format_exception(type(exc), exc, exc.__traceback__)
            ),
        )
        raise
    else:
        stop_heartbeat()
        emit_contextual_research_event(
            "stage_completed",
            context,
            message=f"{message} completed",
            priority=True,
            elapsed_stage_seconds=monotonic() - started,
        )
    finally:
        stop_heartbeat()


class ResearchLogSession:
    """Own human, audit, and debug sinks plus the bounded worker-log listener."""

    def __init__(
        self,
        log_path: str | Path,
        *,
        repository_root: str | Path,
        command_arguments: list[str],
        terminal_stream: TextIO | None = None,
        queue_capacity: int = DEFAULT_QUEUE_CAPACITY,
        session_id: str | None = None,
    ) -> None:
        self.repository_root = Path(repository_root).resolve()
        self.log_path = Path(log_path)
        if not self.log_path.is_absolute():
            self.log_path = self.repository_root / self.log_path
        self.log_path = self.log_path.resolve()
        self.audit_path = self.log_path.with_name(DEFAULT_AUDIT_LOG_NAME)
        self.debug_path = self.log_path.with_name(DEFAULT_DEBUG_LOG_NAME)
        self.command_arguments = list(command_arguments)
        self.terminal_stream = terminal_stream or sys.stderr
        self.session_id = session_id or str(uuid.uuid4())
        self.queue_capacity = max(8, int(queue_capacity))
        self._file: TextIO | None = None
        self._audit_file: TextIO | None = None
        self._debug_file: TextIO | None = None
        self._write_lock = threading.Lock()
        self._listener_stop = threading.Event()
        self._listener: threading.Thread | None = None
        self._root_handler: logging.Handler | None = None
        self._previous_root_level: int | None = None
        self._terminal_event_written = False
        context = multiprocessing.get_context("spawn")
        self.worker_queue = context.Queue(maxsize=self.queue_capacity)
        self.routine_drop_counter = context.Value("Q", 0)
        self.priority_drop_counter = context.Value("Q", 0)
        self._observed_routine_drops = 0
        self._observed_priority_drops = 0
        self._written_traceback_hashes: set[int] = set()

    def __enter__(self) -> "ResearchLogSession":
        global _ACTIVE_EMITTER, _ACTIVE_SESSION
        if _ACTIVE_SESSION is not None:
            raise RuntimeError("a research logging session is already active")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._audit_file = self.audit_path.open(
            "a", encoding="utf-8", newline="\n", buffering=1
        )
        self._debug_file = self.debug_path.open(
            "a", encoding="utf-8", newline="\n", buffering=1
        )
        self._migrate_legacy_json_run_log()
        self._file = self.log_path.open(
            "a", encoding="utf-8", newline="\n", buffering=1
        )
        _ACTIVE_SESSION = self
        _ACTIVE_EMITTER = _ParentEmitter(self)
        _CONTEXT.set({"session_id": self.session_id})
        self._install_python_logging_bridge()
        self._listener = threading.Thread(
            target=self._listen,
            name=f"research-log-listener-{self.session_id}",
            daemon=False,
        )
        self._listener.start()
        emit_research_event(
            "logging_initialized",
            message="Durable append-only research logging initialized",
            priority=True,
            log_path=self.relative_path(self.log_path),
            audit_path=self.relative_path(self.audit_path),
            debug_path=self.relative_path(self.debug_path),
            format="human_text",
            queue_capacity=self.queue_capacity,
            heartbeat_interval_seconds=STAGE_HEARTBEAT_INTERVAL_SECONDS,
        )
        emit_research_event(
            "session_started",
            message="Research runner session started",
            priority=True,
            command_arguments=self.command_arguments,
            repository_root=self.repository_root.as_posix(),
        )
        return self

    def _install_python_logging_bridge(self) -> None:
        root = logging.getLogger()
        self._previous_root_level = root.level
        for handler in root.handlers:
            if getattr(handler, "_research_logging_bridge", False):
                self._root_handler = handler
                return
        handler = _PythonLoggingBridge()
        handler.setLevel(logging.INFO)
        root.addHandler(handler)
        self._root_handler = handler
        if root.level == logging.NOTSET or root.level > logging.INFO:
            root.setLevel(logging.INFO)

    def relative_path(self, path: str | Path) -> str:
        candidate = Path(path).resolve()
        try:
            return candidate.relative_to(self.repository_root).as_posix()
        except ValueError:
            return candidate.as_posix()

    def worker_transport(self) -> tuple[Any, str, Any, Any]:
        return (
            self.worker_queue,
            self.session_id,
            self.routine_drop_counter,
            self.priority_drop_counter,
        )

    def finish(
        self,
        event: str,
        *,
        level: str = "INFO",
        message: str,
        **fields: Any,
    ) -> None:
        emit_research_event(
            event,
            level=level,
            message=message,
            priority=True,
            **fields,
        )
        self._terminal_event_written = True

    def _write_debug_traceback(
        self, payload: Mapping[str, Any], traceback_text: str
    ) -> None:
        if (
            not traceback_text
            or str(payload.get("exception_class")) in _NO_DEBUG_TRACEBACK_EXCEPTIONS
        ):
            return
        traceback_hash = hash(traceback_text)
        if traceback_hash in self._written_traceback_hashes:
            return
        if len(self._written_traceback_hashes) >= 128:
            self._written_traceback_hashes.clear()
        self._written_traceback_hashes.add(traceback_hash)
        if self._debug_file is None:
            return
        header = (
            f"[{_human_timestamp(payload.get('timestamp_utc'))}] "
            f"{payload.get('event', 'error')} | "
            f"{payload.get('exception_class', 'Exception')} | "
            f"{_clean_message(payload.get('message'))}"
        )
        self._debug_file.write(header + "\n")
        self._debug_file.write(traceback_text.rstrip() + "\n\n")
        self._debug_file.flush()

    def _audit_payload(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        audit_payload = dict(payload)
        traceback_text = str(audit_payload.pop("traceback", "") or "")
        if (
            traceback_text
            and str(audit_payload.get("exception_class"))
            not in _NO_DEBUG_TRACEBACK_EXCEPTIONS
        ):
            self._write_debug_traceback(audit_payload, traceback_text)
            audit_payload["debug_log_path"] = self.relative_path(self.debug_path)
        return audit_payload

    def _migrate_legacy_json_run_log(self) -> None:
        """Convert prior JSON run lines to human text while preserving JSON in audit."""

        if not self.log_path.is_file() or self.log_path.stat().st_size == 0:
            return
        temporary = self.log_path.with_name(
            f"{self.log_path.name}.{os.getpid()}.human-migration"
        )
        converted = False
        with self.log_path.open("r", encoding="utf-8", errors="replace") as source:
            with temporary.open("w", encoding="utf-8", newline="\n") as target:
                for raw_line in source:
                    stripped = raw_line.strip()
                    payload: Mapping[str, Any] | None = None
                    if stripped.startswith("{"):
                        try:
                            candidate = json.loads(stripped)
                            if isinstance(candidate, Mapping):
                                payload = candidate
                        except json.JSONDecodeError:
                            payload = None
                    if payload is None:
                        target.write(raw_line if raw_line.endswith("\n") else raw_line + "\n")
                        continue
                    converted = True
                    audit_payload = self._audit_payload(payload)
                    if self._audit_file is not None:
                        self._audit_file.write(
                            json.dumps(
                                audit_payload,
                                sort_keys=True,
                                separators=(",", ":"),
                            )
                            + "\n"
                        )
                    human = _human_line(
                        audit_payload,
                        debug_log_path=self.relative_path(self.debug_path),
                    )
                    if human:
                        target.write(human + "\n")
                target.flush()
                os.fsync(target.fileno())
        if converted:
            if self._audit_file is not None:
                self._audit_file.flush()
            os.replace(temporary, self.log_path)
        else:
            temporary.unlink(missing_ok=True)

    def _write(self, payload: Mapping[str, Any]) -> None:
        with self._write_lock:
            if self._file is None or self._audit_file is None:
                return
            audit_payload = self._audit_payload(payload)
            audit_line = json.dumps(
                audit_payload, sort_keys=True, separators=(",", ":")
            )
            self._audit_file.write(audit_line + "\n")
            self._audit_file.flush()
            human_line = _human_line(
                audit_payload,
                debug_log_path=self.relative_path(self.debug_path),
            )
            if human_line:
                self._file.write(human_line + "\n")
                self._file.flush()
                self.terminal_stream.write(human_line + "\n")
                self.terminal_stream.flush()

    def _report_backpressure(self) -> None:
        routine = int(self.routine_drop_counter.value)
        priority = int(self.priority_drop_counter.value)
        if (
            routine == self._observed_routine_drops
            and priority == self._observed_priority_drops
        ):
            return
        delta_routine = routine - self._observed_routine_drops
        delta_priority = priority - self._observed_priority_drops
        self._observed_routine_drops = routine
        self._observed_priority_drops = priority
        payload = _record(
            "logging_backpressure",
            level="WARNING" if priority == 0 else "ERROR",
            message="Bounded worker logging queue dropped records",
            session_id=self.session_id,
            fields={
                "routine_records_dropped": routine,
                "priority_records_dropped": priority,
                "routine_records_dropped_since_last_report": delta_routine,
                "priority_records_dropped_since_last_report": delta_priority,
            },
        )
        self._write(payload)

    def _listen(self) -> None:
        while True:
            try:
                payload = self.worker_queue.get(timeout=0.1)
            except queue.Empty:
                self._report_backpressure()
                if self._listener_stop.is_set():
                    break
                continue
            except (EOFError, OSError, ValueError):
                break
            if isinstance(payload, Mapping):
                self._write(payload)
            self._report_backpressure()
        while True:
            try:
                payload = self.worker_queue.get_nowait()
            except (queue.Empty, EOFError, OSError, ValueError):
                break
            if isinstance(payload, Mapping):
                self._write(payload)
        self._report_backpressure()

    @property
    def listener_alive(self) -> bool:
        return bool(self._listener and self._listener.is_alive())

    def close(self) -> None:
        global _ACTIVE_EMITTER, _ACTIVE_SESSION
        if self._file is None:
            return
        if not self._terminal_event_written:
            emit_research_event(
                "session_finalized",
                level="WARNING",
                message="Logging session closed without an explicit workflow terminal event",
                priority=True,
            )
        self._listener_stop.set()
        if self._listener is not None:
            self._listener.join(timeout=LISTENER_SHUTDOWN_TIMEOUT_SECONDS)
        if self.listener_alive:
            emit_research_event(
                "logging_listener_shutdown_timeout",
                level="ERROR",
                message="Worker logging listener did not stop within its bounded timeout",
                priority=True,
            )
        emit_research_event(
            "logging_shutdown",
            message="Durable research logging shutdown completed",
            priority=True,
            listener_alive=self.listener_alive,
            routine_records_dropped=int(self.routine_drop_counter.value),
            priority_records_dropped=int(self.priority_drop_counter.value),
        )
        root = logging.getLogger()
        if self._root_handler is not None and self._root_handler in root.handlers:
            root.removeHandler(self._root_handler)
        if self._previous_root_level is not None:
            root.setLevel(self._previous_root_level)
        _ACTIVE_EMITTER = None
        _ACTIVE_SESSION = None
        _CONTEXT.set({})
        try:
            self.worker_queue.cancel_join_thread()
        except (AttributeError, OSError, ValueError):
            pass
        try:
            self.worker_queue.close()
        except (OSError, ValueError):
            pass
        with self._write_lock:
            for handle in (self._file, self._audit_file, self._debug_file):
                if handle is not None:
                    handle.flush()
                    handle.close()
            self._file = None
            self._audit_file = None
            self._debug_file = None

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


def active_worker_transport() -> tuple[Any, str, Any, Any] | None:
    session = _ACTIVE_SESSION
    return None if session is None else session.worker_transport()


def configure_worker_logging(
    target_queue: Any,
    *,
    session_id: str,
    context: Mapping[str, Any],
    routine_drop_counter: Any,
    priority_drop_counter: Any,
) -> None:
    """Configure one spawned worker without opening the durable file."""

    global _ACTIVE_EMITTER, _ACTIVE_SESSION
    _ACTIVE_SESSION = None
    _ACTIVE_EMITTER = _WorkerEmitter(
        target_queue,
        routine_drop_counter=routine_drop_counter,
        priority_drop_counter=priority_drop_counter,
    )
    worker_context = {"session_id": session_id, **dict(context)}
    _CONTEXT.set(worker_context)
    root = logging.getLogger()
    if not any(
        getattr(handler, "_research_logging_bridge", False)
        for handler in root.handlers
    ):
        root.addHandler(_PythonLoggingBridge())
    if root.level == logging.NOTSET or root.level > logging.INFO:
        root.setLevel(logging.INFO)
    emit_research_event(
        "worker_logging_configured",
        message="Spawned worker logging configured",
        priority=True,
        worker_pid=os.getpid(),
    )


__all__ = [
    "DEFAULT_AUDIT_LOG_NAME",
    "DEFAULT_DEBUG_LOG_NAME",
    "DEFAULT_QUEUE_CAPACITY",
    "DEFAULT_RESEARCH_LOG",
    "LOG_SCHEMA_VERSION",
    "ResearchLogSession",
    "STAGE_HEARTBEAT_INTERVAL_SECONDS",
    "active_worker_transport",
    "bind_research_context",
    "configure_worker_logging",
    "emit_contextual_research_event",
    "emit_research_event",
    "logged_stage",
    "research_logging_active",
    "suppress_third_party_output",
]
