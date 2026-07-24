"""Process-safe, append-only JSONL logging for manual research workflows."""

from __future__ import annotations

import json
import logging
import multiprocessing
import os
import queue
import sys
import threading
import traceback as traceback_module
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any, Iterator, Mapping, TextIO


DEFAULT_RESEARCH_LOG = Path("logs/runs.log")
LOG_SCHEMA_VERSION = "research_run_log_v1"
DEFAULT_QUEUE_CAPACITY = 1024
PRIORITY_QUEUE_TIMEOUT_SECONDS = 0.25
LISTENER_SHUTDOWN_TIMEOUT_SECONDS = 2.0
STAGE_HEARTBEAT_INTERVAL_SECONDS = 30.0
_MAX_STRING_LENGTH = 16_384
_MAX_SEQUENCE_ITEMS = 64
_MAX_MAPPING_ITEMS = 64


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
    """Mirror ordinary Python logging into the canonical JSONL stream."""

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
    emit_research_event(
        "stage_started", message=message, priority=True, **context
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
                    emit_research_event(
                        "stage_heartbeat",
                        message=f"{message} remains active",
                        elapsed_stage_seconds=monotonic() - started,
                        **context,
                        **memory,
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
        emit_research_event(
            "stage_interrupted",
            level="ERROR",
            message=f"{message} interrupted",
            priority=True,
            elapsed_stage_seconds=monotonic() - started,
            exception_class="KeyboardInterrupt",
            traceback=traceback_module.format_exc(),
            **context,
        )
        raise
    except BaseException as exc:
        stop_heartbeat()
        emit_research_event(
            "stage_failed",
            level="ERROR",
            message=f"{message} failed: {type(exc).__name__}: {exc}",
            priority=True,
            elapsed_stage_seconds=monotonic() - started,
            exception_class=type(exc).__name__,
            traceback="".join(
                traceback_module.format_exception(type(exc), exc, exc.__traceback__)
            ),
            **context,
        )
        raise
    else:
        stop_heartbeat()
        emit_research_event(
            "stage_completed",
            message=f"{message} completed",
            priority=True,
            elapsed_stage_seconds=monotonic() - started,
            **context,
        )
    finally:
        stop_heartbeat()


class ResearchLogSession:
    """Own the only durable file handle and the bounded worker-log listener."""

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
        self.command_arguments = list(command_arguments)
        self.terminal_stream = terminal_stream or sys.stderr
        self.session_id = session_id or str(uuid.uuid4())
        self.queue_capacity = max(8, int(queue_capacity))
        self._file: TextIO | None = None
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

    def __enter__(self) -> "ResearchLogSession":
        global _ACTIVE_EMITTER, _ACTIVE_SESSION
        if _ACTIVE_SESSION is not None:
            raise RuntimeError("a research logging session is already active")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
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
            format="jsonl",
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

    def _write(self, payload: Mapping[str, Any]) -> None:
        line = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        with self._write_lock:
            if self._file is None:
                return
            self._file.write(line + "\n")
            self._file.flush()
            self.terminal_stream.write(line + "\n")
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
            self._file.flush()
            self._file.close()
            self._file = None

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
    "DEFAULT_QUEUE_CAPACITY",
    "DEFAULT_RESEARCH_LOG",
    "LOG_SCHEMA_VERSION",
    "ResearchLogSession",
    "STAGE_HEARTBEAT_INTERVAL_SECONDS",
    "active_worker_transport",
    "bind_research_context",
    "configure_worker_logging",
    "emit_research_event",
    "logged_stage",
    "research_logging_active",
]
