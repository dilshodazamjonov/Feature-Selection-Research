"""Tiny bounded worker targets used only by resource-safety validation tests."""

from __future__ import annotations

import multiprocessing
import os
import time
from typing import Any

from credit_risk_fs.experiments.research_logging import (
    configure_worker_logging,
    emit_research_event,
)


def _child_wait(stop_event: Any) -> None:
    while not stop_event.is_set():
        stop_event.wait(0.02)


def _child_ignore_stop() -> None:
    while True:
        time.sleep(0.02)


def bounded_memory_worker(
    *,
    stop_event: Any,
    stage_queue: Any,
    chunk_mb: int = 4,
    maximum_allocation_mb: int = 160,
    spawn_child: bool = False,
) -> dict[str, Any]:
    """Allocate gradually up to a hard cap; this intentionally cannot cause an OOM."""

    if chunk_mb <= 0 or maximum_allocation_mb <= 0:
        raise ValueError("synthetic allocation bounds must be positive")
    child = None
    if spawn_child:
        context = multiprocessing.get_context("spawn")
        child = context.Process(target=_child_wait, args=(stop_event,), daemon=False)
        child.start()
    stage_queue.put({"stage": "synthetic_bounded_allocation", "fold_id": None})
    allocations: list[bytearray] = []
    allocated = 0
    try:
        while allocated < maximum_allocation_mb and not stop_event.is_set():
            size_mb = min(chunk_mb, maximum_allocation_mb - allocated)
            block = bytearray(size_mb * 1024 * 1024)
            block[0] = 1
            block[-1] = 1
            allocations.append(block)
            allocated += size_mb
            stop_event.wait(0.04)
        while not stop_event.is_set():
            stop_event.wait(0.05)
        return {"allocated_mb": allocated, "cooperative_stop": True}
    finally:
        if child is not None:
            stop_event.set()
            child.join(timeout=2)
            if child.is_alive():
                child.terminate()
                child.join(timeout=2)


def cooperative_wait_worker(
    *,
    stop_event: Any,
    stage_queue: Any,
) -> dict[str, bool]:
    stage_queue.put({"stage": "waiting", "fold_id": None})
    while not stop_event.is_set():
        stop_event.wait(0.02)
    return {"stopped": True}


def unexpected_exit_worker(
    *,
    stop_event: Any,
    stage_queue: Any,
    exit_code: int = 7,
) -> None:
    del stop_event
    stage_queue.put({"stage": "unexpected_exit", "fold_id": None})
    os._exit(int(exit_code))


def immediate_success_worker(
    *,
    stop_event: Any,
    stage_queue: Any,
) -> dict[str, bool]:
    if stop_event.is_set():
        raise RuntimeError("unexpected stop")
    stage_queue.put({"stage": "completed", "fold_id": None})
    time.sleep(0.03)
    return {"ok": True}


def cooperative_ram_gate_worker(
    *,
    stop_event: Any,
    stage_queue: Any,
    ram_ready_event: Any,
    active_duration_seconds: float = 0.08,
) -> dict[str, bool]:
    """Synthetic loader-like work that pauses only at cooperative boundaries."""

    stage_queue.put({"stage": "data_loading", "fold_id": None})
    active = 0.0
    while active < float(active_duration_seconds):
        if stop_event.is_set():
            return {"completed": False}
        if not ram_ready_event.is_set():
            ram_ready_event.wait(0.01)
            continue
        started = time.monotonic()
        stop_event.wait(0.005)
        active += time.monotonic() - started
    return {"completed": True}


def opaque_ram_stage_worker(
    *,
    stop_event: Any,
    stage_queue: Any,
    ram_ready_event: Any,
    duration_seconds: float = 0.08,
) -> dict[str, bool]:
    """Represent a library call that cannot inspect the cooperative RAM gate."""

    del ram_ready_event
    stage_queue.put({"stage": "final_model_fit", "fold_id": None})
    stop_event.wait(float(duration_seconds))
    if stop_event.is_set():
        return {"completed": False}
    return {"completed": True}


def cooperative_recovery_boundary_worker(
    *,
    stop_event: Any,
    stage_queue: Any,
    ram_ready_event: Any,
    ram_recovery_threshold_bytes: int,
) -> dict[str, Any]:
    """Synthetic opaque-stage barrier; it performs no memory allocation."""

    del ram_recovery_threshold_bytes
    ram_ready_event.clear()
    stage_queue.put(
        {
            "stage": "final_model_fit",
            "fold_id": None,
            "ram_recovery_barrier": True,
            "ram_boundary": "stage:final_model_fit",
        }
    )
    while not ram_ready_event.wait(0.01):
        if stop_event.is_set():
            return {"completed": False}
    return {"completed": True}


def memory_error_worker(
    *, stop_event: Any, stage_queue: Any, ram_ready_event: Any
) -> None:
    del stop_event, ram_ready_event
    stage_queue.put({"stage": "synthetic_memory_error", "fold_id": None})
    raise MemoryError("synthetic allocation failure")


def duplicate_stage_context_worker(
    *,
    stop_event: Any,
    stage_queue: Any,
    spec: dict[str, Any],
) -> dict[str, bool]:
    """Publish stage metadata that intentionally overlaps supervisor context."""

    if stop_event.is_set():
        raise RuntimeError("unexpected stop")
    stage_queue.put(
        {
            "stage": "pilot_dev_data_loading",
            "fold_id": 1,
            "component": "prepare_voting_pilot_dev_data",
            "pilot_cell": spec["cell_id"],
            "dataset": spec["dataset"],
            "method_id": spec["method_id"],
        }
    )
    time.sleep(0.06)
    return {"ok": True}


def uncooperative_wait_worker(
    *,
    stop_event: Any,
    stage_queue: Any,
    spawn_stubborn_child: bool = False,
) -> None:
    """Ignore cooperative cancellation so the supervisor must escalate."""

    del stop_event
    child = None
    if spawn_stubborn_child:
        context = multiprocessing.get_context("spawn")
        child = context.Process(target=_child_ignore_stop, daemon=False)
        child.start()
    stage_queue.put({"stage": "uncooperative_wait", "fold_id": None})
    while True:
        time.sleep(0.02)


def saturated_stage_queue_worker(
    *,
    stop_event: Any,
    stage_queue: Any,
) -> dict[str, bool]:
    """Over-publish stage updates; the worker-side publisher must never block."""

    for index in range(20_000):
        stage_queue.put({"stage": "queue_saturation", "fold_id": index})
    if stop_event.is_set():
        return {"queue_saturation_completed": True}
    return {"queue_saturation_completed": True}


def oversized_result_worker(
    *,
    stop_event: Any,
    stage_queue: Any,
) -> dict[str, bytes]:
    del stop_event
    stage_queue.put({"stage": "oversized_result", "fold_id": None})
    return {"forbidden_large_payload": b"x" * (2 * 1024 * 1024)}


def near_limit_result_worker(
    *,
    stop_event: Any,
    stage_queue: Any,
) -> dict[str, bytes]:
    """Publish enough compact data to require concurrent result-queue draining."""

    del stop_event
    stage_queue.put({"stage": "near_limit_result", "fold_id": None})
    return {"bounded_metadata": b"x" * (512 * 1024)}


def mock_slow_boruta_worker(
    *,
    stop_event: Any,
    stage_queue: Any,
    duration_seconds: float = 0.18,
) -> dict[str, int]:
    """Exercise Boruta observability without fitting a scientific model."""

    stage_queue.put(
        {
            "stage": "voter_boruta",
            "fold_id": 5,
            "component": "boruta",
            "internal_iteration_available": False,
        }
    )
    emit_research_event(
        "component_started",
        message="Synthetic Boruta fit started",
        priority=True,
        component="boruta",
        stage="voter_boruta",
        fold_id=5,
        input_feature_count=12,
        max_iter=100,
        internal_iteration_available=False,
    )
    stop_event.wait(max(0.01, float(duration_seconds)))
    if stop_event.is_set():
        raise KeyboardInterrupt("synthetic Boruta interrupted")
    result = {"confirmed": 4, "tentative": 2, "rejected": 6}
    emit_research_event(
        "component_completed",
        message="Synthetic Boruta fit completed",
        priority=True,
        component="boruta",
        stage="voter_boruta",
        fold_id=5,
        **result,
    )
    return result


def mock_model_fit_worker(
    *, stop_event: Any, stage_queue: Any, duration_seconds: float = 0.04
) -> dict[str, bool]:
    """Exercise model-fit stage logging without loading research data."""

    stage_queue.put(
        {
            "stage": "final_model_fit",
            "fold_id": 2,
            "details": {"component": "catboost"},
        }
    )
    stop_event.wait(max(0.01, float(duration_seconds)))
    if stop_event.is_set():
        raise KeyboardInterrupt("synthetic model fit interrupted")
    return {"fit_completed": True}


def keyboard_interrupt_worker(*, stop_event: Any, stage_queue: Any) -> None:
    del stop_event
    stage_queue.put({"stage": "synthetic_interrupt", "fold_id": None})
    raise KeyboardInterrupt("synthetic worker interrupt")


def exception_worker(*, stop_event: Any, stage_queue: Any) -> None:
    del stop_event
    stage_queue.put({"stage": "synthetic_failure", "fold_id": None})
    raise RuntimeError("synthetic worker failure")


def logging_transport_process(
    target_queue: Any,
    session_id: str,
    routine_drop_counter: Any,
    priority_drop_counter: Any,
    prefix: str,
    count: int,
) -> None:
    """Publish bounded synthetic records from a standalone spawned process."""

    configure_worker_logging(
        target_queue,
        session_id=session_id,
        context={"run_id": f"synthetic-{prefix}", "component": "transport_test"},
        routine_drop_counter=routine_drop_counter,
        priority_drop_counter=priority_drop_counter,
    )
    for index in range(max(0, int(count))):
        emit_research_event(
            "synthetic_transport_record",
            message="Synthetic concurrent logging record",
            record_prefix=prefix,
            record_index=index,
        )
