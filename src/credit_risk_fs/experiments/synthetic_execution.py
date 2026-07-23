"""Tiny bounded worker targets used only by resource-safety validation tests."""

from __future__ import annotations

import multiprocessing
import os
import time
from typing import Any


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
