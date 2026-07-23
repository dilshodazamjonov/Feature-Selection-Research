"""Run only bounded synthetic process-lifecycle checks and publish their timings."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from time import monotonic


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from credit_risk_fs.experiments.resource_monitor import (  # noqa: E402
    RAM_PROCESS_LIMIT,
    ProcessTreeSampler,
    _StopCauseRecorder,
    _shutdown_owned_worker,
    supervise_worker,
)
from credit_risk_fs.experiments.resource_policy import (  # noqa: E402
    DiskPolicy,
    GpuPolicy,
    MemoryPolicy,
    MonitoringPolicy,
    ParallelismPolicy,
    ResolvedExecutionPolicy,
)


AUDIT = PROJECT_ROOT / (
    "cleanup/audits/cross_dataset_voting_resource_stop_resume_safety/"
    "synthetic_stop_validation.json"
)


def _policy() -> ResolvedExecutionPolicy:
    return ResolvedExecutionPolicy(
        schema_version="resource_safe_execution_policy_v1",
        profile_name="cdv1_prompt61_synthetic",
        parallelism=ParallelismPolicy(1, 1, 0, 1, False),
        memory=MemoryPolicy(0.001, 0.000001, 0.000002, 0.001, 1.35),
        gpu=GpuPolicy(0.1, 1.0, 2.0, True),
        disk=DiskPolicy(0.001, 0.001, 2.5),
        monitoring=MonitoringPolicy(0.01, 1.0, 0.2),
        configured_policy_path="synthetic_only",
    )


class _TriggerAfterChildSampler(ProcessTreeSampler):
    def sample(self, *args, **kwargs):
        sample = super().sample(*args, **kwargs)
        return replace(
            sample,
            process_tree_rss_bytes=(10 * 1024**3 if sample.child_pids else 0),
        )


class _FakeStopEvent:
    def set(self):
        return None


class _FakeProcess:
    pid = 60001

    def __init__(self):
        self.alive = True

    def join(self, timeout=None):
        del timeout

    def is_alive(self):
        return self.alive


class _FakeStubbornRegistry:
    def __init__(self, process):
        self.process = process
        self.force_kill_called = False

    def alive(self):
        return [self.process] if self.process.alive else []

    def terminate_phase(self, *, timeout_seconds):
        del timeout_seconds
        return self.alive()

    def kill_phase(self, *, timeout_seconds):
        del timeout_seconds
        self.force_kill_called = True
        self.process.alive = False
        return []

    def survivor_records(self):
        return ()


def _run(
    worker: str,
    kwargs: dict,
    *,
    sampler_factory=ProcessTreeSampler,
    policy: ResolvedExecutionPolicy | None = None,
) -> dict:
    started = monotonic()
    result = supervise_worker(
        worker_target=worker,
        worker_kwargs=kwargs,
        policy=policy or _policy(),
        results_root=PROJECT_ROOT / "tests_runtime",
        temp_root=PROJECT_ROOT / "tests_runtime",
        sampler_factory=sampler_factory,
        run_association=f"prompt61-synthetic:{worker.rsplit(':', 1)[-1]}",
    )
    return {
        "status": result.status,
        "primary_stop_code": result.primary_stop_code,
        "shutdown_elapsed_seconds": result.shutdown_elapsed_seconds,
        "wall_elapsed_seconds": monotonic() - started,
        "graceful_stop_completed": result.graceful_stop_completed,
        "lifecycle_states": [item["state"] for item in result.stop_lifecycle],
        "owned_process_count": len(result.owned_processes),
        "survivor_process_count": len(result.survivor_processes),
        "child_cleanup_confirmed": result.child_cleanup_confirmed,
        "queue_cleanup_confirmed": result.queue_cleanup_confirmed,
    }


def main() -> int:
    import multiprocessing
    import psutil

    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    before_children = [
        int(item.pid) for item in psutil.Process(os.getpid()).children(recursive=True)
    ]
    cooperative = _run(
        "credit_risk_fs.experiments.synthetic_execution:cooperative_wait_worker", {}
    )
    uncooperative = _run(
        "credit_risk_fs.experiments.synthetic_execution:uncooperative_wait_worker", {}
    )
    nested = _run(
        "credit_risk_fs.experiments.synthetic_execution:uncooperative_wait_worker",
        {"spawn_stubborn_child": True},
        sampler_factory=_TriggerAfterChildSampler,
    )
    queue_case = _run(
        "credit_risk_fs.experiments.synthetic_execution:saturated_stage_queue_worker",
        {},
        policy=replace(
            _policy(),
            memory=MemoryPolicy(1.0, 1.0, 2.0, 0.001, 1.35),
        ),
    )
    process = _FakeProcess()
    registry = _FakeStubbornRegistry(process)
    recorder = _StopCauseRecorder()
    recorder.observe(RAM_PROCESS_LIMIT, elapsed_seconds=0.0, detail="synthetic")
    lifecycle = []
    started = monotonic()
    ok, survivors, condition, shutdown, graceful = _shutdown_owned_worker(
        process=process,
        stop_event=_FakeStopEvent(),
        ownership=registry,
        policy=_policy(),
        recorder=recorder,
        lifecycle=lifecycle,
        supervisor_started=started,
    )
    force_kill = {
        "status": "passed" if ok else "failed",
        "shutdown_elapsed_seconds": shutdown,
        "wall_elapsed_seconds": monotonic() - started,
        "graceful_stop_completed": graceful,
        "force_kill_called": registry.force_kill_called,
        "survivor_process_count": len(survivors),
        "termination_condition": condition,
        "lifecycle_states": [item["state"] for item in lifecycle],
    }
    after_children = [
        int(item.pid) for item in psutil.Process(os.getpid()).children(recursive=True)
    ]
    active_children = [
        int(item.pid) for item in multiprocessing.active_children() if item.is_alive()
    ]
    payload = {
        "schema_version": "cdv1_synthetic_stop_validation_v1",
        "policy_seconds": {
            "grace": 1.0,
            "terminate_wait": 0.2,
            "force_kill_wait": 0.2,
            "tested_supervisor_bound": 3.0,
        },
        "cooperative_stop": cooperative,
        "uncooperative_terminate": uncooperative,
        "nested_tree_terminate": nested,
        "force_kill_unit_path": force_kill,
        "saturated_queue": queue_case,
        "before_child_pids": before_children,
        "after_child_pids": after_children,
        "multiprocessing_active_child_pids": active_children,
        "no_orphans": not after_children and not active_children,
        "real_dataset_opened": False,
        "oot_opened": False,
    }
    checks = (
        cooperative["graceful_stop_completed"] is True,
        uncooperative["graceful_stop_completed"] is False,
        uncooperative["wall_elapsed_seconds"] < 3.0,
        nested["owned_process_count"] >= 2,
        nested["survivor_process_count"] == 0,
        force_kill["force_kill_called"] is True,
        force_kill["survivor_process_count"] == 0,
        queue_case["queue_cleanup_confirmed"] is True,
        payload["no_orphans"] is True,
    )
    payload["result"] = "pass" if all(checks) else "fail"
    temporary = AUDIT.with_name(AUDIT.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, AUDIT)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["result"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
