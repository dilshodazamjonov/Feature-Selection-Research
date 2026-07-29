"""Runtime-only RAM waiting policy and deterministic wait-state transitions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from credit_risk_fs.experiments.atomic_io import sha256_file
from credit_risk_fs.experiments.config import _parse_simple_yaml
from credit_risk_fs.experiments.resource_policy import GIB


RAM_CONTROL_SCHEMA_VERSION = "ram_wait_resume_policy_v1"
DEFAULT_RAM_CONTROL_PATH = Path("configs/execution/ram_wait_resume_v1.yaml")


@dataclass(frozen=True, slots=True)
class ResolvedRamControlPolicy:
    schema_version: str
    profile_name: str
    emergency_min_available_ram_bytes: int
    emergency_total_ram_fraction: float
    emergency_margin_bytes: int
    configured_recovery_threshold_bytes: int
    recovery_threshold_bytes: int
    recovery_consecutive_checks: int
    check_interval_seconds: float
    log_interval_seconds: float
    opaque_stage_pause_mode: str
    source_path: str
    source_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RamWaitTransition:
    action: str
    waiting_seconds: float
    consecutive_recovery_checks: int


class RamWaitState:
    """Pure state machine for emergency wait, stable recovery, and active time."""

    def __init__(self, policy: ResolvedRamControlPolicy) -> None:
        self.policy = policy
        self.waiting = False
        self.waiting_since: float | None = None
        self.last_log_at: float | None = None
        self.total_waiting_seconds = 0.0
        self.wait_count = 0
        self.consecutive_recovery_checks = 0

    def begin_wait(self, *, now: float) -> RamWaitTransition | None:
        """Enter a wait at a cooperative allocation/stage boundary."""

        if self.waiting:
            return None
        current = float(now)
        self.waiting = True
        self.waiting_since = current
        self.last_log_at = current
        self.wait_count += 1
        self.consecutive_recovery_checks = 0
        return RamWaitTransition("wait_started", 0.0, 0)

    def observe(self, available_bytes: int, *, now: float) -> RamWaitTransition | None:
        available = int(available_bytes)
        current = float(now)
        if not self.waiting:
            if available > self.policy.emergency_margin_bytes:
                return None
            return self.begin_wait(now=current)

        assert self.waiting_since is not None
        waiting_seconds = max(0.0, current - self.waiting_since)
        if available >= self.policy.recovery_threshold_bytes:
            self.consecutive_recovery_checks += 1
        else:
            self.consecutive_recovery_checks = 0
        if (
            self.consecutive_recovery_checks
            >= self.policy.recovery_consecutive_checks
        ):
            transition = RamWaitTransition(
                "resumed",
                waiting_seconds,
                self.consecutive_recovery_checks,
            )
            self.total_waiting_seconds += waiting_seconds
            self.waiting = False
            self.waiting_since = None
            self.last_log_at = None
            self.consecutive_recovery_checks = 0
            return transition

        if (
            self.last_log_at is not None
            and current - self.last_log_at >= self.policy.log_interval_seconds
        ):
            intervals = int(
                (current - self.last_log_at) // self.policy.log_interval_seconds
            )
            self.last_log_at += max(1, intervals) * self.policy.log_interval_seconds
            return RamWaitTransition(
                "wait_periodic",
                waiting_seconds,
                self.consecutive_recovery_checks,
            )
        return None

    def waiting_seconds(self, *, now: float) -> float:
        current = self.total_waiting_seconds
        if self.waiting and self.waiting_since is not None:
            current += max(0.0, float(now) - self.waiting_since)
        return current

    def active_seconds(self, *, started: float, now: float) -> float:
        return max(
            0.0,
            float(now) - float(started) - self.waiting_seconds(now=float(now)),
        )


def _positive_float(payload: Mapping[str, Any], key: str) -> float:
    try:
        value = float(payload[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"RAM-control field {key!r} must be numeric") from exc
    if value <= 0:
        raise ValueError(f"RAM-control field {key!r} must be positive")
    return value


def _positive_int(payload: Mapping[str, Any], key: str) -> int:
    value = _positive_float(payload, key)
    if not value.is_integer():
        raise ValueError(f"RAM-control field {key!r} must be an integer")
    return int(value)


def resolve_ram_control_policy(
    payload: Mapping[str, Any],
    *,
    total_physical_ram_bytes: int,
    source_path: str = "built-in",
    source_sha256: str = "built-in",
) -> ResolvedRamControlPolicy:
    if payload.get("schema_version") != RAM_CONTROL_SCHEMA_VERSION:
        raise ValueError("unsupported RAM-control policy schema")
    profile = str(payload.get("profile_name", "")).strip()
    if not profile:
        raise ValueError("RAM-control profile_name must not be empty")
    total = int(total_physical_ram_bytes)
    if total <= 0:
        raise ValueError("total physical RAM must be positive")
    minimum_bytes = int(
        _positive_float(payload, "emergency_min_available_ram_gb") * GIB
    )
    fraction = _positive_float(payload, "emergency_total_ram_fraction")
    if fraction >= 1:
        raise ValueError("emergency_total_ram_fraction must be below 1")
    emergency = max(minimum_bytes, int(total * fraction))
    configured_recovery = int(
        _positive_float(payload, "recovery_available_ram_gb") * GIB
    )
    recovery = max(configured_recovery, emergency)
    pause_mode = str(payload.get("opaque_stage_pause_mode", "")).strip()
    if pause_mode != "process_tree_suspend":
        raise ValueError("opaque_stage_pause_mode must be process_tree_suspend")
    return ResolvedRamControlPolicy(
        schema_version=RAM_CONTROL_SCHEMA_VERSION,
        profile_name=profile,
        emergency_min_available_ram_bytes=minimum_bytes,
        emergency_total_ram_fraction=fraction,
        emergency_margin_bytes=emergency,
        configured_recovery_threshold_bytes=configured_recovery,
        recovery_threshold_bytes=recovery,
        recovery_consecutive_checks=_positive_int(
            payload, "recovery_consecutive_checks"
        ),
        check_interval_seconds=_positive_float(payload, "check_interval_seconds"),
        log_interval_seconds=_positive_float(payload, "log_interval_seconds"),
        opaque_stage_pause_mode=pause_mode,
        source_path=str(source_path),
        source_sha256=str(source_sha256),
    )


def load_ram_control_policy(
    repository_root: str | Path,
    path: str | Path = DEFAULT_RAM_CONTROL_PATH,
    *,
    total_physical_ram_bytes: int | None = None,
) -> ResolvedRamControlPolicy:
    root = Path(repository_root).resolve()
    supplied = Path(path)
    resolved_path = (
        supplied.resolve() if supplied.is_absolute() else (root / supplied).resolve()
    )
    if not resolved_path.is_file():
        raise FileNotFoundError(f"RAM-control policy is missing: {resolved_path}")
    payload = _parse_simple_yaml(resolved_path.read_text(encoding="utf-8"))
    if total_physical_ram_bytes is None:
        import psutil

        total_physical_ram_bytes = int(psutil.virtual_memory().total)
    return resolve_ram_control_policy(
        payload,
        total_physical_ram_bytes=int(total_physical_ram_bytes),
        source_path=str(resolved_path),
        source_sha256=sha256_file(resolved_path),
    )


def default_ram_control_policy(
    *, total_physical_ram_bytes: int
) -> ResolvedRamControlPolicy:
    return resolve_ram_control_policy(
        {
            "schema_version": RAM_CONTROL_SCHEMA_VERSION,
            "profile_name": "ram_wait_resume_default",
            "emergency_min_available_ram_gb": 1,
            "emergency_total_ram_fraction": 0.02,
            "recovery_available_ram_gb": 4,
            "recovery_consecutive_checks": 3,
            "check_interval_seconds": 5,
            "log_interval_seconds": 300,
            "opaque_stage_pause_mode": "process_tree_suspend",
        },
        total_physical_ram_bytes=total_physical_ram_bytes,
    )


def wait_for_ram_ready(
    ready_event: Any | None,
    stop_event: Any | None,
    *,
    boundary: str,
    poll_seconds: float = 1.0,
) -> None:
    """Worker-side cooperative gate used between chunks and large allocations."""

    if ready_event is None:
        return
    while not ready_event.is_set():
        if stop_event is not None and stop_event.is_set():
            raise RuntimeError(
                f"cooperative stop requested while waiting for RAM at {boundary}"
            )
        ready_event.wait(max(0.01, float(poll_seconds)))


__all__ = [
    "DEFAULT_RAM_CONTROL_PATH",
    "RAM_CONTROL_SCHEMA_VERSION",
    "RamWaitState",
    "RamWaitTransition",
    "ResolvedRamControlPolicy",
    "default_ram_control_policy",
    "load_ram_control_policy",
    "resolve_ram_control_policy",
    "wait_for_ram_ready",
]
