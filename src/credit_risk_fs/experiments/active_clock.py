"""Platform clock for active supervisor time that excludes system suspension."""

from __future__ import annotations

import os
from collections.abc import Callable
from time import monotonic as _fallback_monotonic


WINDOWS_TICKS_PER_SECOND = 10_000_000.0


def _windows_unbiased_interrupt_ticks() -> int:
    """Return Windows awake time in 100 ns units, excluding sleep/hibernate."""

    import ctypes

    ticks = ctypes.c_ulonglong()
    query = ctypes.windll.kernel32.QueryUnbiasedInterruptTime
    query.argtypes = [ctypes.POINTER(ctypes.c_ulonglong)]
    query.restype = ctypes.c_ubyte
    if not query(ctypes.byref(ticks)):
        raise ctypes.WinError(ctypes.get_last_error())
    return int(ticks.value)


def resolve_awake_monotonic(
    *,
    platform_name: str,
    windows_tick_reader: Callable[[], int] = _windows_unbiased_interrupt_ticks,
    fallback_clock: Callable[[], float] = _fallback_monotonic,
) -> tuple[Callable[[], float], str]:
    """Resolve a monotonic awake-time clock and its auditable identity."""

    if platform_name == "nt":
        def windows_awake_seconds() -> float:
            return float(windows_tick_reader()) / WINDOWS_TICKS_PER_SECOND

        return windows_awake_seconds, "windows_query_unbiased_interrupt_time_v1"
    return fallback_clock, "platform_monotonic_v1"


awake_monotonic, ACTIVE_CLOCK_SOURCE = resolve_awake_monotonic(
    platform_name=os.name
)


__all__ = [
    "ACTIVE_CLOCK_SOURCE",
    "WINDOWS_TICKS_PER_SECOND",
    "awake_monotonic",
    "resolve_awake_monotonic",
]
