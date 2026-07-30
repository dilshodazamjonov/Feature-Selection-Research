from __future__ import annotations

import os

from credit_risk_fs.experiments.active_clock import (
    ACTIVE_CLOCK_SOURCE,
    awake_monotonic,
    resolve_awake_monotonic,
)


def test_windows_awake_clock_uses_unbiased_interrupt_time():
    ticks = iter((100_000_000, 110_000_000))
    clock, source = resolve_awake_monotonic(
        platform_name="nt",
        windows_tick_reader=lambda: next(ticks),
        fallback_clock=lambda: 9999.0,
    )

    assert source == "windows_query_unbiased_interrupt_time_v1"
    assert clock() == 10.0
    assert clock() == 11.0


def test_live_clock_is_monotonic_and_windows_is_sleep_excluding():
    before = awake_monotonic()
    after = awake_monotonic()

    assert after >= before
    if os.name == "nt":
        assert ACTIVE_CLOCK_SOURCE == "windows_query_unbiased_interrupt_time_v1"
