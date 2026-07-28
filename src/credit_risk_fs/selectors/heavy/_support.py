"""Shared plumbing for heavy selectors: estimator hashing and stage logging.

Nothing here changes an algorithm. It exists so the three heavy selectors declare
their estimator configuration and emit stage boundaries the same way, using the
repository's existing logger rather than a new one.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from credit_risk_fs.utils.hashing import sha256_text


def estimator_config_hash(configuration: Mapping[str, Any]) -> str:
    """Stable hash of the estimator configuration actually handed to the model."""

    return sha256_text(
        json.dumps(dict(configuration), sort_keys=True, separators=(",", ":"), default=str)
    )


def ordered_rows_hash(values: Any) -> str:
    """Hash an ordered row-identity sequence (explanation samples, fold rows)."""

    return sha256_text(
        json.dumps([str(value) for value in values], separators=(",", ":"))
    )


@contextmanager
def heavy_stage(
    logger: logging.Logger,
    *,
    method_id: str,
    stage: str,
    detail: str = "",
) -> Iterator[dict[str, Any]]:
    """Emit START / DONE / ERROR boundaries for one heavy stage.

    Uses the module logger so the existing research-run handlers decide where the
    line lands; concise progress goes to the run log and full tracebacks stay in
    the debug log. The context yields a mutable dict so a stage can attach counts
    that are only known once it finishes.

    This wrapper must never influence the algorithm: it does not seed, sample,
    reorder, or retry anything.
    """

    observations: dict[str, Any] = {}
    started = time.perf_counter()
    suffix = f" | {detail}" if detail else ""
    logger.info("START  | %s | %s%s", method_id, stage, suffix)
    try:
        yield observations
    except Exception as error:
        logger.info(
            "ERROR  | %s | %s | %.2fs | %s",
            method_id,
            stage,
            time.perf_counter() - started,
            type(error).__name__,
        )
        # The full traceback goes to the debug logger, not the concise run log.
        logger.debug("heavy stage failed: %s / %s", method_id, stage, exc_info=True)
        raise
    elapsed = time.perf_counter() - started
    observations["elapsed_seconds"] = round(elapsed, 6)
    counts = " ".join(
        f"{key}={value}"
        for key, value in observations.items()
        if key != "elapsed_seconds"
    )
    logger.info(
        "DONE   | %s | %s | %.2fs%s",
        method_id,
        stage,
        elapsed,
        f" | {counts}" if counts else "",
    )


def process_rss_bytes() -> int | None:
    """Current resident set size, when the existing psutil dependency is present."""

    try:
        import os

        import psutil
    except ImportError:  # pragma: no cover - psutil is a pinned dependency
        return None
    return int(psutil.Process(os.getpid()).memory_info().rss)


def available_ram_bytes() -> int | None:
    try:
        import psutil
    except ImportError:  # pragma: no cover
        return None
    return int(psutil.virtual_memory().available)


__all__ = [
    "available_ram_bytes",
    "estimator_config_hash",
    "heavy_stage",
    "ordered_rows_hash",
    "process_rss_bytes",
]
