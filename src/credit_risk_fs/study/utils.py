"""I/O, timing and resource helpers for the study runner."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def log(message: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", file=sys.stdout, flush=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path: str | Path, frame: pd.DataFrame, columns: list[str] | None = None) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    output = frame if columns is None else frame.reindex(columns=columns)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    output.to_csv(temporary, index=False, lineterminator="\n", encoding="utf-8")
    temporary.replace(target)


def save_array(path: str | Path, values: np.ndarray) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp.npy")
    np.save(temporary, np.asarray(values), allow_pickle=False)
    temporary.replace(target)


def load_array(path: str | Path) -> np.ndarray:
    return np.load(Path(path), allow_pickle=False)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ResourceMonitor:
    """Wall-clock time and the peak resident memory observed while the block runs."""

    def __init__(self, interval_seconds: float = 0.5) -> None:
        self.interval_seconds = float(interval_seconds)
        self.seconds = 0.0
        self.peak_rss_bytes = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = 0.0

    def _rss(self) -> int:
        try:
            import psutil

            return int(psutil.Process().memory_info().rss)
        except Exception:  # pragma: no cover - psutil missing or process query failed
            return 0

    def _poll(self) -> None:
        while not self._stop.is_set():
            self.peak_rss_bytes = max(self.peak_rss_bytes, self._rss())
            self._stop.wait(self.interval_seconds)

    def __enter__(self) -> "ResourceMonitor":
        self._started = time.perf_counter()
        self.peak_rss_bytes = self._rss()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.peak_rss_bytes = max(self.peak_rss_bytes, self._rss())
        self.seconds = time.perf_counter() - self._started


__all__ = [
    "ResourceMonitor",
    "ensure_dir",
    "load_array",
    "log",
    "read_json",
    "save_array",
    "sha256_file",
    "sha256_text",
    "utc_now",
    "write_csv",
    "write_json",
]
