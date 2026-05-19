from __future__ import annotations

from pathlib import Path

import joblib


def dump_joblib(obj, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, output)
    return output


def load_joblib(path: str | Path):
    return joblib.load(Path(path))
