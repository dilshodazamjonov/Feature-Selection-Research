from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_text(path: str | Path, *, encoding: str = "utf-8") -> str:
    return Path(path).read_text(encoding=encoding)


def write_text(path: str | Path, content: str, *, encoding: str = "utf-8") -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding=encoding)
    return output


def read_json(path: str | Path) -> Any:
    return json.loads(read_text(path))


def write_json(path: str | Path, payload: Any) -> Path:
    return write_text(path, json.dumps(payload, indent=2, ensure_ascii=False, default=str))
