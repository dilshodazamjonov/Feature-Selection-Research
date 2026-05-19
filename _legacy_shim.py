from __future__ import annotations

import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def reexport(module_name: str, target_globals: dict[str, object]) -> list[str]:
    module = importlib.import_module(module_name)
    public_names = getattr(
        module,
        "__all__",
        [name for name in dir(module) if not name.startswith("_")],
    )
    for name in public_names:
        target_globals[name] = getattr(module, name)
    return list(public_names)
