from __future__ import annotations

from pathlib import Path
import pkgutil


__path__ = pkgutil.extend_path(__path__, __name__)

SRC_PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "src" / "credit_risk_fs"
if SRC_PACKAGE_ROOT.exists():
    src_path = str(SRC_PACKAGE_ROOT)
    if src_path not in __path__:
        __path__.append(src_path)

from .data.dataset_registry import get_dataset_config  # noqa: E402

__all__ = ["get_dataset_config"]
