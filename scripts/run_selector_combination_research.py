#!/usr/bin/env python
"""Canonical Prompt 11 selector-combination research entry point."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from credit_risk_fs.experiments.selector_combinations import main


if __name__ == "__main__":
    raise SystemExit(main())
