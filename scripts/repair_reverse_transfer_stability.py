from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from credit_risk_fs.repairs.reverse_transfer_stability import repair


if __name__ == "__main__":
    print(json.dumps(repair(), indent=2, default=str))
