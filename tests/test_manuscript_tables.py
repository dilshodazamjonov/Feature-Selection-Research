"""The published ``results/`` CSVs reproduce the manuscript (schema and values).

Runs ``scripts/verify_manuscript_tables.py --schema`` and ``--check``.  A manuscript value whose CSV cell is
empty is a documented gap (``docs/repository_freeze.md``), not a failure; a value that disagrees with the
manuscript at the manuscript's precision is a failure.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_manuscript_tables.py"
FOLDERS = [ROOT / "results" / name for name in ("homecredit_metrics_output", "lendingclub_metrics_output", "stability_metrics_output")]


def _run(flag: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), flag], cwd=ROOT, capture_output=True, text=True)


@pytest.fixture(scope="module", autouse=True)
def _results_present():
    missing = [str(p) for p in FOLDERS if not p.is_dir()]
    if missing:
        pytest.skip(f"published results folders missing: {missing}")


def test_schema_is_uniform():
    proc = _run("--schema")
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_values_match_manuscript():
    proc = _run("--check")
    summary = [line for line in proc.stdout.splitlines() if line.startswith("CHECK:")]
    assert summary, proc.stdout + proc.stderr
    mismatches = [line for line in proc.stdout.splitlines() if line.startswith("MISMATCH")]
    assert proc.returncode == 0 and not mismatches, "\n".join(mismatches) + "\n" + summary[-1]
    # Report the documented gaps without failing.
    blanks = int(summary[-1].split(",")[2].strip().split()[0])
    if blanks:
        print(f"{blanks} manuscript values have no CSV source - see docs/repository_freeze.md")
