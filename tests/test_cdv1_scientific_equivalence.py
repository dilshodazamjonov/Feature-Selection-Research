from __future__ import annotations

import json
from pathlib import Path

from tests.support.cdv1_scientific_equivalence_probe import build_probe


ROOT = Path(__file__).resolve().parents[1]


def test_mechanics_patch_preserves_exact_synthetic_scientific_outputs():
    expected = json.loads(
        (ROOT / "tests/fixtures/cdv1_scientific_equivalence_golden.json").read_text(
            encoding="utf-8"
        )
    )
    assert build_probe(ROOT) == expected
