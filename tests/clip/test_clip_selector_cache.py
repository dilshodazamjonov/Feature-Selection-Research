from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from credit_risk_fs.clip.selector_adapter import ClipScoreAdapter, materialize_score_caches


def _tmp_selector_config(tmp_path: Path) -> Path:
    text = Path("configs/clip/selector.yaml").read_text(encoding="utf-8")
    text = text.replace(
        "output_dir: results/clip/selector_integration",
        f"output_dir: {tmp_path.as_posix()}/selector_integration",
    )
    text = text.replace(
        "cache_dir: results/clip/selector_integration/cache",
        f"cache_dir: {tmp_path.as_posix()}/selector_integration/cache",
    )
    path = tmp_path / "selector.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_score_cache_materializes_with_reproducible_keys(tmp_path):
    config_path = _tmp_selector_config(tmp_path)
    paths = materialize_score_caches(config_path)

    home = pd.read_csv(paths["homecredit"])
    lc = pd.read_csv(paths["lendingclub_v2"])

    assert len(home) == 436
    assert len(lc) == 576
    assert home["score_cache_key"].is_unique
    assert lc["score_cache_key"].is_unique
    assert set(home["statistical_view_scope"]) == {"missingness_only"}
    assert set(lc["statistical_view_scope"]) == {"missingness_only"}


def test_stale_score_cache_is_rejected(tmp_path):
    config_path = _tmp_selector_config(tmp_path)
    paths = materialize_score_caches(config_path)
    cache = pd.read_csv(paths["homecredit"])
    cache.loc[0, "checkpoint_hash"] = "stale"
    cache.to_csv(paths["homecredit"], index=False)

    with pytest.raises(RuntimeError, match="checkpoint hash mismatch"):
        ClipScoreAdapter(config_path, dataset="homecredit").score_frame(use_cache=True)
