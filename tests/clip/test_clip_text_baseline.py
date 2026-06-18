from __future__ import annotations

from pathlib import Path

import pandas as pd

from credit_risk_fs.clip.text_baseline import build_text_baseline, load_text_baseline_config
from credit_risk_fs.clip.text_encoder import MockFrozenTextEncoder


def test_dry_run_performs_no_model_loading(tmp_path):
    config = load_text_baseline_config()
    config = config.__class__(**{**config.__dict__, "output_dir": tmp_path})

    result = build_text_baseline(config=config, dry_run=True)

    assert result.summary["encoder_loaded"] is False
    assert result.summary["model_trained"] is False
    assert result.summary["expected_embedding_count"] == {"homecredit": 436, "lendingclub_v2": 576}


def test_text_baseline_with_mock_encoder_uses_homecredit_anchor_unchanged(tmp_path, monkeypatch):
    config = load_text_baseline_config()
    config = config.__class__(**{**config.__dict__, "output_dir": tmp_path})

    def fake_save(frame: pd.DataFrame, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out.with_suffix(".csv"), index=False)
        return out

    monkeypatch.setattr("credit_risk_fs.clip.text_baseline.save_embedding_frame", fake_save)
    result = build_text_baseline(config=config, dry_run=False, encoder=MockFrozenTextEncoder())

    home_rank = pd.read_csv(result.output_paths["homecredit_text_only_ranking"])
    lc_rank = pd.read_csv(result.output_paths["lendingclub_v2_text_only_ranking"])
    anchors = pd.read_csv(result.output_paths["homecredit_anchor_features"])

    assert set(anchors["dataset"]) == {"homecredit"}
    assert home_rank["is_anchor_feature"].any()
    assert not lc_rank["is_anchor_feature"].any()
    assert set(lc_rank["dataset"]) == {"lendingclub_v2"}
    assert result.summary["model_trained"] is False
    assert result.summary["contrastive_pairs_created"] is False


def test_legacy_lendingclub_is_rejected(tmp_path):
    config = load_text_baseline_config()
    bad = config.__class__(**{**config.__dict__, "external_validation_dataset": "lendingclub", "output_dir": tmp_path})

    try:
        build_text_baseline(config=bad, dry_run=True)
    except RuntimeError as exc:
        assert "lendingclub_v2" in str(exc) or "legacy" in str(exc)
    else:
        raise AssertionError("legacy LendingClub config should fail")
