from __future__ import annotations

import pandas as pd
import pytest

from scripts.run_clip_final_evaluation import _paired_auc_bootstrap


def test_paired_auc_bootstrap_requires_matching_prediction_counts():
    left = pd.DataFrame({"y_true": [0, 1], "y_pred_proba": [0.1, 0.9]})
    right = pd.DataFrame({"y_true": [0], "y_pred_proba": [0.2]})

    with pytest.raises(ValueError, match="prediction lengths"):
        _paired_auc_bootstrap(left, right)
