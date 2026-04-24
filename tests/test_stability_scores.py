import pandas as pd

from evaluation.stability_scores import calculate_psi


def test_calculate_psi_detects_shift_outside_training_range():
    expected = pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=float)
    actual = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109], dtype=float)

    psi = calculate_psi(expected, actual)

    assert psi > 0.1
