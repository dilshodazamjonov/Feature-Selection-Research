import pytest

from evaluation.metrics import determine_threshold, evaluate_model


def test_evaluate_model_uses_training_threshold_for_thresholded_metrics():
    y_train = [0, 0, 1, 1]
    train_proba = [0.1, 0.3, 0.8, 0.9]
    threshold = determine_threshold(y_train, train_proba)

    y_val = [0, 1, 1, 0]
    val_proba = [0.2, 0.82, 0.9, 0.81]
    metrics = evaluate_model(y_val, val_proba, threshold=threshold)

    assert metrics["decision_threshold"] == pytest.approx(threshold)
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(0.8)
