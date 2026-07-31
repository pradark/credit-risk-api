"""Tests for credit risk model performance metrics."""

import pandas as pd
import pytest

from app.monitoring.performance_metrics import (
    calculate_auc,
    calculate_average_predicted_pd,
    calculate_bad_rate,
    calculate_calibration_error,
    calculate_gini,
    calculate_ks,
    calculate_performance_metrics,
    calculate_precision,
    calculate_recall,
    validate_performance_data,
    validate_threshold,
)


@pytest.fixture
def performance_data() -> pd.DataFrame:
    """Return a small model performance dataset."""
    return pd.DataFrame(
        {
            "application_id": [
                1001,
                1002,
                1003,
                1004,
                1005,
                1006,
                1007,
                1008,
                1009,
                1010,
            ],
            "predicted_probability": [
                0.05,
                0.10,
                0.15,
                0.20,
                0.30,
                0.45,
                0.55,
                0.65,
                0.80,
                0.90,
            ],
            "actual_default": [
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                1,
                1,
                1,
            ],
        }
    )


def test_validate_performance_data_accepts_valid_data(
    performance_data: pd.DataFrame,
) -> None:
    """Valid performance data should not raise an exception."""
    validate_performance_data(performance_data)


def test_validate_performance_data_rejects_missing_columns() -> None:
    """Missing required columns should raise a ValueError."""
    data = pd.DataFrame({"actual_default": [0, 1]})

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_performance_data(data)


def test_validate_performance_data_rejects_invalid_actual_values() -> None:
    """Actual outcomes must contain only zero and one."""
    data = pd.DataFrame(
        {
            "actual_default": [0, 2],
            "predicted_probability": [0.10, 0.80],
        }
    )

    with pytest.raises(ValueError, match="must contain only 0 and 1"):
        validate_performance_data(data)


def test_validate_performance_data_rejects_invalid_probabilities() -> None:
    """Predicted probabilities must be between zero and one."""
    data = pd.DataFrame(
        {
            "actual_default": [0, 1],
            "predicted_probability": [0.10, 1.20],
        }
    )

    with pytest.raises(ValueError, match="between 0 and 1"):
        validate_performance_data(data)


def test_validate_performance_data_requires_both_classes() -> None:
    """AUC and KS require defaults and non-defaults."""
    data = pd.DataFrame(
        {
            "actual_default": [0, 0, 0],
            "predicted_probability": [0.10, 0.20, 0.30],
        }
    )

    with pytest.raises(
        ValueError,
        match="Both default and non-default outcomes are required",
    ):
        validate_performance_data(data)


def test_validate_threshold_rejects_invalid_threshold() -> None:
    """Classification threshold must be between zero and one."""
    with pytest.raises(ValueError, match="must be between 0 and 1"):
        validate_threshold(1.10)


def test_calculate_auc(performance_data: pd.DataFrame) -> None:
    """AUC should match the expected ranking performance."""
    result = calculate_auc(performance_data)

    assert result == pytest.approx(0.9583333333)


def test_calculate_ks(performance_data: pd.DataFrame) -> None:
    """KS should match the expected maximum separation."""
    result = calculate_ks(performance_data)

    assert result == pytest.approx(0.8333333333)


def test_calculate_gini() -> None:
    """Gini should equal two times AUC minus one."""
    result = calculate_gini(0.75)

    assert result == pytest.approx(0.50)


def test_calculate_bad_rate(performance_data: pd.DataFrame) -> None:
    """Bad rate should equal the observed default rate."""
    result = calculate_bad_rate(performance_data)

    assert result == pytest.approx(0.40)


def test_calculate_average_predicted_pd(
    performance_data: pd.DataFrame,
) -> None:
    """Average predicted PD should equal the mean prediction."""
    result = calculate_average_predicted_pd(performance_data)

    assert result == pytest.approx(0.415)


def test_calculate_precision(performance_data: pd.DataFrame) -> None:
    """Precision should be calculated at the selected threshold."""
    result = calculate_precision(
        performance_data,
        threshold=0.50,
    )

    assert result == pytest.approx(0.75)


def test_calculate_recall(performance_data: pd.DataFrame) -> None:
    """Recall should be calculated at the selected threshold."""
    result = calculate_recall(
        performance_data,
        threshold=0.50,
    )

    assert result == pytest.approx(0.75)


def test_calculate_calibration_error(
    performance_data: pd.DataFrame,
) -> None:
    """Calibration error should compare mean PD with bad rate."""
    result = calculate_calibration_error(performance_data)

    assert result == pytest.approx(0.015)


def test_calculate_performance_metrics(
    performance_data: pd.DataFrame,
) -> None:
    """The combined function should return one complete metric row."""
    result = calculate_performance_metrics(
        performance_data,
        threshold=0.50,
    )

    assert len(result) == 1
    assert result.loc[0, "record_count"] == 10
    assert result.loc[0, "default_count"] == 4
    assert result.loc[0, "non_default_count"] == 6
    assert result.loc[0, "auc"] == pytest.approx(0.9583)
    assert result.loc[0, "ks"] == pytest.approx(0.8333)
    assert result.loc[0, "gini"] == pytest.approx(0.9167)
    assert result.loc[0, "bad_rate"] == pytest.approx(0.40)
    assert result.loc[0, "average_predicted_pd"] == pytest.approx(0.415)
    assert result.loc[0, "precision"] == pytest.approx(0.75)
    assert result.loc[0, "recall"] == pytest.approx(0.75)
    assert result.loc[0, "calibration_error"] == pytest.approx(0.015)
    assert result.loc[0, "classification_threshold"] == pytest.approx(0.50)