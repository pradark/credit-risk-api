"""Tests for credit risk model performance metrics."""

import pandas as pd
import pytest

from app.monitoring.performance_metrics import (
    build_predicted_class,
    calculate_accuracy,
    calculate_auc,
    calculate_average_predicted_pd,
    calculate_bad_rate,
    calculate_balanced_accuracy,
    calculate_brier_score,
    calculate_calibration_error,
    calculate_confusion_matrix_counts,
    calculate_f1,
    calculate_false_negative_rate,
    calculate_false_positive_rate,
    calculate_gini,
    calculate_ks,
    calculate_log_loss,
    calculate_performance_metrics,
    calculate_precision,
    calculate_predicted_positive_rate,
    calculate_prediction_standard_deviation,
    calculate_recall,
    calculate_specificity,
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

    validate_performance_data(
        performance_data
    )


def test_validate_performance_data_rejects_non_dataframe() -> None:
    """Performance data must be a pandas DataFrame."""

    with pytest.raises(
        TypeError,
        match="must be a pandas DataFrame",
    ):
        validate_performance_data(
            []
        )


def test_validate_performance_data_rejects_empty_data() -> None:
    """Empty performance data should be rejected."""

    with pytest.raises(
        ValueError,
        match="Performance data is empty",
    ):
        validate_performance_data(
            pd.DataFrame()
        )


def test_validate_performance_data_rejects_missing_columns() -> None:
    """Missing required columns should raise a ValueError."""

    data = pd.DataFrame(
        {
            "actual_default": [
                0,
                1,
            ]
        }
    )

    with pytest.raises(
        ValueError,
        match="Missing required columns",
    ):
        validate_performance_data(
            data
        )


def test_validate_performance_data_rejects_missing_actual_values() -> None:
    """Actual outcomes cannot contain missing values."""

    data = pd.DataFrame(
        {
            "actual_default": [
                0,
                None,
            ],
            "predicted_probability": [
                0.10,
                0.80,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="actual_default contains missing values",
    ):
        validate_performance_data(
            data
        )


def test_validate_performance_data_rejects_missing_probabilities() -> None:
    """Predicted probabilities cannot contain missing values."""

    data = pd.DataFrame(
        {
            "actual_default": [
                0,
                1,
            ],
            "predicted_probability": [
                0.10,
                None,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="predicted_probability contains missing values",
    ):
        validate_performance_data(
            data
        )


def test_validate_performance_data_rejects_invalid_actual_values() -> None:
    """Actual outcomes must contain only zero and one."""

    data = pd.DataFrame(
        {
            "actual_default": [
                0,
                2,
            ],
            "predicted_probability": [
                0.10,
                0.80,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="must contain only 0 and 1",
    ):
        validate_performance_data(
            data
        )


def test_validate_performance_data_rejects_invalid_probabilities() -> None:
    """Predicted probabilities must be between zero and one."""

    data = pd.DataFrame(
        {
            "actual_default": [
                0,
                1,
            ],
            "predicted_probability": [
                0.10,
                1.20,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        validate_performance_data(
            data
        )


def test_validate_performance_data_requires_both_classes() -> None:
    """AUC and KS require defaults and non-defaults."""

    data = pd.DataFrame(
        {
            "actual_default": [
                0,
                0,
                0,
            ],
            "predicted_probability": [
                0.10,
                0.20,
                0.30,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match=(
            "Both default and non-default "
            "outcomes are required"
        ),
    ):
        validate_performance_data(
            data
        )


@pytest.mark.parametrize(
    "threshold",
    [
        -0.01,
        1.01,
    ],
)
def test_validate_threshold_rejects_invalid_threshold(
    threshold: float,
) -> None:
    """Classification threshold must be between zero and one."""

    with pytest.raises(
        ValueError,
        match="must be between 0 and 1",
    ):
        validate_threshold(
            threshold
        )


def test_validate_threshold_rejects_non_numeric_value() -> None:
    """Classification threshold must be numeric."""

    with pytest.raises(
        TypeError,
        match="must be numeric",
    ):
        validate_threshold(
            "0.50"
        )


def test_build_predicted_class(
    performance_data: pd.DataFrame,
) -> None:
    """Probabilities should be converted using the supplied threshold."""

    result = build_predicted_class(
        performance_data,
        threshold=0.50,
    )

    assert result.tolist() == [
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
    ]


def test_build_predicted_class_uses_custom_threshold(
    performance_data: pd.DataFrame,
) -> None:
    """A custom threshold should change binary predictions."""

    result = build_predicted_class(
        performance_data,
        threshold=0.80,
    )

    assert result.tolist() == [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
    ]


def test_calculate_auc(
    performance_data: pd.DataFrame,
) -> None:
    """AUC should match the expected ranking performance."""

    result = calculate_auc(
        performance_data
    )

    assert result == pytest.approx(
        0.9583333333
    )


def test_calculate_ks(
    performance_data: pd.DataFrame,
) -> None:
    """KS should match the expected maximum separation."""

    result = calculate_ks(
        performance_data
    )

    assert result == pytest.approx(
        0.8333333333
    )


def test_calculate_gini() -> None:
    """Gini should equal two times AUC minus one."""

    result = calculate_gini(
        0.75
    )

    assert result == pytest.approx(
        0.50
    )


@pytest.mark.parametrize(
    "auc",
    [
        -0.01,
        1.01,
    ],
)
def test_calculate_gini_rejects_invalid_auc(
    auc: float,
) -> None:
    """AUC used for Gini must be between zero and one."""

    with pytest.raises(
        ValueError,
        match="AUC must be between 0 and 1",
    ):
        calculate_gini(
            auc
        )


def test_calculate_bad_rate(
    performance_data: pd.DataFrame,
) -> None:
    """Bad rate should equal the observed default rate."""

    result = calculate_bad_rate(
        performance_data
    )

    assert result == pytest.approx(
        0.40
    )


def test_calculate_average_predicted_pd(
    performance_data: pd.DataFrame,
) -> None:
    """Average predicted PD should equal the mean prediction."""

    result = calculate_average_predicted_pd(
        performance_data
    )

    assert result == pytest.approx(
        0.415
    )


def test_calculate_prediction_standard_deviation(
    performance_data: pd.DataFrame,
) -> None:
    """Prediction standard deviation should use population variance."""

    result = calculate_prediction_standard_deviation(
        performance_data
    )

    assert result == pytest.approx(
        0.2864000698
    )


def test_calculate_precision(
    performance_data: pd.DataFrame,
) -> None:
    """Precision should be calculated at the selected threshold."""

    result = calculate_precision(
        performance_data,
        threshold=0.50,
    )

    assert result == pytest.approx(
        0.75
    )


def test_calculate_recall(
    performance_data: pd.DataFrame,
) -> None:
    """Recall should be calculated at the selected threshold."""

    result = calculate_recall(
        performance_data,
        threshold=0.50,
    )

    assert result == pytest.approx(
        0.75
    )


def test_calculate_accuracy(
    performance_data: pd.DataFrame,
) -> None:
    """Accuracy should equal correctly classified records over total."""

    result = calculate_accuracy(
        performance_data,
        threshold=0.50,
    )

    assert result == pytest.approx(
        0.80
    )


def test_calculate_balanced_accuracy(
    performance_data: pd.DataFrame,
) -> None:
    """Balanced accuracy should average recall and specificity."""

    result = calculate_balanced_accuracy(
        performance_data,
        threshold=0.50,
    )

    assert result == pytest.approx(
        0.7916666667
    )


def test_calculate_f1(
    performance_data: pd.DataFrame,
) -> None:
    """F1 should balance precision and recall."""

    result = calculate_f1(
        performance_data,
        threshold=0.50,
    )

    assert result == pytest.approx(
        0.75
    )


def test_calculate_confusion_matrix_counts(
    performance_data: pd.DataFrame,
) -> None:
    """Confusion-matrix counts should match known predictions."""

    result = calculate_confusion_matrix_counts(
        performance_data,
        threshold=0.50,
    )

    assert result == {
        "true_positive": 3,
        "false_positive": 1,
        "true_negative": 5,
        "false_negative": 1,
    }


def test_calculate_specificity(
    performance_data: pd.DataFrame,
) -> None:
    """Specificity should measure correctly classified non-defaults."""

    result = calculate_specificity(
        performance_data,
        threshold=0.50,
    )

    assert result == pytest.approx(
        5 / 6
    )


def test_calculate_false_positive_rate(
    performance_data: pd.DataFrame,
) -> None:
    """False-positive rate should equal one minus specificity."""

    result = calculate_false_positive_rate(
        performance_data,
        threshold=0.50,
    )

    assert result == pytest.approx(
        1 / 6
    )


def test_calculate_false_negative_rate(
    performance_data: pd.DataFrame,
) -> None:
    """False-negative rate should equal one minus recall."""

    result = calculate_false_negative_rate(
        performance_data,
        threshold=0.50,
    )

    assert result == pytest.approx(
        0.25
    )


def test_calculate_predicted_positive_rate(
    performance_data: pd.DataFrame,
) -> None:
    """Predicted-positive rate should equal positive predictions over total."""

    result = calculate_predicted_positive_rate(
        performance_data,
        threshold=0.50,
    )

    assert result == pytest.approx(
        0.40
    )


def test_calculate_brier_score(
    performance_data: pd.DataFrame,
) -> None:
    """Brier score should measure squared probability error."""

    result = calculate_brier_score(
        performance_data
    )

    assert result == pytest.approx(
        0.09425
    )


def test_calculate_log_loss(
    performance_data: pd.DataFrame,
) -> None:
    """Log loss should measure probabilistic classification error."""

    result = calculate_log_loss(
        performance_data
    )

    assert result == pytest.approx(
        0.3255293610
    )


def test_calculate_calibration_error(
    performance_data: pd.DataFrame,
) -> None:
    """Calibration error should compare mean PD with bad rate."""

    result = calculate_calibration_error(
        performance_data
    )

    assert result == pytest.approx(
        0.015
    )


def test_metrics_return_zero_when_no_positive_predictions(
    performance_data: pd.DataFrame,
) -> None:
    """Precision, recall, and F1 should handle no positive predictions."""

    threshold = 1.0

    assert calculate_precision(
        performance_data,
        threshold=threshold,
    ) == pytest.approx(
        0.0
    )

    assert calculate_recall(
        performance_data,
        threshold=threshold,
    ) == pytest.approx(
        0.0
    )

    assert calculate_f1(
        performance_data,
        threshold=threshold,
    ) == pytest.approx(
        0.0
    )


def test_calculate_performance_metrics(
    performance_data: pd.DataFrame,
) -> None:
    """The combined function should return one complete metric row."""

    result = calculate_performance_metrics(
        performance_data,
        threshold=0.50,
    )

    assert len(
        result
    ) == 1

    assert result.loc[
        0,
        "record_count",
    ] == 10

    assert result.loc[
        0,
        "default_count",
    ] == 4

    assert result.loc[
        0,
        "non_default_count",
    ] == 6

    assert result.loc[
        0,
        "auc",
    ] == pytest.approx(
        0.9583
    )

    assert result.loc[
        0,
        "ks",
    ] == pytest.approx(
        0.8333
    )

    assert result.loc[
        0,
        "gini",
    ] == pytest.approx(
        0.9167
    )

    assert result.loc[
        0,
        "bad_rate",
    ] == pytest.approx(
        0.40
    )

    assert result.loc[
        0,
        "average_predicted_pd",
    ] == pytest.approx(
        0.415
    )

    assert result.loc[
        0,
        "prediction_standard_deviation",
    ] == pytest.approx(
        0.2864
    )

    assert result.loc[
        0,
        "minimum_predicted_pd",
    ] == pytest.approx(
        0.05
    )

    assert result.loc[
        0,
        "maximum_predicted_pd",
    ] == pytest.approx(
        0.90
    )

    assert result.loc[
        0,
        "accuracy",
    ] == pytest.approx(
        0.80
    )

    assert result.loc[
        0,
        "balanced_accuracy",
    ] == pytest.approx(
        0.7917
    )

    assert result.loc[
        0,
        "precision",
    ] == pytest.approx(
        0.75
    )

    assert result.loc[
        0,
        "recall",
    ] == pytest.approx(
        0.75
    )

    assert result.loc[
        0,
        "specificity",
    ] == pytest.approx(
        0.8333
    )

    assert result.loc[
        0,
        "f1",
    ] == pytest.approx(
        0.75
    )

    assert result.loc[
        0,
        "false_positive_rate",
    ] == pytest.approx(
        0.1667
    )

    assert result.loc[
        0,
        "false_negative_rate",
    ] == pytest.approx(
        0.25
    )

    assert result.loc[
        0,
        "predicted_positive_rate",
    ] == pytest.approx(
        0.40
    )

    assert result.loc[
        0,
        "brier_score",
    ] == pytest.approx(
        0.0943
    )

    assert result.loc[
        0,
        "log_loss",
    ] == pytest.approx(
        0.3255
    )

    assert result.loc[
        0,
        "calibration_error",
    ] == pytest.approx(
        0.015
    )

    assert result.loc[
        0,
        "true_positive",
    ] == 3

    assert result.loc[
        0,
        "false_positive",
    ] == 1

    assert result.loc[
        0,
        "true_negative",
    ] == 5

    assert result.loc[
        0,
        "false_negative",
    ] == 1

    assert result.loc[
        0,
        "classification_threshold",
    ] == pytest.approx(
        0.50
    )


def test_calculate_performance_metrics_preserves_integer_counts(
    performance_data: pd.DataFrame,
) -> None:
    """Record and confusion-matrix counts should remain integers."""

    result = calculate_performance_metrics(
        performance_data,
        threshold=0.50,
    )

    integer_columns = [
        "record_count",
        "default_count",
        "non_default_count",
        "true_positive",
        "false_positive",
        "true_negative",
        "false_negative",
    ]

    for column in integer_columns:
        assert pd.api.types.is_integer_dtype(
            result[
                column
            ]
        )