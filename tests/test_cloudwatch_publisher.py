"""Tests for CloudWatch monitoring metric publication."""

from unittest.mock import Mock

import pandas as pd
import pytest

from app.monitoring.cloudwatch_publisher import (
    build_cloudwatch_metric_data,
    build_dimensions,
    chunk_metric_data,
    publish_monitoring_metrics,
    validate_calibration_metric,
    validate_performance_metrics,
)


@pytest.fixture
def performance_metrics() -> pd.DataFrame:
    """Return one row of calculated performance metrics."""

    return pd.DataFrame(
        [
            {
                "record_count": 100,
                "default_count": 12,
                "auc": 0.84,
                "ks": 0.51,
                "gini": 0.68,
                "bad_rate": 0.12,
                "average_predicted_pd": 0.11,
                "accuracy": 0.82,
                "balanced_accuracy": 0.78,
                "precision": 0.65,
                "recall": 0.71,
                "specificity": 0.85,
                "f1": 0.68,
                "brier_score": 0.09,
                "log_loss": 0.31,
            }
        ]
    )


def test_validate_performance_metrics_accepts_valid_data(
    performance_metrics: pd.DataFrame,
) -> None:
    validate_performance_metrics(
        performance_metrics
    )


def test_validate_performance_metrics_rejects_non_dataframe() -> None:
    with pytest.raises(
        TypeError,
        match="must be a pandas DataFrame",
    ):
        validate_performance_metrics(
            []
        )


def test_validate_performance_metrics_rejects_empty_data() -> None:
    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        validate_performance_metrics(
            pd.DataFrame()
        )


def test_validate_performance_metrics_rejects_multiple_rows() -> None:
    with pytest.raises(
        ValueError,
        match="exactly one row",
    ):
        validate_performance_metrics(
            pd.DataFrame(
                [
                    {
                        "auc": 0.80,
                    },
                    {
                        "auc": 0.82,
                    },
                ]
            )
        )


def test_validate_calibration_metric_accepts_valid_value() -> None:
    validate_calibration_metric(
        "ece",
        0.02,
    )


def test_validate_calibration_metric_rejects_non_numeric() -> None:
    with pytest.raises(
        TypeError,
        match="must be numeric",
    ):
        validate_calibration_metric(
            "ece",
            "0.02",
        )


def test_validate_calibration_metric_rejects_negative_value() -> None:
    with pytest.raises(
        ValueError,
        match="must be non-negative",
    ):
        validate_calibration_metric(
            "ece",
            -0.01,
        )


def test_build_dimensions() -> None:
    result = build_dimensions(
        model_version="model-v2",
        environment="production",
    )

    assert result == [
        {
            "Name": "ModelVersion",
            "Value": "model-v2",
        },
        {
            "Name": "Environment",
            "Value": "production",
        },
    ]


def test_build_dimensions_adds_custom_dimensions() -> None:
    result = build_dimensions(
        model_version="model-v2",
        environment="production",
        additional_dimensions={
            "Portfolio": "PersonalLoans",
        },
    )

    assert {
        "Name": "Portfolio",
        "Value": "PersonalLoans",
    } in result


def test_build_cloudwatch_metric_data(
    performance_metrics: pd.DataFrame,
) -> None:
    result = build_cloudwatch_metric_data(
        performance_metrics=performance_metrics,
        expected_calibration_error=0.02,
        maximum_calibration_error=0.05,
        model_version="model-v2",
        environment="production",
    )

    metric_names = {
        metric[
            "MetricName"
        ]
        for metric in result
    }

    assert "AUC" in metric_names
    assert "KS" in metric_names
    assert "Gini" in metric_names
    assert "BadRate" in metric_names
    assert "RecordCount" in metric_names
    assert "DefaultCount" in metric_names
    assert (
        "ExpectedCalibrationError"
        in metric_names
    )
    assert (
        "MaximumCalibrationError"
        in metric_names
    )


def test_count_metrics_use_count_unit(
    performance_metrics: pd.DataFrame,
) -> None:
    result = build_cloudwatch_metric_data(
        performance_metrics=performance_metrics,
        expected_calibration_error=0.02,
        maximum_calibration_error=0.05,
    )

    metric_by_name = {
        metric[
            "MetricName"
        ]: metric
        for metric in result
    }

    assert metric_by_name[
        "RecordCount"
    ][
        "Unit"
    ] == "Count"

    assert metric_by_name[
        "DefaultCount"
    ][
        "Unit"
    ] == "Count"

    assert metric_by_name[
        "AUC"
    ][
        "Unit"
    ] == "None"


def test_build_metric_data_skips_missing_columns() -> None:
    metrics = pd.DataFrame(
        [
            {
                "auc": 0.84,
            }
        ]
    )

    result = build_cloudwatch_metric_data(
        performance_metrics=metrics,
        expected_calibration_error=0.02,
        maximum_calibration_error=0.05,
    )

    metric_names = {
        metric[
            "MetricName"
        ]
        for metric in result
    }

    assert metric_names == {
        "AUC",
        "ExpectedCalibrationError",
        "MaximumCalibrationError",
    }


def test_build_metric_data_skips_null_values() -> None:
    metrics = pd.DataFrame(
        [
            {
                "auc": 0.84,
                "ks": None,
            }
        ]
    )

    result = build_cloudwatch_metric_data(
        performance_metrics=metrics,
        expected_calibration_error=0.02,
        maximum_calibration_error=0.05,
    )

    metric_names = {
        metric[
            "MetricName"
        ]
        for metric in result
    }

    assert "AUC" in metric_names
    assert "KS" not in metric_names


def test_chunk_metric_data() -> None:
    metric_data = [
        {
            "MetricName": f"Metric{index}",
            "Value": float(
                index
            ),
        }
        for index in range(
            45
        )
    ]

    result = chunk_metric_data(
        metric_data,
        chunk_size=20,
    )

    assert len(
        result
    ) == 3

    assert len(
        result[
            0
        ]
    ) == 20

    assert len(
        result[
            1
        ]
    ) == 20

    assert len(
        result[
            2
        ]
    ) == 5


def test_chunk_metric_data_rejects_invalid_size() -> None:
    with pytest.raises(
        ValueError,
        match="at least 1",
    ):
        chunk_metric_data(
            [],
            chunk_size=0,
        )


def test_publish_monitoring_metrics(
    performance_metrics: pd.DataFrame,
) -> None:
    client = Mock()

    result = publish_monitoring_metrics(
        performance_metrics=performance_metrics,
        expected_calibration_error=0.02,
        maximum_calibration_error=0.05,
        namespace="CreditRiskTest",
        model_version="model-v2",
        environment="test",
        cloudwatch_client=client,
    )

    assert result[
        "metric_count"
    ] == 17

    assert result[
        "request_count"
    ] == 1

    client.put_metric_data.assert_called_once()

    call_arguments = (
        client
        .put_metric_data
        .call_args
        .kwargs
    )

    assert call_arguments[
        "Namespace"
    ] == "CreditRiskTest"

    assert len(
        call_arguments[
            "MetricData"
        ]
    ) == 17


def test_publish_monitoring_metrics_rejects_empty_namespace(
    performance_metrics: pd.DataFrame,
) -> None:
    with pytest.raises(
        ValueError,
        match="namespace cannot be empty",
    ):
        publish_monitoring_metrics(
            performance_metrics=performance_metrics,
            expected_calibration_error=0.02,
            maximum_calibration_error=0.05,
            namespace="",
            cloudwatch_client=Mock(),
        )