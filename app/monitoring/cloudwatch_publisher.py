"""Publish calculated model monitoring metrics to AWS CloudWatch."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import boto3
import pandas as pd

from app.config import (
    AWS_REGION,
    MODEL_VERSION,
)


DEFAULT_NAMESPACE = "CreditRiskModelMonitoring"

METRIC_COLUMN_MAPPING = {
    "auc": "AUC",
    "ks": "KS",
    "gini": "Gini",
    "bad_rate": "BadRate",
    "average_predicted_pd": "AveragePredictedPD",
    "accuracy": "Accuracy",
    "balanced_accuracy": "BalancedAccuracy",
    "precision": "Precision",
    "recall": "Recall",
    "specificity": "Specificity",
    "f1": "F1",
    "brier_score": "BrierScore",
    "log_loss": "LogLoss",
    "record_count": "RecordCount",
    "default_count": "DefaultCount",
}


def validate_performance_metrics(
    performance_metrics: pd.DataFrame,
) -> None:
    """Validate a one-row performance metric DataFrame."""

    if not isinstance(
        performance_metrics,
        pd.DataFrame,
    ):
        raise TypeError(
            "performance_metrics must be a pandas DataFrame."
        )

    if performance_metrics.empty:
        raise ValueError(
            "performance_metrics cannot be empty."
        )

    if len(
        performance_metrics
    ) != 1:
        raise ValueError(
            "performance_metrics must contain exactly one row."
        )


def validate_calibration_metric(
    metric_name: str,
    value: float,
) -> None:
    """Validate a calibration metric before publication."""

    if not isinstance(
        value,
        (
            int,
            float,
        ),
    ):
        raise TypeError(
            f"{metric_name} must be numeric."
        )

    if value < 0:
        raise ValueError(
            f"{metric_name} must be non-negative."
        )


def build_dimensions(
    model_version: str = MODEL_VERSION,
    environment: str = "development",
    additional_dimensions: Mapping[str, str] | None = None,
) -> list[dict[str, str]]:
    """Build CloudWatch metric dimensions."""

    if not model_version.strip():
        raise ValueError(
            "model_version cannot be empty."
        )

    if not environment.strip():
        raise ValueError(
            "environment cannot be empty."
        )

    dimensions = [
        {
            "Name": "ModelVersion",
            "Value": model_version,
        },
        {
            "Name": "Environment",
            "Value": environment,
        },
    ]

    if additional_dimensions:
        for name, value in additional_dimensions.items():
            if not str(
                name
            ).strip():
                raise ValueError(
                    "Dimension names cannot be empty."
                )

            if not str(
                value
            ).strip():
                raise ValueError(
                    "Dimension values cannot be empty."
                )

            dimensions.append(
                {
                    "Name": str(
                        name
                    ),
                    "Value": str(
                        value
                    ),
                }
            )

    return dimensions


def build_cloudwatch_metric_data(
    performance_metrics: pd.DataFrame,
    expected_calibration_error: float,
    maximum_calibration_error: float,
    model_version: str = MODEL_VERSION,
    environment: str = "development",
    additional_dimensions: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Build CloudWatch PutMetricData payload entries."""

    validate_performance_metrics(
        performance_metrics
    )

    validate_calibration_metric(
        "expected_calibration_error",
        expected_calibration_error,
    )

    validate_calibration_metric(
        "maximum_calibration_error",
        maximum_calibration_error,
    )

    metric_row = performance_metrics.iloc[
        0
    ]

    dimensions = build_dimensions(
        model_version=model_version,
        environment=environment,
        additional_dimensions=additional_dimensions,
    )

    metric_data: list[dict[str, Any]] = []

    for column_name, metric_name in (
        METRIC_COLUMN_MAPPING.items()
    ):
        if column_name not in metric_row.index:
            continue

        value = metric_row[
            column_name
        ]

        if pd.isna(
            value
        ):
            continue

        unit = (
            "Count"
            if column_name
            in {
                "record_count",
                "default_count",
            }
            else "None"
        )

        metric_data.append(
            {
                "MetricName": metric_name,
                "Dimensions": dimensions,
                "Value": float(
                    value
                ),
                "Unit": unit,
            }
        )

    metric_data.extend(
        [
            {
                "MetricName": (
                    "ExpectedCalibrationError"
                ),
                "Dimensions": dimensions,
                "Value": float(
                    expected_calibration_error
                ),
                "Unit": "None",
            },
            {
                "MetricName": (
                    "MaximumCalibrationError"
                ),
                "Dimensions": dimensions,
                "Value": float(
                    maximum_calibration_error
                ),
                "Unit": "None",
            },
        ]
    )

    return metric_data


def chunk_metric_data(
    metric_data: Sequence[
        dict[str, Any]
    ],
    chunk_size: int = 20,
) -> list[list[dict[str, Any]]]:
    """Split metrics into CloudWatch-sized request batches."""

    if not isinstance(
        chunk_size,
        int,
    ):
        raise TypeError(
            "chunk_size must be an integer."
        )

    if chunk_size < 1:
        raise ValueError(
            "chunk_size must be at least 1."
        )

    return [
        list(
            metric_data[
                start:
                start + chunk_size
            ]
        )
        for start in range(
            0,
            len(
                metric_data
            ),
            chunk_size,
        )
    ]


def publish_monitoring_metrics(
    performance_metrics: pd.DataFrame,
    expected_calibration_error: float,
    maximum_calibration_error: float,
    namespace: str = DEFAULT_NAMESPACE,
    model_version: str = MODEL_VERSION,
    environment: str = "development",
    additional_dimensions: Mapping[str, str] | None = None,
    cloudwatch_client: Any | None = None,
) -> dict[str, int]:
    """Publish monitoring metrics to AWS CloudWatch."""

    if not namespace.strip():
        raise ValueError(
            "namespace cannot be empty."
        )

    metric_data = build_cloudwatch_metric_data(
        performance_metrics=performance_metrics,
        expected_calibration_error=(
            expected_calibration_error
        ),
        maximum_calibration_error=(
            maximum_calibration_error
        ),
        model_version=model_version,
        environment=environment,
        additional_dimensions=additional_dimensions,
    )

    client = (
        cloudwatch_client
        if cloudwatch_client is not None
        else boto3.client(
            "cloudwatch",
            region_name=AWS_REGION,
        )
    )

    batches = chunk_metric_data(
        metric_data
    )

    for batch in batches:
        client.put_metric_data(
            Namespace=namespace,
            MetricData=batch,
        )

    return {
        "metric_count": len(
            metric_data
        ),
        "request_count": len(
            batches
        ),
    }