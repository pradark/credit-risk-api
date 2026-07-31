"""Model performance monitoring workflow."""

from __future__ import annotations

import datetime
from typing import Any

import boto3
import pandas as pd

from app.config import (
    AWS_REGION,
    MIN_PERFORMANCE_SAMPLES,
    MODEL_CLASSIFICATION_THRESHOLD,
    MODEL_VERSION,
    S3_BUCKET,
    S3_MONITORING_PREFIX,
    S3_OUTCOME_PREFIX,
    S3_PREDICTION_PREFIX,
)
from app.monitoring.performance_metrics import (
    calculate_performance_metrics,
)
from app.monitoring.s3_utils import (
    read_parquet_from_s3,
    write_parquet_to_s3,
)


cloudwatch = boto3.client(
    "cloudwatch",
    region_name=AWS_REGION,
)


def build_performance_s3_keys(
    run_date: str,
) -> dict[str, str]:
    """
    Build S3 keys used by the model performance monitoring workflow.

    Parameters
    ----------
    run_date:
        Monitoring date in YYYY-MM-DD format.

    Returns
    -------
    dict[str, str]
        Prediction, outcome, and output S3 keys.
    """
    return {
        "prediction_key": (
            f"{S3_PREDICTION_PREFIX}/"
            f"dt={run_date}/"
            "predictions.parquet"
        ),
        "outcome_key": (
            f"{S3_OUTCOME_PREFIX}/"
            f"dt={run_date}/"
            "outcomes.parquet"
        ),
        "output_key": (
            f"{S3_MONITORING_PREFIX}/"
            "performance/"
            f"model_performance_{run_date}.parquet"
        ),
    }


def validate_prediction_data(
    predictions: pd.DataFrame,
) -> None:
    """
    Validate the prediction dataset before joining outcomes.
    """
    required_columns = {
        "application_id",
        "predicted_probability",
    }

    missing_columns = sorted(
        required_columns.difference(predictions.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Prediction data is missing required columns: {missing_columns}"
        )

    if predictions.empty:
        raise ValueError("Prediction data is empty.")

    if predictions["application_id"].isna().any():
        raise ValueError(
            "Prediction data contains missing application_id values."
        )

    if predictions["application_id"].duplicated().any():
        raise ValueError(
            "Prediction data contains duplicate application_id values."
        )

    if predictions["predicted_probability"].isna().any():
        raise ValueError(
            "Prediction data contains missing predicted_probability values."
        )

    if not predictions["predicted_probability"].between(
        0.0,
        1.0,
    ).all():
        raise ValueError(
            "predicted_probability must contain values between 0 and 1."
        )


def validate_outcome_data(
    outcomes: pd.DataFrame,
) -> None:
    """
    Validate the observed outcome dataset before joining predictions.
    """
    required_columns = {
        "application_id",
        "actual_default",
    }

    missing_columns = sorted(
        required_columns.difference(outcomes.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Outcome data is missing required columns: {missing_columns}"
        )

    if outcomes.empty:
        raise ValueError("Outcome data is empty.")

    if outcomes["application_id"].isna().any():
        raise ValueError(
            "Outcome data contains missing application_id values."
        )

    if outcomes["application_id"].duplicated().any():
        raise ValueError(
            "Outcome data contains duplicate application_id values."
        )

    if outcomes["actual_default"].isna().any():
        raise ValueError(
            "Outcome data contains missing actual_default values."
        )

    actual_values = set(
        outcomes["actual_default"].unique()
    )

    if not actual_values.issubset({0, 1}):
        raise ValueError(
            "actual_default must contain only 0 and 1."
        )


def join_predictions_and_outcomes(
    predictions: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join predictions and outcomes using application_id.
    """
    validate_prediction_data(predictions)
    validate_outcome_data(outcomes)

    performance_data = predictions.merge(
        outcomes[
            [
                "application_id",
                "actual_default",
            ]
        ],
        on="application_id",
        how="inner",
        validate="one_to_one",
    )

    if performance_data.empty:
        raise ValueError(
            "No matching application_id values were found "
            "between predictions and outcomes."
        )

    return performance_data


def validate_minimum_sample_size(
    performance_data: pd.DataFrame,
    minimum_samples: int = MIN_PERFORMANCE_SAMPLES,
) -> None:
    """
    Validate that enough matured records exist for monitoring.
    """
    if minimum_samples < 1:
        raise ValueError(
            "Minimum performance sample size must be at least 1."
        )

    if len(performance_data) < minimum_samples:
        raise ValueError(
            "Not enough matched samples for model performance monitoring. "
            f"Found {len(performance_data)}, "
            f"need at least {minimum_samples}."
        )


def add_monitoring_metadata(
    metrics: pd.DataFrame,
    run_date: str,
    prediction_key: str,
    outcome_key: str,
    matched_sample_size: int,
) -> pd.DataFrame:
    """
    Add operational metadata to the performance report.
    """
    report = metrics.copy()

    report["run_date"] = run_date
    report["prediction_dataset"] = prediction_key
    report["outcome_dataset"] = outcome_key
    report["matched_sample_size"] = matched_sample_size
    report["model_version"] = MODEL_VERSION
    report["monitoring_type"] = "model_performance"

    return report


def publish_performance_metrics(
    report: pd.DataFrame,
) -> None:
    """
    Publish model performance metrics to CloudWatch.
    """
    if report.empty:
        raise ValueError(
            "Cannot publish an empty model performance report."
        )

    row = report.iloc[0]

    metric_names = {
        "auc": "ModelAUC",
        "ks": "ModelKS",
        "gini": "ModelGini",
        "bad_rate": "ObservedBadRate",
        "average_predicted_pd": "AveragePredictedPD",
        "precision": "ModelPrecision",
        "recall": "ModelRecall",
        "calibration_error": "CalibrationError",
        "record_count": "PerformanceRecordCount",
        "default_count": "PerformanceDefaultCount",
        "non_default_count": "PerformanceNonDefaultCount",
    }

    metric_data: list[dict[str, Any]] = []

    for column_name, metric_name in metric_names.items():
        metric_data.append(
            {
                "MetricName": metric_name,
                "Value": float(row[column_name]),
                "Unit": "Count"
                if column_name.endswith("_count")
                else "None",
                "Dimensions": [
                    {
                        "Name": "ModelVersion",
                        "Value": str(row["model_version"]),
                    }
                ],
            }
        )

    cloudwatch.put_metric_data(
        Namespace="CreditRiskAPI",
        MetricData=metric_data,
    )


def run_performance_monitoring(
    run_date: str | None = None,
) -> pd.DataFrame:
    """
    Run the complete model performance monitoring workflow.

    Parameters
    ----------
    run_date:
        Monitoring date in YYYY-MM-DD format. Defaults to today's date.

    Returns
    -------
    pandas.DataFrame
        One-row model performance report.
    """
    monitoring_date = (
        run_date
        if run_date is not None
        else datetime.date.today().isoformat()
    )

    s3_keys = build_performance_s3_keys(
        monitoring_date
    )

    prediction_key = s3_keys["prediction_key"]
    outcome_key = s3_keys["outcome_key"]
    output_key = s3_keys["output_key"]

    print("Loading prediction data...")

    predictions = read_parquet_from_s3(
        S3_BUCKET,
        prediction_key,
    )

    print("Loading outcome data...")

    outcomes = read_parquet_from_s3(
        S3_BUCKET,
        outcome_key,
    )

    print("Joining predictions and outcomes...")

    performance_data = join_predictions_and_outcomes(
        predictions=predictions,
        outcomes=outcomes,
    )

    validate_minimum_sample_size(
        performance_data=performance_data,
    )

    print("Calculating model performance metrics...")

    metrics = calculate_performance_metrics(
        data=performance_data,
        threshold=MODEL_CLASSIFICATION_THRESHOLD,
    )

    report = add_monitoring_metadata(
        metrics=metrics,
        run_date=monitoring_date,
        prediction_key=prediction_key,
        outcome_key=outcome_key,
        matched_sample_size=len(performance_data),
    )

    print()
    print("Model Performance Results")
    print("-" * 80)
    print(
        report.to_string(
            index=False
        )
    )

    write_parquet_to_s3(
        report,
        S3_BUCKET,
        output_key,
    )

    print(
        f"Successfully wrote model performance report to S3: {output_key}"
    )

    publish_performance_metrics(
        report
    )

    print(
        "Successfully published model performance metrics to CloudWatch"
    )

    print(
        "Model performance monitoring complete"
    )

    return report


if __name__ == "__main__":
    run_performance_monitoring()
