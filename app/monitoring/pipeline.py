"""AWS-native credit risk model monitoring pipeline.

This pipeline:
1. Validates prediction and outcome data.
2. Calculates performance metrics.
3. Calculates calibration table, ECE, and MCE.
4. Calculates PSI against a reference dataset.
5. Calculates configured segment metrics.
6. Calculates adverse reason summaries when data is available.
7. Stamps all datasets with model_version, report_date, environment,
   and run_id.
8. Writes all analytical datasets to S3 as partitioned Parquet.
9. Publishes summary KPIs to CloudWatch.
10. Records pipeline success or failure.
11. Returns a structured result.

This pipeline must never:
- Generate HTML reports.
- Call generate_html_report().
- Retrain the model.
- Recalibrate probabilities.
- Update thresholds.
- Deploy a new model version.
- Change governance status.
"""

from __future__ import annotations

import uuid
from datetime import date
from typing import Any

import pandas as pd

from app.config import (
    CLOUDWATCH_MONITORING_NAMESPACE,
    ENVIRONMENT,
    MIN_PERFORMANCE_SAMPLES,
    MODEL_CLASSIFICATION_THRESHOLD,
    MODEL_VERSION,
    PSI_ALERT_THRESHOLD,
    PSI_WARNING_THRESHOLD,
    S3_ANALYTICS_PREFIX,
    S3_BUCKET,
)
from app.monitoring.adverse_reason_metrics import (
    calculate_adverse_reason_summary,
)
from app.monitoring.bi_dataset_writer import (
    write_monitoring_datasets,
)
from app.monitoring.calibration_metrics import (
    calculate_calibration_table,
    calculate_expected_calibration_error,
    calculate_maximum_calibration_error,
)
from app.monitoring.cloudwatch_publisher import (
    publish_monitoring_metrics,
)
from app.monitoring.performance_metrics import (
    calculate_performance_metrics,
)
from app.monitoring.psi import (
    calculate_feature_psi,
)
from app.monitoring.segment_metrics import (
    calculate_segment_metrics,
)


def validate_pipeline_data(
    predictions: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> None:
    """Validate prediction and outcome DataFrames before joining.

    Parameters
    ----------
    predictions:
        Must contain application_id and predicted_probability.
    outcomes:
        Must contain application_id and actual_default.

    Raises
    ------
    TypeError / ValueError
        On structural violations.
    """

    for name, df in (
        (
            "predictions",
            predictions,
        ),
        (
            "outcomes",
            outcomes,
        ),
    ):
        if not isinstance(
            df,
            pd.DataFrame,
        ):
            raise TypeError(
                f"{name} must be a pandas DataFrame."
            )

        if df.empty:
            raise ValueError(
                f"{name} DataFrame is empty."
            )

    pred_required = {
        "application_id",
        "predicted_probability",
    }

    pred_missing = sorted(
        pred_required.difference(
            predictions.columns
        )
    )

    if pred_missing:
        raise ValueError(
            f"Predictions missing columns: {pred_missing}"
        )

    out_required = {
        "application_id",
        "actual_default",
    }

    out_missing = sorted(
        out_required.difference(
            outcomes.columns
        )
    )

    if out_missing:
        raise ValueError(
            f"Outcomes missing columns: {out_missing}"
        )


def join_predictions_and_outcomes(
    predictions: pd.DataFrame,
    outcomes: pd.DataFrame,
    minimum_samples: int | None = None,
) -> pd.DataFrame:
    """Inner-join predictions to outcomes on application_id.

    Parameters
    ----------
    predictions:
        Prediction DataFrame.
    outcomes:
        Outcomes DataFrame.
    minimum_samples:
        Minimum matched rows required to proceed. Defaults to
        MIN_PERFORMANCE_SAMPLES from config when None.

    Returns
    -------
    pd.DataFrame
        Joined DataFrame with predicted_probability and actual_default.

    Raises
    ------
    ValueError
        If no matching records or too few records are found.
    """

    from app.config import (
        MIN_PERFORMANCE_SAMPLES as _DEFAULT_MIN,
    )

    resolved_minimum = (
        minimum_samples
        if minimum_samples is not None
        else _DEFAULT_MIN
    )

    joined = predictions.merge(
        outcomes[
            [
                "application_id",
                "actual_default",
            ]
        ],
        on="application_id",
        how="inner",
    )

    if joined.empty:
        raise ValueError(
            "No matching application_id values found between "
            "predictions and outcomes."
        )

    if len(
        joined
    ) < resolved_minimum:
        raise ValueError(
            f"Too few matched samples for monitoring: "
            f"found {len(joined)}, need at least {resolved_minimum}."
        )

    return joined


def _build_psi_dataframe(
    psi_raw: pd.DataFrame,
    model_version: str,
    report_date: str,
    environment: str,
) -> pd.DataFrame:
    """Stamp PSI results and normalise status labels."""

    df = psi_raw.copy()

    df[
        "status"
    ] = df[
        "psi"
    ].apply(
        lambda v: (
            "stable"
            if v < PSI_WARNING_THRESHOLD
            else (
                "warning"
                if v < PSI_ALERT_THRESHOLD
                else "alert"
            )
        )
    )

    df[
        "model_version"
    ] = model_version

    df[
        "report_date"
    ] = report_date

    df[
        "environment"
    ] = environment

    return df


def run_monitoring_pipeline(
    predictions: pd.DataFrame,
    outcomes: pd.DataFrame,
    reference_data: pd.DataFrame | None = None,
    model_version: str = MODEL_VERSION,
    report_date: date | None = None,
    environment: str = ENVIRONMENT,
    threshold: float = MODEL_CLASSIFICATION_THRESHOLD,
    number_of_bands: int = 10,
    minimum_samples: int | None = None,
    segment_columns: list[str] | None = None,
    adverse_reason_data: pd.DataFrame | None = None,
    bucket: str = S3_BUCKET,
    analytics_prefix: str = S3_ANALYTICS_PREFIX,
    cloudwatch_namespace: str = CLOUDWATCH_MONITORING_NAMESPACE,
    publish_to_cloudwatch: bool = True,
    write_to_s3: bool = True,
    run_id: str | None = None,
    s3_client: Any | None = None,
    cloudwatch_client: Any | None = None,
) -> dict[str, object]:
    """Run the full credit risk model monitoring pipeline.

    This pipeline reads prediction and outcome data, computes all
    monitoring metrics, writes Parquet datasets to S3, and publishes
    summary KPIs to CloudWatch.

    It does not generate HTML reports, retrain the model, recalibrate
    probabilities, change thresholds, or alter governance status.

    Parameters
    ----------
    predictions:
        DataFrame with application_id and predicted_probability.
    outcomes:
        DataFrame with application_id and actual_default.
    reference_data:
        Reference dataset used for PSI calculation. PSI is skipped
        when this is None.
    model_version:
        Model version tag.
    report_date:
        Report date; defaults to today.
    environment:
        Deployment environment.
    threshold:
        Binary classification threshold.
    number_of_bands:
        Number of calibration bands.
    minimum_samples:
        Minimum matched samples required. Defaults to config value.
    segment_columns:
        Column names in predictions to segment by.
    adverse_reason_data:
        Optional DataFrame with reason_code, adverse_reason, and
        contribution columns. Adverse reason summary is skipped when
        this is None.
    bucket:
        Target S3 bucket.
    analytics_prefix:
        S3 prefix for analytical datasets.
    cloudwatch_namespace:
        CloudWatch metric namespace.
    publish_to_cloudwatch:
        Whether to publish metrics to CloudWatch.
    write_to_s3:
        Whether to write datasets to S3.
    run_id:
        Pipeline run identifier; generated if None.
    s3_client:
        Injected S3 client for testing.
    cloudwatch_client:
        Injected CloudWatch client for testing.

    Returns
    -------
    dict[str, object]
        Structured result with run_id, model_version, report_date,
        environment, record_count, cloudwatch_publish_result,
        s3_dataset_results, performance_metrics,
        expected_calibration_error, maximum_calibration_error,
        maximum_psi, warning_feature_count, alert_feature_count,
        and status.
    """

    resolved_run_id = (
        run_id
        if run_id is not None
        else str(
            uuid.uuid4()
        )
    )

    resolved_date = (
        report_date
        if report_date is not None
        else date.today()
    )

    report_date_str = resolved_date.isoformat()

    result: dict[str, object] = {
        "run_id": resolved_run_id,
        "model_version": model_version,
        "report_date": report_date_str,
        "environment": environment,
        "record_count": 0,
        "cloudwatch_publish_result": None,
        "s3_dataset_results": None,
        "performance_metrics": None,
        "expected_calibration_error": None,
        "maximum_calibration_error": None,
        "maximum_psi": None,
        "warning_feature_count": 0,
        "alert_feature_count": 0,
        "status": "failed",
    }

    try:
        # ----------------------------------------------------------
        # 1. Validate and join
        # ----------------------------------------------------------

        validate_pipeline_data(
            predictions=predictions,
            outcomes=outcomes,
        )

        joined = join_predictions_and_outcomes(
            predictions=predictions,
            outcomes=outcomes,
            minimum_samples=minimum_samples,
        )

        result[
            "record_count"
        ] = len(
            joined
        )

        # ----------------------------------------------------------
        # 2. Performance metrics
        # ----------------------------------------------------------

        performance_metrics = calculate_performance_metrics(
            data=joined,
            threshold=threshold,
        )

        performance_metrics[
            "model_version"
        ] = model_version

        performance_metrics[
            "report_date"
        ] = report_date_str

        performance_metrics[
            "environment"
        ] = environment

        performance_metrics[
            "run_id"
        ] = resolved_run_id

        result[
            "performance_metrics"
        ] = performance_metrics

        # ----------------------------------------------------------
        # 3. Calibration metrics
        # ----------------------------------------------------------

        calibration_table = calculate_calibration_table(
            data=joined,
            number_of_bands=number_of_bands,
        )

        ece = calculate_expected_calibration_error(
            calibration_table
        )

        mce = calculate_maximum_calibration_error(
            calibration_table
        )

        calibration_table[
            "model_version"
        ] = model_version

        calibration_table[
            "report_date"
        ] = report_date_str

        calibration_table[
            "environment"
        ] = environment

        result[
            "expected_calibration_error"
        ] = ece

        result[
            "maximum_calibration_error"
        ] = mce

        # ----------------------------------------------------------
        # 4. PSI
        # ----------------------------------------------------------

        psi_df: pd.DataFrame | None = None
        maximum_psi = 0.0
        warning_feature_count = 0
        alert_feature_count = 0

        if reference_data is not None and not reference_data.empty:
            feature_columns = [
                col
                for col in reference_data.columns
                if col in joined.columns
                and col not in (
                    "application_id",
                    "actual_default",
                    "predicted_probability",
                )
            ]

            if feature_columns:
                psi_raw = calculate_feature_psi(
                    reference_data[
                        feature_columns
                    ],
                    joined[
                        feature_columns
                    ],
                )

                psi_df = _build_psi_dataframe(
                    psi_raw=psi_raw,
                    model_version=model_version,
                    report_date=report_date_str,
                    environment=environment,
                )

                maximum_psi = float(
                    psi_df[
                        "psi"
                    ].max()
                )

                warning_feature_count = int(
                    (
                        psi_df[
                            "status"
                        ] == "warning"
                    ).sum()
                )

                alert_feature_count = int(
                    (
                        psi_df[
                            "status"
                        ] == "alert"
                    ).sum()
                )

        result[
            "maximum_psi"
        ] = maximum_psi

        result[
            "warning_feature_count"
        ] = warning_feature_count

        result[
            "alert_feature_count"
        ] = alert_feature_count

        # ----------------------------------------------------------
        # 5. Segment metrics
        # ----------------------------------------------------------

        segment_df: pd.DataFrame | None = None

        if segment_columns:
            # Merge segment columns from predictions into joined df
            segment_source = predictions[
                [
                    col
                    for col in (
                        ["application_id"]
                        + segment_columns
                    )
                    if col in predictions.columns
                ]
            ]

            joined_with_segments = joined.merge(
                segment_source,
                on="application_id",
                how="left",
                suffixes=(
                    "",
                    "_seg",
                ),
            )

            available_segments = [
                col
                for col in segment_columns
                if col in joined_with_segments.columns
            ]

            if available_segments:
                segment_df = calculate_segment_metrics(
                    data=joined_with_segments,
                    segment_columns=available_segments,
                    model_version=model_version,
                    report_date=report_date_str,
                    environment=environment,
                )

        # ----------------------------------------------------------
        # 6. Adverse reason summary
        # ----------------------------------------------------------

        adverse_df: pd.DataFrame | None = None

        if (
            adverse_reason_data is not None
            and not adverse_reason_data.empty
        ):
            adverse_df = calculate_adverse_reason_summary(
                data=adverse_reason_data,
                model_version=model_version,
                report_date=report_date_str,
                environment=environment,
                total_predictions=len(
                    joined
                ),
            )

        # ----------------------------------------------------------
        # 7. Build pipeline_runs record
        # ----------------------------------------------------------

        pipeline_runs_df = pd.DataFrame(
            [
                {
                    "run_id": resolved_run_id,
                    "model_version": model_version,
                    "report_date": report_date_str,
                    "environment": environment,
                    "record_count": len(
                        joined
                    ),
                    "expected_calibration_error": ece,
                    "maximum_calibration_error": mce,
                    "maximum_psi": maximum_psi,
                    "warning_feature_count": warning_feature_count,
                    "alert_feature_count": alert_feature_count,
                    "status": "success",
                }
            ]
        )

        # ----------------------------------------------------------
        # 8. Write S3 datasets
        # ----------------------------------------------------------

        s3_datasets: dict[str, pd.DataFrame] = {
            "performance_metrics": performance_metrics,
            "calibration_metrics": calibration_table,
            "pipeline_runs": pipeline_runs_df,
        }

        if psi_df is not None:
            s3_datasets[
                "psi_metrics"
            ] = psi_df

        if segment_df is not None and not segment_df.empty:
            s3_datasets[
                "segment_metrics"
            ] = segment_df

        if adverse_df is not None and not adverse_df.empty:
            s3_datasets[
                "adverse_reason_summary"
            ] = adverse_df

        s3_result: dict[str, object] | None = None

        if write_to_s3:
            s3_result = write_monitoring_datasets(
                datasets=s3_datasets,
                bucket=bucket,
                model_version=model_version,
                report_date=resolved_date,
                analytics_prefix=analytics_prefix,
                environment=environment,
                run_id=resolved_run_id,
                s3_client=s3_client,
            )

        result[
            "s3_dataset_results"
        ] = s3_result

        # ----------------------------------------------------------
        # 9. Publish CloudWatch metrics
        # ----------------------------------------------------------

        cw_result: dict[str, int] | None = None

        if publish_to_cloudwatch:
            psi_dimensions = {
                "MaximumPSI": maximum_psi,
                "WarningFeatureCount": float(
                    warning_feature_count
                ),
                "AlertFeatureCount": float(
                    alert_feature_count
                ),
            }

            try:
                cw_result = publish_monitoring_metrics(
                    performance_metrics=performance_metrics[
                        [
                            col
                            for col in performance_metrics.columns
                            if col not in (
                                "model_version",
                                "report_date",
                                "environment",
                                "run_id",
                            )
                        ]
                    ],
                    expected_calibration_error=ece,
                    maximum_calibration_error=mce,
                    namespace=cloudwatch_namespace,
                    model_version=model_version,
                    environment=environment,
                    cloudwatch_client=cloudwatch_client,
                )

                # Publish PSI summary separately
                if cloudwatch_client is not None:
                    _publish_psi_summary(
                        cloudwatch_client=cloudwatch_client,
                        namespace=cloudwatch_namespace,
                        model_version=model_version,
                        environment=environment,
                        maximum_psi=maximum_psi,
                        warning_feature_count=warning_feature_count,
                        alert_feature_count=alert_feature_count,
                    )

            except Exception as cw_exc:
                cw_result = {
                    "error": str(
                        cw_exc
                    )
                }

        result[
            "cloudwatch_publish_result"
        ] = cw_result

        # ----------------------------------------------------------
        # 10. Record success
        # ----------------------------------------------------------

        result[
            "status"
        ] = "success"

    except Exception as exc:
        result[
            "status"
        ] = "failed"

        result[
            "error"
        ] = str(
            exc
        )

        # Write failure record to pipeline_runs if S3 is enabled
        if write_to_s3:
            try:
                failure_df = pd.DataFrame(
                    [
                        {
                            "run_id": resolved_run_id,
                            "model_version": model_version,
                            "report_date": report_date_str,
                            "environment": environment,
                            "record_count": result.get(
                                "record_count",
                                0,
                            ),
                            "status": "failed",
                            "error": str(
                                exc
                            ),
                        }
                    ]
                )

                write_monitoring_datasets(
                    datasets={
                        "pipeline_runs": failure_df,
                    },
                    bucket=bucket,
                    model_version=model_version,
                    report_date=resolved_date,
                    analytics_prefix=analytics_prefix,
                    environment=environment,
                    run_id=resolved_run_id,
                    s3_client=s3_client,
                )

            except Exception:
                pass

        raise

    return result


def _publish_psi_summary(
    cloudwatch_client: Any,
    namespace: str,
    model_version: str,
    environment: str,
    maximum_psi: float,
    warning_feature_count: int,
    alert_feature_count: int,
) -> None:
    """Publish PSI summary metrics to CloudWatch."""

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

    cloudwatch_client.put_metric_data(
        Namespace=namespace,
        MetricData=[
            {
                "MetricName": "MaximumPSI",
                "Dimensions": dimensions,
                "Value": float(
                    maximum_psi
                ),
                "Unit": "None",
            },
            {
                "MetricName": "WarningFeatureCount",
                "Dimensions": dimensions,
                "Value": float(
                    warning_feature_count
                ),
                "Unit": "Count",
            },
            {
                "MetricName": "AlertFeatureCount",
                "Dimensions": dimensions,
                "Value": float(
                    alert_feature_count
                ),
                "Unit": "Count",
            },
        ],
    )
