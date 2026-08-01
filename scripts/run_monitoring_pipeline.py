"""AWS-native credit risk model monitoring pipeline CLI.

Runs the full monitoring pipeline:
  - Loads prediction and outcome data (CSV or Parquet).
  - Joins on application_id.
  - Calculates performance, calibration, PSI, segment, and adverse
    reason metrics.
  - Writes partitioned Parquet datasets to S3.
  - Publishes summary KPIs to CloudWatch.

This script never generates HTML reports and never calls
generate_html_report().

Example
-------
uv run python scripts/run_monitoring_pipeline.py \\
    --input monitoring/performance_joined.parquet \\
    --reference-input monitoring/reference/reference_data.parquet \\
    --output-bucket credit-risk-monitoring-pradark \\
    --analytics-prefix analytics \\
    --model-version credit-risk-model-v1 \\
    --environment development \\
    --publish-to-cloudwatch \\
    --write-to-s3
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path


PROJECT_ROOT = Path(
    __file__
).resolve().parent.parent

if str(
    PROJECT_ROOT
) not in sys.path:
    sys.path.insert(
        0,
        str(
            PROJECT_ROOT
        ),
    )


import pandas as pd  # noqa: E402

from app.config import (  # noqa: E402
    CLOUDWATCH_MONITORING_NAMESPACE,
    ENVIRONMENT,
    MIN_PERFORMANCE_SAMPLES,
    MODEL_CLASSIFICATION_THRESHOLD,
    MODEL_VERSION,
    S3_ANALYTICS_PREFIX,
    S3_BUCKET,
)
from app.monitoring.pipeline import (  # noqa: E402
    run_monitoring_pipeline,
)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "AWS-native credit risk model monitoring pipeline. "
            "Writes Parquet datasets to S3 and publishes metrics "
            "to CloudWatch. Does not generate HTML reports."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help=(
            "Path to the joined predictions and outcomes file "
            "(CSV or Parquet). Must contain application_id, "
            "predicted_probability, and actual_default."
        ),
    )

    parser.add_argument(
        "--reference-input",
        type=Path,
        default=None,
        help=(
            "Path to the reference dataset for PSI calculation "
            "(CSV or Parquet). PSI is skipped when not provided."
        ),
    )

    parser.add_argument(
        "--output-bucket",
        type=str,
        default=S3_BUCKET,
        help="Target S3 bucket for analytical datasets.",
    )

    parser.add_argument(
        "--analytics-prefix",
        type=str,
        default=S3_ANALYTICS_PREFIX,
        help="S3 key prefix for analytical datasets.",
    )

    parser.add_argument(
        "--model-version",
        type=str,
        default=MODEL_VERSION,
        help="Model version tag used in S3 partitions and metrics.",
    )

    parser.add_argument(
        "--environment",
        type=str,
        default=ENVIRONMENT,
        help="Deployment environment (development, staging, production).",
    )

    parser.add_argument(
        "--report-date",
        type=str,
        default=None,
        help=(
            "Report date in YYYY-MM-DD format. "
            "Defaults to today."
        ),
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=MODEL_CLASSIFICATION_THRESHOLD,
        help="Binary classification probability threshold.",
    )

    parser.add_argument(
        "--number-of-bands",
        type=int,
        default=10,
        help="Number of quantile-based calibration bands.",
    )

    parser.add_argument(
        "--publish-to-cloudwatch",
        action="store_true",
        default=False,
        help="Publish summary KPIs to CloudWatch.",
    )

    parser.add_argument(
        "--write-to-s3",
        action="store_true",
        default=False,
        help="Write Parquet datasets to S3.",
    )

    parser.add_argument(
        "--segment-columns",
        type=str,
        nargs="*",
        default=None,
        help="Column names to segment metrics by.",
    )

    parser.add_argument(
        "--cloudwatch-namespace",
        type=str,
        default=CLOUDWATCH_MONITORING_NAMESPACE,
        help="CloudWatch metric namespace.",
    )

    return parser.parse_args()


def load_data(
    path: Path,
) -> pd.DataFrame:
    """Load a CSV or Parquet file into a DataFrame.

    Parameters
    ----------
    path:
        File path to load.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file extension is not supported.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Input file does not exist: {path}"
        )

    suffix = path.suffix.lower()

    if suffix == ".parquet":
        return pd.read_parquet(
            path
        )

    if suffix == ".csv":
        return pd.read_csv(
            path
        )

    raise ValueError(
        f"Unsupported file extension '{suffix}'. "
        "Only .parquet and .csv files are accepted."
    )


def parse_report_date(
    date_string: str | None,
) -> date | None:
    """Parse an optional YYYY-MM-DD date string.

    Returns None if date_string is None, which causes the pipeline to
    use today's date.
    """

    if date_string is None:
        return None

    try:
        return date.fromisoformat(
            date_string
        )

    except ValueError as exc:
        raise ValueError(
            f"Invalid report date '{date_string}'. "
            "Use YYYY-MM-DD format."
        ) from exc


def print_summary(
    result: dict[str, object],
) -> None:
    """Print a structured pipeline summary to stdout."""

    print(
        "\n"
        + "=" * 70
    )

    print(
        "MONITORING PIPELINE SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        f"Status         : {result.get('status', 'unknown')}"
    )

    print(
        f"Run ID         : {result.get('run_id', 'N/A')}"
    )

    print(
        f"Model version  : {result.get('model_version', 'N/A')}"
    )

    print(
        f"Report date    : {result.get('report_date', 'N/A')}"
    )

    print(
        f"Environment    : {result.get('environment', 'N/A')}"
    )

    print(
        f"Record count   : {result.get('record_count', 0):,}"
    )

    perf = result.get(
        "performance_metrics"
    )

    if perf is not None and hasattr(
        perf,
        "iloc",
    ):
        row = perf.iloc[
            0
        ]

        print(
            f"\nPerformance metrics:"
        )

        print(
            f"  AUC          : {row.get('auc', 'N/A')}"
        )

        print(
            f"  KS           : {row.get('ks', 'N/A')}"
        )

        print(
            f"  Gini         : {row.get('gini', 'N/A')}"
        )

        print(
            f"  Bad rate     : {row.get('bad_rate', 'N/A')}"
        )

        print(
            f"  Avg PD       : {row.get('average_predicted_pd', 'N/A')}"
        )

    ece = result.get(
        "expected_calibration_error"
    )

    mce = result.get(
        "maximum_calibration_error"
    )

    if ece is not None:
        print(
            f"\nCalibration:"
        )

        print(
            f"  ECE          : {ece:.4f}"
        )

        if mce is not None:
            print(
                f"  MCE          : {mce:.4f}"
            )

    max_psi = result.get(
        "maximum_psi"
    )

    if max_psi is not None:
        print(
            f"\nPSI:"
        )

        print(
            f"  Maximum PSI        : {max_psi:.4f}"
        )

        print(
            f"  Warning features   : {result.get('warning_feature_count', 0)}"
        )

        print(
            f"  Alert features     : {result.get('alert_feature_count', 0)}"
        )

    s3 = result.get(
        "s3_dataset_results"
    )

    if s3 is not None:
        print(
            f"\nS3 datasets written: {s3.get('dataset_count', 0)}"
        )

        if s3.get(
            "errors"
        ):
            print(
                f"  S3 errors: {s3['errors']}"
            )

    cw = result.get(
        "cloudwatch_publish_result"
    )

    if cw is not None:
        if isinstance(
            cw,
            dict,
        ) and "metric_count" in cw:
            print(
                f"\nCloudWatch: {cw['metric_count']} metrics in "
                f"{cw['request_count']} requests"
            )

        elif isinstance(
            cw,
            dict,
        ) and "error" in cw:
            print(
                f"\nCloudWatch error: {cw['error']}"
            )

    error = result.get(
        "error"
    )

    if error:
        print(
            f"\nError: {error}"
        )

    print(
        "=" * 70
        + "\n"
    )


def main() -> None:
    """Entry point for the monitoring pipeline CLI."""

    args = parse_arguments()

    # Load input data
    print(
        f"Loading input data from: {args.input}"
    )

    combined_data = load_data(
        args.input
    )

    # Split into predictions and outcomes if both columns exist
    if (
        "predicted_probability" in combined_data.columns
        and "actual_default" in combined_data.columns
    ):
        # Data is pre-joined
        predictions = combined_data

        outcomes = combined_data[
            [
                "application_id",
                "actual_default",
            ]
        ].copy()

    else:
        predictions = combined_data

        outcomes = combined_data[
            [
                "application_id",
                "actual_default",
            ]
        ].copy()

    # Load reference data for PSI
    reference_data = None

    if args.reference_input is not None:
        print(
            f"Loading reference data from: {args.reference_input}"
        )

        reference_data = load_data(
            args.reference_input
        )

    # Parse report date
    report_date = parse_report_date(
        args.report_date
    )

    print(
        f"\nStarting monitoring pipeline:"
        f"\n  Model version  : {args.model_version}"
        f"\n  Environment    : {args.environment}"
        f"\n  Report date    : {report_date or 'today'}"
        f"\n  Bucket         : {args.output_bucket}"
        f"\n  CloudWatch     : {args.publish_to_cloudwatch}"
        f"\n  S3             : {args.write_to_s3}"
    )

    result = run_monitoring_pipeline(
        predictions=predictions,
        outcomes=outcomes,
        reference_data=reference_data,
        model_version=args.model_version,
        report_date=report_date,
        environment=args.environment,
        threshold=args.threshold,
        number_of_bands=args.number_of_bands,
        segment_columns=args.segment_columns,
        bucket=args.output_bucket,
        analytics_prefix=args.analytics_prefix,
        cloudwatch_namespace=args.cloudwatch_namespace,
        publish_to_cloudwatch=args.publish_to_cloudwatch,
        write_to_s3=args.write_to_s3,
    )

    print_summary(
        result
    )

    sys.exit(
        0
        if result.get(
            "status"
        ) == "success"
        else 1
    )


if __name__ == "__main__":
    main()
