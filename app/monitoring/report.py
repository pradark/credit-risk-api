"""Utilities for generating model monitoring reports."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from app.monitoring.calibration_metrics import (
    calculate_calibration_table,
    calculate_expected_calibration_error,
    calculate_maximum_calibration_error,
)
from app.monitoring.performance_metrics import (
    calculate_performance_metrics,
)
from app.monitoring.plotting import (
    plot_expected_vs_actual_default_rate,
    plot_roc_curve,
)


def ensure_report_directory(
    output_directory: str | Path,
) -> Path:
    """Create the report directory if it does not exist."""

    path = Path(
        output_directory
    )

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


def build_metrics_summary(
    performance_metrics: pd.DataFrame,
) -> dict[str, object]:
    """Convert one-row performance metrics into a dictionary."""

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

    return (
        performance_metrics
        .iloc[
            0
        ]
        .to_dict()
    )


def generate_monitoring_dashboard(
    prediction_data: pd.DataFrame,
    output_directory: str | Path,
    threshold: float = 0.50,
    number_of_bands: int = 10,
    actual_column: str = "actual_default",
    probability_column: str = "predicted_probability",
) -> dict[str, object]:
    """Generate monitoring metrics, tables, and plots.

    This function coordinates the existing monitoring modules. It does
    not calculate AUC, KS, Gini, calibration, or chart data directly.
    """

    report_directory = ensure_report_directory(
        output_directory
    )

    performance_metrics = calculate_performance_metrics(
        data=prediction_data,
        threshold=threshold,
        actual_column=actual_column,
        probability_column=probability_column,
    )

    calibration_table = calculate_calibration_table(
        data=prediction_data,
        number_of_bands=number_of_bands,
        actual_column=actual_column,
        probability_column=probability_column,
    )

    expected_calibration_error = (
        calculate_expected_calibration_error(
            calibration_table
        )
    )

    maximum_calibration_error = (
        calculate_maximum_calibration_error(
            calibration_table
        )
    )

    performance_metrics_path = (
        report_directory
        / "performance_metrics.csv"
    )

    calibration_table_path = (
        report_directory
        / "calibration_table.csv"
    )

    roc_curve_path = (
        report_directory
        / "roc_curve.png"
    )

    expected_vs_actual_path = (
        report_directory
        / "expected_vs_actual.png"
    )

    performance_metrics.to_csv(
        performance_metrics_path,
        index=False,
    )

    calibration_table.to_csv(
        calibration_table_path,
        index=False,
    )

    roc_figure = plot_roc_curve(
        data=prediction_data,
        actual_column=actual_column,
        probability_column=probability_column,
        output_path=roc_curve_path,
    )

    calibration_figure = (
        plot_expected_vs_actual_default_rate(
            calibration_table=calibration_table,
            output_path=expected_vs_actual_path,
        )
    )

    plt.close(
        roc_figure
    )

    plt.close(
        calibration_figure
    )

    return {
        "output_directory": report_directory,
        "performance_metrics": performance_metrics,
        "performance_metrics_path": performance_metrics_path,
        "calibration_table": calibration_table,
        "calibration_table_path": calibration_table_path,
        "expected_calibration_error": (
            expected_calibration_error
        ),
        "maximum_calibration_error": (
            maximum_calibration_error
        ),
        "roc_curve_path": roc_curve_path,
        "expected_vs_actual_path": expected_vs_actual_path,
    }

def generate_html_report(
    dashboard: dict[str, object],
) -> Path:
    """Deprecated: HTML reports are not part of the production workflow.

    This function is retained for backward compatibility only. The
    production monitoring workflow uses Amazon CloudWatch, S3 Parquet
    datasets, AWS Glue, Amazon Athena, and Amazon QuickSight.

    Calling this function in any production or scheduled context is a
    policy violation. Use the monitoring pipeline and BI dataset writer
    instead.
    """

    import warnings

    warnings.warn(
        "generate_html_report() is deprecated and must not be called "
        "in the production monitoring workflow. Use run_monitoring_pipeline() "
        "and write_monitoring_datasets() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    output_directory = dashboard[
        "output_directory"
    ]

    report_path = (
        output_directory
        / "monitoring_report.html"
    )

    return report_path