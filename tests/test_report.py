"""Tests for monitoring report utilities."""

from pathlib import Path

import pandas as pd
import pytest

from app.monitoring.report import (
    build_metrics_summary,
    ensure_report_directory,
    generate_monitoring_dashboard,
)


@pytest.fixture
def prediction_data() -> pd.DataFrame:
    """Return deterministic prediction and outcome data."""

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


def test_ensure_report_directory(
    tmp_path: Path,
) -> None:
    """The report directory should be created and returned."""

    report_directory = (
        tmp_path
        / "reports"
    )

    result = ensure_report_directory(
        report_directory
    )

    assert result == report_directory
    assert report_directory.exists()
    assert report_directory.is_dir()


def test_ensure_report_directory_accepts_existing_directory(
    tmp_path: Path,
) -> None:
    """An existing report directory should be reused."""

    report_directory = (
        tmp_path
        / "reports"
    )

    report_directory.mkdir()

    result = ensure_report_directory(
        report_directory
    )

    assert result == report_directory
    assert report_directory.exists()


def test_build_metrics_summary() -> None:
    """A one-row metrics DataFrame should become a dictionary."""

    metrics = pd.DataFrame(
        [
            {
                "auc": 0.84,
                "ks": 0.52,
                "gini": 0.68,
            }
        ]
    )

    summary = build_metrics_summary(
        metrics
    )

    assert summary == {
        "auc": 0.84,
        "ks": 0.52,
        "gini": 0.68,
    }


def test_build_metrics_summary_rejects_non_dataframe() -> None:
    """Metrics input must be a pandas DataFrame."""

    with pytest.raises(
        TypeError,
        match="must be a pandas DataFrame",
    ):
        build_metrics_summary(
            []
        )


def test_build_metrics_summary_rejects_empty_dataframe() -> None:
    """An empty metrics DataFrame should be rejected."""

    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        build_metrics_summary(
            pd.DataFrame()
        )


def test_build_metrics_summary_rejects_multiple_rows() -> None:
    """The metrics summary must contain exactly one row."""

    metrics = pd.DataFrame(
        [
            {
                "auc": 0.80,
            },
            {
                "auc": 0.82,
            },
        ]
    )

    with pytest.raises(
        ValueError,
        match="exactly one row",
    ):
        build_metrics_summary(
            metrics
        )


def test_generate_monitoring_dashboard_returns_expected_keys(
    prediction_data: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """The dashboard should return all generated artifacts."""

    result = generate_monitoring_dashboard(
        prediction_data=prediction_data,
        output_directory=tmp_path,
        threshold=0.50,
        number_of_bands=5,
    )

    assert set(
        result
    ) == {
        "output_directory",
        "performance_metrics",
        "performance_metrics_path",
        "calibration_table",
        "calibration_table_path",
        "expected_calibration_error",
        "maximum_calibration_error",
        "roc_curve_path",
        "expected_vs_actual_path",
    }


def test_generate_monitoring_dashboard_creates_csv_files(
    prediction_data: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """The dashboard should create both monitoring CSV files."""

    result = generate_monitoring_dashboard(
        prediction_data=prediction_data,
        output_directory=tmp_path,
        threshold=0.50,
        number_of_bands=5,
    )

    performance_path = result[
        "performance_metrics_path"
    ]

    calibration_path = result[
        "calibration_table_path"
    ]

    assert performance_path.exists()
    assert calibration_path.exists()

    assert performance_path.stat().st_size > 0
    assert calibration_path.stat().st_size > 0


def test_generate_monitoring_dashboard_creates_plot_files(
    prediction_data: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """The dashboard should create both monitoring PNG files."""

    result = generate_monitoring_dashboard(
        prediction_data=prediction_data,
        output_directory=tmp_path,
        threshold=0.50,
        number_of_bands=5,
    )

    roc_path = result[
        "roc_curve_path"
    ]

    expected_vs_actual_path = result[
        "expected_vs_actual_path"
    ]

    assert roc_path.exists()
    assert expected_vs_actual_path.exists()

    assert roc_path.stat().st_size > 0
    assert expected_vs_actual_path.stat().st_size > 0


def test_generate_monitoring_dashboard_returns_metric_tables(
    prediction_data: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """The dashboard should return calculated metric DataFrames."""

    result = generate_monitoring_dashboard(
        prediction_data=prediction_data,
        output_directory=tmp_path,
        threshold=0.50,
        number_of_bands=5,
    )

    performance_metrics = result[
        "performance_metrics"
    ]

    calibration_table = result[
        "calibration_table"
    ]

    assert isinstance(
        performance_metrics,
        pd.DataFrame,
    )

    assert isinstance(
        calibration_table,
        pd.DataFrame,
    )

    assert len(
        performance_metrics
    ) == 1

    assert len(
        calibration_table
    ) == 5

    assert performance_metrics.loc[
        0,
        "record_count",
    ] == 10

    assert performance_metrics.loc[
        0,
        "auc",
    ] == pytest.approx(
        0.9583
    )


def test_generate_monitoring_dashboard_returns_calibration_metrics(
    prediction_data: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """ECE and MCE should be calculated from the calibration table."""

    result = generate_monitoring_dashboard(
        prediction_data=prediction_data,
        output_directory=tmp_path,
        threshold=0.50,
        number_of_bands=5,
    )

    calibration_table = result[
        "calibration_table"
    ]

    expected_ece = (
        calibration_table[
            "population_percentage"
        ]
        * calibration_table[
            "absolute_calibration_gap"
        ]
    ).sum()

    expected_mce = calibration_table[
        "absolute_calibration_gap"
    ].max()

    assert result[
        "expected_calibration_error"
    ] == pytest.approx(
        expected_ece
    )

    assert result[
        "maximum_calibration_error"
    ] == pytest.approx(
        expected_mce
    )


def test_generated_csv_matches_returned_dataframes(
    prediction_data: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """Persisted CSV data should match the returned metric tables."""

    result = generate_monitoring_dashboard(
        prediction_data=prediction_data,
        output_directory=tmp_path,
        threshold=0.50,
        number_of_bands=5,
    )

    stored_performance = pd.read_csv(
        result[
            "performance_metrics_path"
        ]
    )

    stored_calibration = pd.read_csv(
        result[
            "calibration_table_path"
        ]
    )

    pd.testing.assert_frame_equal(
        stored_performance,
        result[
            "performance_metrics"
        ],
        check_dtype=False,
    )

    pd.testing.assert_frame_equal(
        stored_calibration,
        result[
            "calibration_table"
        ],
        check_dtype=False,
    )