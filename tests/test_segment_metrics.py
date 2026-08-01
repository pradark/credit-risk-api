"""Tests for segment metrics calculation."""

import pandas as pd
import pytest

from app.monitoring.segment_metrics import (
    calculate_segment_metrics,
    validate_segment_data,
)


@pytest.fixture
def segment_data() -> pd.DataFrame:
    """Return sample data with segment columns."""

    return pd.DataFrame(
        {
            "application_id": [
                f"APP-{i:04d}"
                for i in range(1, 11)
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
                0, 0, 0, 0, 0, 1, 0, 1, 1, 1,
            ],
            "score_band": [
                "low",
                "low",
                "low",
                "medium",
                "medium",
                "medium",
                "medium",
                "high",
                "high",
                "high",
            ],
        }
    )


def test_validate_segment_data_accepts_valid_data(
    segment_data: pd.DataFrame,
) -> None:
    """Valid segment data should not raise."""

    validate_segment_data(
        segment_data,
        segment_column="score_band",
    )


def test_validate_segment_data_rejects_missing_column(
    segment_data: pd.DataFrame,
) -> None:
    """Missing segment column should raise ValueError."""

    with pytest.raises(
        ValueError,
        match="Missing required columns",
    ):
        validate_segment_data(
            segment_data,
            segment_column="non_existent_column",
        )


def test_calculate_segment_metrics_returns_dataframe(
    segment_data: pd.DataFrame,
) -> None:
    """Result should be a DataFrame."""

    result = calculate_segment_metrics(
        data=segment_data,
        segment_columns=[
            "score_band",
        ],
        model_version="v1",
        report_date="2026-08-01",
        environment="test",
    )

    assert isinstance(
        result,
        pd.DataFrame,
    )


def test_calculate_segment_metrics_returns_all_bands(
    segment_data: pd.DataFrame,
) -> None:
    """Result should contain one row per segment value."""

    result = calculate_segment_metrics(
        data=segment_data,
        segment_columns=[
            "score_band",
        ],
        model_version="v1",
        report_date="2026-08-01",
        environment="test",
    )

    # Three bands: low, medium, high
    assert len(
        result
    ) == 3


def test_calculate_segment_metrics_output_columns(
    segment_data: pd.DataFrame,
) -> None:
    """Result must contain all required output columns."""

    result = calculate_segment_metrics(
        data=segment_data,
        segment_columns=[
            "score_band",
        ],
        model_version="v1",
        report_date="2026-08-01",
        environment="test",
    )

    required_columns = {
        "segment_name",
        "segment_value",
        "record_count",
        "default_count",
        "bad_rate",
        "average_predicted_pd",
        "calibration_gap",
        "model_version",
        "report_date",
        "environment",
    }

    assert required_columns.issubset(
        set(
            result.columns
        )
    )


def test_calculate_segment_metrics_stamps_metadata(
    segment_data: pd.DataFrame,
) -> None:
    """Output should be stamped with model_version, report_date, environment."""

    result = calculate_segment_metrics(
        data=segment_data,
        segment_columns=["score_band"],
        model_version="test-model-v2",
        report_date="2026-12-01",
        environment="production",
    )

    assert (result["model_version"] == "test-model-v2").all()
    assert (result["report_date"] == "2026-12-01").all()
    assert (result["environment"] == "production").all()


def test_calculate_segment_metrics_skips_missing_columns(
    segment_data: pd.DataFrame,
) -> None:
    """Columns not in data should be silently skipped."""

    result = calculate_segment_metrics(
        data=segment_data,
        segment_columns=[
            "score_band",
            "non_existent_column",
        ],
        model_version="v1",
        report_date="2026-08-01",
        environment="test",
    )

    # Only score_band is in data, so still 3 rows
    assert len(result) == 3


def test_calculate_segment_metrics_returns_empty_df_when_no_valid_columns(
    segment_data: pd.DataFrame,
) -> None:
    """No valid segment columns should return an empty DataFrame."""

    result = calculate_segment_metrics(
        data=segment_data,
        segment_columns=["non_existent_col"],
        model_version="v1",
        report_date="2026-08-01",
        environment="test",
    )

    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_calculate_segment_metrics_integer_counts(
    segment_data: pd.DataFrame,
) -> None:
    """record_count and default_count must be integers."""

    result = calculate_segment_metrics(
        data=segment_data,
        segment_columns=["score_band"],
        model_version="v1",
        report_date="2026-08-01",
        environment="test",
    )

    assert pd.api.types.is_integer_dtype(result["record_count"])
    assert pd.api.types.is_integer_dtype(result["default_count"])
