"""Tests for adverse reason summary metrics."""

import pandas as pd
import pytest

from app.monitoring.adverse_reason_metrics import (
    calculate_adverse_reason_summary,
    validate_adverse_reason_data,
)


@pytest.fixture
def adverse_data() -> pd.DataFrame:
    """Return sample adverse reason data."""

    return pd.DataFrame(
        {
            "reason_code": [
                "R01", "R02", "R01", "R03",
                "R01", "R02", "R04", "R01",
            ],
            "adverse_reason": [
                "High utilization",
                "Short history",
                "High utilization",
                "High debt",
                "High utilization",
                "Short history",
                "Low income",
                "High utilization",
            ],
            "contribution": [
                0.15, 0.10, 0.18, 0.12,
                0.14, 0.11, 0.09, 0.16,
            ],
        }
    )


# ---------------------------------------------------------------------------
# validate_adverse_reason_data
# ---------------------------------------------------------------------------

def test_validate_adverse_reason_data_accepts_valid_data(
    adverse_data: pd.DataFrame,
) -> None:
    """Valid data should not raise."""

    validate_adverse_reason_data(adverse_data)


def test_validate_adverse_reason_data_rejects_non_dataframe() -> None:
    """Non-DataFrame input must raise TypeError."""

    with pytest.raises(TypeError, match="must be a pandas DataFrame"):
        validate_adverse_reason_data([])


def test_validate_adverse_reason_data_rejects_empty() -> None:
    """Empty DataFrame must raise ValueError."""

    with pytest.raises(ValueError, match="empty"):
        validate_adverse_reason_data(pd.DataFrame())


def test_validate_adverse_reason_data_rejects_missing_columns() -> None:
    """Missing required columns must raise ValueError."""

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_adverse_reason_data(
            pd.DataFrame({"reason_code": ["R01"]})
        )


# ---------------------------------------------------------------------------
# calculate_adverse_reason_summary
# ---------------------------------------------------------------------------

def test_calculate_adverse_reason_summary_returns_dataframe(
    adverse_data: pd.DataFrame,
) -> None:
    """Result should be a DataFrame."""

    result = calculate_adverse_reason_summary(
        data=adverse_data,
        model_version="v1",
        report_date="2026-08-01",
        environment="test",
    )

    assert isinstance(result, pd.DataFrame)


def test_calculate_adverse_reason_summary_output_columns(
    adverse_data: pd.DataFrame,
) -> None:
    """Result must contain all required output columns."""

    result = calculate_adverse_reason_summary(
        data=adverse_data,
        model_version="v1",
        report_date="2026-08-01",
        environment="test",
    )

    required = {
        "reason_code",
        "adverse_reason",
        "selection_count",
        "selection_rate",
        "average_contribution",
        "model_version",
        "report_date",
        "environment",
    }

    assert required.issubset(set(result.columns))


def test_calculate_adverse_reason_summary_counts(
    adverse_data: pd.DataFrame,
) -> None:
    """R01 appears 4 times in the test data."""

    result = calculate_adverse_reason_summary(
        data=adverse_data,
        model_version="v1",
        report_date="2026-08-01",
        environment="test",
    )

    r01_row = result[result["reason_code"] == "R01"].iloc[0]

    assert r01_row["selection_count"] == 4


def test_calculate_adverse_reason_summary_sorted_by_frequency(
    adverse_data: pd.DataFrame,
) -> None:
    """Result should be sorted by selection_count descending."""

    result = calculate_adverse_reason_summary(
        data=adverse_data,
        model_version="v1",
        report_date="2026-08-01",
        environment="test",
    )

    counts = result["selection_count"].tolist()

    assert counts == sorted(counts, reverse=True)


def test_calculate_adverse_reason_summary_selection_rate_sums_to_one(
    adverse_data: pd.DataFrame,
) -> None:
    """Selection rates should sum to approximately 1.0 when using total rows."""

    result = calculate_adverse_reason_summary(
        data=adverse_data,
        model_version="v1",
        report_date="2026-08-01",
        environment="test",
        total_predictions=len(adverse_data),
    )

    assert abs(result["selection_rate"].sum() - 1.0) < 0.01


def test_calculate_adverse_reason_summary_stamps_metadata(
    adverse_data: pd.DataFrame,
) -> None:
    """Result should be stamped with model_version, report_date, environment."""

    result = calculate_adverse_reason_summary(
        data=adverse_data,
        model_version="model-v99",
        report_date="2026-12-31",
        environment="production",
    )

    assert (result["model_version"] == "model-v99").all()
    assert (result["report_date"] == "2026-12-31").all()
    assert (result["environment"] == "production").all()


def test_calculate_adverse_reason_summary_integer_counts(
    adverse_data: pd.DataFrame,
) -> None:
    """selection_count must be integer type."""

    result = calculate_adverse_reason_summary(
        data=adverse_data,
        model_version="v1",
        report_date="2026-08-01",
        environment="test",
    )

    assert pd.api.types.is_integer_dtype(result["selection_count"])
