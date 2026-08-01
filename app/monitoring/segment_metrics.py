"""Calculate model performance metrics segmented by configurable columns."""

from __future__ import annotations

from typing import Final

import pandas as pd

from app.monitoring.performance_metrics import (
    DEFAULT_ACTUAL_COLUMN,
    DEFAULT_CLASSIFICATION_THRESHOLD,
    DEFAULT_PROBABILITY_COLUMN,
)


DEFAULT_SEGMENT_COLUMNS: Final[list[str]] = [
    "score_band",
]


def validate_segment_data(
    data: pd.DataFrame,
    segment_column: str,
    actual_column: str = DEFAULT_ACTUAL_COLUMN,
    probability_column: str = DEFAULT_PROBABILITY_COLUMN,
) -> None:
    """Validate data for segment metric calculation."""

    if not isinstance(
        data,
        pd.DataFrame,
    ):
        raise TypeError(
            "Segment data must be a pandas DataFrame."
        )

    if data.empty:
        raise ValueError(
            "Segment data is empty."
        )

    required = {
        actual_column,
        probability_column,
        segment_column,
    }

    missing = sorted(
        required.difference(
            data.columns
        )
    )

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )


def calculate_segment_metrics(
    data: pd.DataFrame,
    segment_columns: list[str],
    model_version: str,
    report_date: str,
    environment: str,
    actual_column: str = DEFAULT_ACTUAL_COLUMN,
    probability_column: str = DEFAULT_PROBABILITY_COLUMN,
) -> pd.DataFrame:
    """Calculate per-segment model performance metrics.

    For each segment column and each segment value, computes record
    count, default count, bad rate, average predicted PD, and
    calibration gap.

    Segments that do not exist in data are skipped. Missing values
    within a segment column are dropped before aggregation.

    Parameters
    ----------
    data:
        Joined prediction and outcome DataFrame.
    segment_columns:
        Column names to segment by. Columns not present in data are
        skipped with a warning appended to the result.
    model_version:
        Model version tag.
    report_date:
        Report date string.
    environment:
        Deployment environment.
    actual_column:
        Name of the actual outcome column.
    probability_column:
        Name of the predicted probability column.

    Returns
    -------
    pd.DataFrame
        Segment metrics with columns: segment_name, segment_value,
        record_count, default_count, bad_rate, average_predicted_pd,
        calibration_gap, model_version, report_date, environment.
    """

    if not isinstance(
        data,
        pd.DataFrame,
    ):
        raise TypeError(
            "Segment data must be a pandas DataFrame."
        )

    if data.empty:
        raise ValueError(
            "Segment data is empty."
        )

    rows: list[dict[str, object]] = []

    for segment_column in segment_columns:
        if segment_column not in data.columns:
            continue

        segment_data = data[
            [
                actual_column,
                probability_column,
                segment_column,
            ]
        ].dropna(
            subset=[
                segment_column
            ]
        )

        if segment_data.empty:
            continue

        grouped = segment_data.groupby(
            segment_column,
            observed=True,
        )

        for segment_value, group in grouped:
            record_count = len(
                group
            )

            default_count = int(
                group[
                    actual_column
                ].sum()
            )

            bad_rate = float(
                group[
                    actual_column
                ].mean()
            )

            average_predicted_pd = float(
                group[
                    probability_column
                ].mean()
            )

            calibration_gap = round(
                average_predicted_pd - bad_rate,
                4,
            )

            rows.append(
                {
                    "segment_name": segment_column,
                    "segment_value": str(
                        segment_value
                    ),
                    "record_count": record_count,
                    "default_count": default_count,
                    "bad_rate": round(
                        bad_rate,
                        4,
                    ),
                    "average_predicted_pd": round(
                        average_predicted_pd,
                        4,
                    ),
                    "calibration_gap": calibration_gap,
                    "model_version": model_version,
                    "report_date": report_date,
                    "environment": environment,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
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
            ]
        )

    result = pd.DataFrame(
        rows
    )

    result[
        [
            "record_count",
            "default_count",
        ]
    ] = result[
        [
            "record_count",
            "default_count",
        ]
    ].astype(
        int
    )

    return result.reset_index(
        drop=True
    )
