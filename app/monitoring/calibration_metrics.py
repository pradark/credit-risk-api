"""Utilities for evaluating probability calibration."""

from __future__ import annotations

from typing import Final

import pandas as pd


DEFAULT_ACTUAL_COLUMN: Final[str] = "actual_default"
DEFAULT_PROBABILITY_COLUMN: Final[str] = "predicted_probability"


def validate_calibration_data(
    data: pd.DataFrame,
    actual_column: str = DEFAULT_ACTUAL_COLUMN,
    probability_column: str = DEFAULT_PROBABILITY_COLUMN,
) -> None:
    """Validate data required for calibration analysis."""

    if not isinstance(
        data,
        pd.DataFrame,
    ):
        raise TypeError(
            "Calibration data must be a pandas DataFrame."
        )

    if data.empty:
        raise ValueError(
            "Calibration data is empty."
        )

    required_columns = {
        actual_column,
        probability_column,
    }

    missing_columns = sorted(
        required_columns.difference(
            data.columns
        )
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if data[
        actual_column
    ].isna().any():
        raise ValueError(
            f"{actual_column} contains missing values."
        )

    if data[
        probability_column
    ].isna().any():
        raise ValueError(
            f"{probability_column} contains missing values."
        )

    actual_values = set(
        data[
            actual_column
        ].unique()
    )

    if not actual_values.issubset(
        {
            0,
            1,
        }
    ):
        raise ValueError(
            f"{actual_column} must contain only 0 and 1."
        )

    if not data[
        probability_column
    ].between(
        0.0,
        1.0,
    ).all():
        raise ValueError(
            f"{probability_column} must contain values "
            "between 0 and 1."
        )

def validate_number_of_bands(
    number_of_bands: int,
) -> None:
    """Validate the requested number of calibration bands."""

    if not isinstance(
        number_of_bands,
        int,
    ):
        raise TypeError(
            "number_of_bands must be an integer."
        )

    if number_of_bands < 2:
        raise ValueError(
            "number_of_bands must be at least 2."
        )


def assign_calibration_bands(
    data: pd.DataFrame,
    number_of_bands: int = 10,
    probability_column: str = DEFAULT_PROBABILITY_COLUMN,
) -> pd.Series:
    """Assign observations to quantile-based calibration bands."""

    validate_number_of_bands(
        number_of_bands
    )

    if not isinstance(
        data,
        pd.DataFrame,
    ):
        raise TypeError(
            "Calibration data must be a pandas DataFrame."
        )

    if probability_column not in data.columns:
        raise ValueError(
            f"Missing required column: {probability_column}"
        )

    ranked_probabilities = data[
        probability_column
    ].rank(
        method="first"
    )

    band_numbers = pd.qcut(
        ranked_probabilities,
        q=min(
            number_of_bands,
            len(
                data
            ),
        ),
        labels=False,
        duplicates="drop",
    )

    return (
        band_numbers
        .astype(
            int
        )
        + 1
    )


def calculate_calibration_table(
    data: pd.DataFrame,
    number_of_bands: int = 10,
    actual_column: str = DEFAULT_ACTUAL_COLUMN,
    probability_column: str = DEFAULT_PROBABILITY_COLUMN,
) -> pd.DataFrame:
    """Calculate predicted versus actual default rates by score band."""

    validate_calibration_data(
        data=data,
        actual_column=actual_column,
        probability_column=probability_column,
    )

    validate_number_of_bands(
        number_of_bands
    )

    working_data = data[
        [
            actual_column,
            probability_column,
        ]
    ].copy()

    working_data[
        "calibration_band"
    ] = assign_calibration_bands(
        data=working_data,
        number_of_bands=number_of_bands,
        probability_column=probability_column,
    )

    calibration_table = (
        working_data.groupby(
            "calibration_band",
            as_index=False,
            observed=True,
        )
        .agg(
            record_count=(
                actual_column,
                "size",
            ),
            default_count=(
                actual_column,
                "sum",
            ),
            minimum_predicted_pd=(
                probability_column,
                "min",
            ),
            maximum_predicted_pd=(
                probability_column,
                "max",
            ),
            average_predicted_pd=(
                probability_column,
                "mean",
            ),
            actual_default_rate=(
                actual_column,
                "mean",
            ),
        )
    )

    calibration_table[
        "population_percentage"
    ] = (
        calibration_table[
            "record_count"
        ]
        / len(
            working_data
        )
    )

    calibration_table[
        "calibration_gap"
    ] = (
        calibration_table[
            "average_predicted_pd"
        ]
        - calibration_table[
            "actual_default_rate"
        ]
    )

    calibration_table[
        "absolute_calibration_gap"
    ] = calibration_table[
        "calibration_gap"
    ].abs()

    count_columns = [
        "calibration_band",
        "record_count",
        "default_count",
    ]

    calibration_table[
        count_columns
    ] = calibration_table[
        count_columns
    ].astype(
        int
    )

    numeric_columns = [
        "minimum_predicted_pd",
        "maximum_predicted_pd",
        "average_predicted_pd",
        "actual_default_rate",
        "population_percentage",
        "calibration_gap",
        "absolute_calibration_gap",
    ]

    calibration_table[
        numeric_columns
    ] = calibration_table[
        numeric_columns
    ].round(
        4
    )

    return calibration_table.sort_values(
        "calibration_band"
    ).reset_index(
        drop=True
    )

def calculate_expected_calibration_error(
    calibration_table: pd.DataFrame,
) -> float:
    """Calculate population-weighted expected calibration error."""

    required_columns = {
        "population_percentage",
        "absolute_calibration_gap",
    }

    missing_columns = sorted(
        required_columns.difference(
            calibration_table.columns
        )
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    return float(
        (
            calibration_table[
                "population_percentage"
            ]
            * calibration_table[
                "absolute_calibration_gap"
            ]
        ).sum()
    )


def calculate_maximum_calibration_error(
    calibration_table: pd.DataFrame,
) -> float:
    """Calculate the largest absolute calibration gap."""

    if (
        "absolute_calibration_gap"
        not in calibration_table.columns
    ):
        raise ValueError(
            "Missing required column: "
            "absolute_calibration_gap"
        )

    if calibration_table.empty:
        raise ValueError(
            "Calibration table is empty."
        )

    return float(
        calibration_table[
            "absolute_calibration_gap"
        ].max()
    )