"""Tests for probability calibration metrics."""

import pandas as pd
import pytest

from app.monitoring.calibration_metrics import (
    assign_calibration_bands,
    calculate_calibration_table,
    validate_calibration_data,
    validate_number_of_bands,
    calculate_expected_calibration_error,
    calculate_maximum_calibration_error,
)


@pytest.fixture
def calibration_data() -> pd.DataFrame:
    """Return a deterministic calibration dataset."""

    return pd.DataFrame(
        {
            "predicted_probability": [
                0.05,
                0.10,
                0.20,
                0.30,
                0.40,
                0.50,
                0.60,
                0.70,
                0.80,
                0.90,
            ],
            "actual_default": [
                0,
                0,
                0,
                1,
                0,
                1,
                1,
                0,
                1,
                1,
            ],
        }
    )


def test_validate_accepts_valid_data(
    calibration_data: pd.DataFrame,
) -> None:
    validate_calibration_data(
        calibration_data
    )


def test_validate_rejects_non_dataframe() -> None:
    with pytest.raises(
        TypeError,
        match="must be a pandas DataFrame",
    ):
        validate_calibration_data(
            []
        )


def test_validate_rejects_empty_dataframe() -> None:
    with pytest.raises(
        ValueError,
        match="empty",
    ):
        validate_calibration_data(
            pd.DataFrame()
        )


def test_validate_rejects_missing_columns() -> None:
    with pytest.raises(
        ValueError,
        match="Missing required columns",
    ):
        validate_calibration_data(
            pd.DataFrame(
                {
                    "actual_default": [
                        0,
                        1,
                    ]
                }
            )
        )


def test_validate_rejects_missing_actual_values() -> None:
    data = pd.DataFrame(
        {
            "actual_default": [
                0,
                None,
            ],
            "predicted_probability": [
                0.10,
                0.80,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="actual_default contains missing values",
    ):
        validate_calibration_data(
            data
        )


def test_validate_rejects_missing_probabilities() -> None:
    data = pd.DataFrame(
        {
            "actual_default": [
                0,
                1,
            ],
            "predicted_probability": [
                0.10,
                None,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match=(
            "predicted_probability contains "
            "missing values"
        ),
    ):
        validate_calibration_data(
            data
        )


def test_validate_rejects_invalid_actual_values() -> None:
    data = pd.DataFrame(
        {
            "actual_default": [
                0,
                2,
            ],
            "predicted_probability": [
                0.10,
                0.80,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="must contain only 0 and 1",
    ):
        validate_calibration_data(
            data
        )


def test_validate_rejects_invalid_probabilities() -> None:
    data = pd.DataFrame(
        {
            "actual_default": [
                0,
                1,
            ],
            "predicted_probability": [
                0.10,
                1.20,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        validate_calibration_data(
            data
        )


def test_validate_number_of_bands_accepts_valid_value() -> None:
    validate_number_of_bands(
        10
    )


@pytest.mark.parametrize(
    "number_of_bands",
    [
        0,
        1,
        -1,
    ],
)
def test_validate_number_of_bands_rejects_small_values(
    number_of_bands: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="must be at least 2",
    ):
        validate_number_of_bands(
            number_of_bands
        )


def test_validate_number_of_bands_rejects_non_integer() -> None:
    with pytest.raises(
        TypeError,
        match="must be an integer",
    ):
        validate_number_of_bands(
            5.5
        )


def test_assign_calibration_bands(
    calibration_data: pd.DataFrame,
) -> None:
    result = assign_calibration_bands(
        calibration_data,
        number_of_bands=5,
    )

    assert result.tolist() == [
        1,
        1,
        2,
        2,
        3,
        3,
        4,
        4,
        5,
        5,
    ]


def test_assign_calibration_bands_is_deterministic(
    calibration_data: pd.DataFrame,
) -> None:
    first = assign_calibration_bands(
        calibration_data,
        number_of_bands=5,
    )

    second = assign_calibration_bands(
        calibration_data,
        number_of_bands=5,
    )

    pd.testing.assert_series_equal(
        first,
        second,
    )


def test_calculate_calibration_table_columns(
    calibration_data: pd.DataFrame,
) -> None:
    result = calculate_calibration_table(
        calibration_data,
        number_of_bands=5,
    )

    assert result.columns.tolist() == [
        "calibration_band",
        "record_count",
        "default_count",
        "minimum_predicted_pd",
        "maximum_predicted_pd",
        "average_predicted_pd",
        "actual_default_rate",
        "population_percentage",
        "calibration_gap",
        "absolute_calibration_gap",
    ]


def test_calculate_calibration_table_returns_five_bands(
    calibration_data: pd.DataFrame,
) -> None:
    result = calculate_calibration_table(
        calibration_data,
        number_of_bands=5,
    )

    assert len(
        result
    ) == 5

    assert result[
        "calibration_band"
    ].tolist() == [
        1,
        2,
        3,
        4,
        5,
    ]


def test_calibration_table_counts_sum_to_total(
    calibration_data: pd.DataFrame,
) -> None:
    result = calculate_calibration_table(
        calibration_data,
        number_of_bands=5,
    )

    assert result[
        "record_count"
    ].sum() == len(
        calibration_data
    )

    assert result[
        "default_count"
    ].sum() == calibration_data[
        "actual_default"
    ].sum()


def test_calibration_population_percentages_sum_to_one(
    calibration_data: pd.DataFrame,
) -> None:
    result = calculate_calibration_table(
        calibration_data,
        number_of_bands=5,
    )

    assert result[
        "population_percentage"
    ].sum() == pytest.approx(
        1.0
    )


def test_first_calibration_band_statistics(
    calibration_data: pd.DataFrame,
) -> None:
    result = calculate_calibration_table(
        calibration_data,
        number_of_bands=5,
    )

    first_band = result.iloc[
        0
    ]

    assert first_band[
        "record_count"
    ] == 2

    assert first_band[
        "default_count"
    ] == 0

    assert first_band[
        "minimum_predicted_pd"
    ] == pytest.approx(
        0.05
    )

    assert first_band[
        "maximum_predicted_pd"
    ] == pytest.approx(
        0.10
    )

    assert first_band[
        "average_predicted_pd"
    ] == pytest.approx(
        0.075
    )

    assert first_band[
        "actual_default_rate"
    ] == pytest.approx(
        0.0
    )

    assert first_band[
        "calibration_gap"
    ] == pytest.approx(
        0.075
    )

    assert first_band[
        "absolute_calibration_gap"
    ] == pytest.approx(
        0.075
    )


def test_calibration_table_uses_fewer_bands_than_rows(
    calibration_data: pd.DataFrame,
) -> None:
    small_data = calibration_data.head(
        3
    )

    result = calculate_calibration_table(
        small_data,
        number_of_bands=10,
    )

    assert len(
        result
    ) == 3

    assert result[
        "record_count"
    ].sum() == 3


def test_calibration_table_preserves_integer_counts(
    calibration_data: pd.DataFrame,
) -> None:
    result = calculate_calibration_table(
        calibration_data,
        number_of_bands=5,
    )

    integer_columns = [
        "calibration_band",
        "record_count",
        "default_count",
    ]

    for column in integer_columns:
        assert pd.api.types.is_integer_dtype(
            result[
                column
            ]
        )

def test_calculate_expected_calibration_error(
    calibration_data: pd.DataFrame,
) -> None:
    table = calculate_calibration_table(
        calibration_data,
        number_of_bands=5,
    )

    result = calculate_expected_calibration_error(
        table
    )

    expected = (
        table[
            "population_percentage"
        ]
        * table[
            "absolute_calibration_gap"
        ]
    ).sum()

    assert result == pytest.approx(
        expected
    )


def test_calculate_maximum_calibration_error(
    calibration_data: pd.DataFrame,
) -> None:
    table = calculate_calibration_table(
        calibration_data,
        number_of_bands=5,
    )

    result = calculate_maximum_calibration_error(
        table
    )

    assert result == pytest.approx(
        table[
            "absolute_calibration_gap"
        ].max()
    )


def test_expected_calibration_error_rejects_missing_columns() -> None:
    with pytest.raises(
        ValueError,
        match="Missing required columns",
    ):
        calculate_expected_calibration_error(
            pd.DataFrame(
                {
                    "population_percentage": [
                        1.0
                    ]
                }
            )
        )


def test_maximum_calibration_error_rejects_empty_table() -> None:
    with pytest.raises(
        ValueError,
        match="empty",
    ):
        calculate_maximum_calibration_error(
            pd.DataFrame(
                columns=[
                    "absolute_calibration_gap"
                ]
            )
        )