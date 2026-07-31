"""Tests for the SHAP background-data loader."""

from pathlib import Path

import pandas as pd
import pytest

from app.explainability.background_loader import (
    load_background_data,
    sample_background_data,
    validate_background_dataframe,
    validate_background_sample_size,
)


FEATURE_NAMES = [
    "feature_a",
    "feature_b",
    "feature_c",
]


@pytest.fixture
def valid_background_data() -> pd.DataFrame:
    """Return valid generic background data."""

    return pd.DataFrame(
        {
            "feature_a": [
                1.0,
                2.0,
                3.0,
                4.0,
            ],
            "feature_b": [
                10.0,
                20.0,
                30.0,
                40.0,
            ],
            "feature_c": [
                100,
                200,
                300,
                400,
            ],
            "unused_column": [
                "a",
                "b",
                "c",
                "d",
            ],
        }
    )


def test_validate_background_sample_size_accepts_integer() -> None:
    result = validate_background_sample_size(
        100
    )

    assert result == 100


def test_validate_background_sample_size_accepts_none() -> None:
    result = validate_background_sample_size(
        None
    )

    assert result is None


@pytest.mark.parametrize(
    "sample_size",
    [
        0,
        -1,
    ],
)
def test_validate_background_sample_size_rejects_invalid_value(
    sample_size: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="sample_size must be at least 1",
    ):
        validate_background_sample_size(
            sample_size
        )


def test_validate_background_sample_size_rejects_non_integer() -> None:
    with pytest.raises(
        TypeError,
        match=(
            "sample_size must be a positive "
            "integer or None"
        ),
    ):
        validate_background_sample_size(
            10.5
        )


def test_validate_background_dataframe_accepts_valid_data(
    valid_background_data: pd.DataFrame,
) -> None:
    result = validate_background_dataframe(
        background_data=valid_background_data,
        feature_names=FEATURE_NAMES,
    )

    assert result == FEATURE_NAMES


def test_validate_background_dataframe_rejects_non_dataframe() -> None:
    with pytest.raises(
        TypeError,
        match=(
            "background_data must be "
            "a pandas DataFrame"
        ),
    ):
        validate_background_dataframe(
            background_data=[],
            feature_names=FEATURE_NAMES,
        )


def test_validate_background_dataframe_rejects_empty_data() -> None:
    empty_data = pd.DataFrame(
        columns=FEATURE_NAMES
    )

    with pytest.raises(
        ValueError,
        match="background_data cannot be empty",
    ):
        validate_background_dataframe(
            background_data=empty_data,
            feature_names=FEATURE_NAMES,
        )


def test_validate_background_dataframe_rejects_empty_features(
    valid_background_data: pd.DataFrame,
) -> None:
    with pytest.raises(
        ValueError,
        match="feature_names cannot be empty",
    ):
        validate_background_dataframe(
            background_data=valid_background_data,
            feature_names=[],
        )


def test_validate_background_dataframe_rejects_duplicate_features(
    valid_background_data: pd.DataFrame,
) -> None:
    with pytest.raises(
        ValueError,
        match=(
            "feature_names must not "
            "contain duplicates"
        ),
    ):
        validate_background_dataframe(
            background_data=valid_background_data,
            feature_names=[
                "feature_a",
                "feature_a",
            ],
        )


def test_validate_background_dataframe_rejects_missing_features(
    valid_background_data: pd.DataFrame,
) -> None:
    with pytest.raises(
        ValueError,
        match=(
            "Background data is missing "
            "required features"
        ),
    ):
        validate_background_dataframe(
            background_data=valid_background_data,
            feature_names=[
                "feature_a",
                "missing_feature",
            ],
        )


def test_validate_background_dataframe_rejects_null_values(
    valid_background_data: pd.DataFrame,
) -> None:
    invalid_data = valid_background_data.copy()

    invalid_data.loc[
        0,
        "feature_b",
    ] = None

    with pytest.raises(
        ValueError,
        match=(
            "Background data contains null values"
        ),
    ):
        validate_background_dataframe(
            background_data=invalid_data,
            feature_names=FEATURE_NAMES,
        )


def test_validate_background_dataframe_rejects_non_numeric_features(
    valid_background_data: pd.DataFrame,
) -> None:
    invalid_data = valid_background_data.copy()

    invalid_data[
        "feature_b"
    ] = [
        "a",
        "b",
        "c",
        "d",
    ]

    with pytest.raises(
        ValueError,
        match=(
            "Background data contains non-numeric "
            "model features"
        ),
    ):
        validate_background_dataframe(
            background_data=invalid_data,
            feature_names=FEATURE_NAMES,
        )


def test_sample_background_data_returns_requested_sample(
    valid_background_data: pd.DataFrame,
) -> None:
    result = sample_background_data(
        valid_background_data,
        sample_size=2,
        random_state=42,
    )

    assert len(result) == 2
    assert result.index.tolist() == [
        0,
        1,
    ]


def test_sample_background_data_is_deterministic(
    valid_background_data: pd.DataFrame,
) -> None:
    first = sample_background_data(
        valid_background_data,
        sample_size=2,
        random_state=42,
    )

    second = sample_background_data(
        valid_background_data,
        sample_size=2,
        random_state=42,
    )

    pd.testing.assert_frame_equal(
        first,
        second,
    )


def test_sample_background_data_returns_all_rows_when_size_is_large(
    valid_background_data: pd.DataFrame,
) -> None:
    result = sample_background_data(
        valid_background_data,
        sample_size=100,
    )

    assert len(result) == len(
        valid_background_data
    )


def test_sample_background_data_returns_all_rows_for_none(
    valid_background_data: pd.DataFrame,
) -> None:
    result = sample_background_data(
        valid_background_data,
        sample_size=None,
    )

    assert len(result) == len(
        valid_background_data
    )


def test_load_background_data_orders_and_samples_columns(
    tmp_path: Path,
    valid_background_data: pd.DataFrame,
) -> None:
    path = (
        tmp_path
        / "background.parquet"
    )

    valid_background_data.to_parquet(
        path,
        index=False,
    )

    result = load_background_data(
        path=path,
        feature_names=[
            "feature_c",
            "feature_a",
            "feature_b",
        ],
        sample_size=2,
        random_state=42,
    )

    assert result.shape == (
        2,
        3,
    )

    assert result.columns.tolist() == [
        "feature_c",
        "feature_a",
        "feature_b",
    ]


def test_load_background_data_rejects_missing_file(
    tmp_path: Path,
) -> None:
    missing_path = (
        tmp_path
        / "missing.parquet"
    )

    with pytest.raises(
        FileNotFoundError,
        match=(
            "Explanation background data "
            "was not found"
        ),
    ):
        load_background_data(
            path=missing_path,
            feature_names=FEATURE_NAMES,
            sample_size=2,
        )
