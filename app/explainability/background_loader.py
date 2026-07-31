"""Load and validate background data for model explanations.

TreeSHAP uses a reference population to calculate local feature
contributions. This module loads that population from Parquet,
validates the required model features, orders the columns correctly,
and optionally samples the data for faster API inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from app.explainability.model_config import (
    EXPLANATION_BACKGROUND_PATH,
    EXPLANATION_BACKGROUND_SAMPLE_SIZE,
    MODEL_FEATURE_NAMES,
)


def validate_background_sample_size(
    sample_size: int | None,
) -> int | None:
    """Validate the optional background sample size.

    Args:
        sample_size:
            Positive integer or ``None`` to retain all rows.

    Returns:
        The validated sample size.

    Raises:
        TypeError:
            If sample_size is not an integer or None.
        ValueError:
            If sample_size is less than one.
    """

    if sample_size is None:
        return None

    if not isinstance(sample_size, int):
        raise TypeError(
            "sample_size must be a positive integer or None"
        )

    if sample_size < 1:
        raise ValueError(
            "sample_size must be at least 1"
        )

    return sample_size


def validate_background_dataframe(
    background_data: pd.DataFrame,
    feature_names: Sequence[str],
) -> list[str]:
    """Validate background data and return normalized feature names.

    Args:
        background_data:
            Reference population.
        feature_names:
            Ordered model feature names.

    Returns:
        Feature names converted to a list.

    Raises:
        TypeError:
            If background_data is not a DataFrame.
        ValueError:
            If the data is empty, features are empty or duplicated,
            required columns are missing, or feature values contain nulls.
    """

    if not isinstance(
        background_data,
        pd.DataFrame,
    ):
        raise TypeError(
            "background_data must be a pandas DataFrame"
        )

    if background_data.empty:
        raise ValueError(
            "background_data cannot be empty"
        )

    normalized_feature_names = list(
        feature_names
    )

    if not normalized_feature_names:
        raise ValueError(
            "feature_names cannot be empty"
        )

    if len(normalized_feature_names) != len(
        set(normalized_feature_names)
    ):
        raise ValueError(
            "feature_names must not contain duplicates"
        )

    missing_features = sorted(
        set(normalized_feature_names).difference(
            background_data.columns
        )
    )

    if missing_features:
        raise ValueError(
            "Background data is missing required features: "
            f"{missing_features}"
        )

    null_counts = (
        background_data[
            normalized_feature_names
        ]
        .isna()
        .sum()
    )

    features_with_nulls = sorted(
        null_counts[
            null_counts > 0
        ].index.tolist()
    )

    if features_with_nulls:
        raise ValueError(
            "Background data contains null values "
            "for features: "
            f"{features_with_nulls}"
        )

    non_numeric_features = [
        feature_name
        for feature_name
        in normalized_feature_names
        if not pd.api.types.is_numeric_dtype(
            background_data[
                feature_name
            ]
        )
    ]

    if non_numeric_features:
        raise ValueError(
            "Background data contains non-numeric "
            "model features: "
            f"{sorted(non_numeric_features)}"
        )

    return normalized_feature_names


def sample_background_data(
    background_data: pd.DataFrame,
    *,
    sample_size: int | None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Return a deterministic background sample.

    If sample_size is None or is at least the number of available rows,
    all background rows are returned.

    Args:
        background_data:
            Validated reference population.
        sample_size:
            Requested sample size or None.
        random_state:
            Random seed used for reproducibility.

    Returns:
        A copied and reindexed DataFrame.
    """

    normalized_sample_size = (
        validate_background_sample_size(
            sample_size
        )
    )

    if (
        normalized_sample_size is None
        or normalized_sample_size
        >= len(background_data)
    ):
        return background_data.copy().reset_index(
            drop=True
        )

    return (
        background_data.sample(
            n=normalized_sample_size,
            random_state=random_state,
            replace=False,
        )
        .reset_index(
            drop=True
        )
    )


def load_background_data(
    *,
    path: str | Path = EXPLANATION_BACKGROUND_PATH,
    feature_names: Sequence[str] = MODEL_FEATURE_NAMES,
    sample_size: int | None = (
        EXPLANATION_BACKGROUND_SAMPLE_SIZE
    ),
    random_state: int = 42,
) -> pd.DataFrame:
    """Load, validate, order, and sample SHAP background data.

    Args:
        path:
            Parquet file containing the reference population.
        feature_names:
            Ordered model feature names.
        sample_size:
            Number of background rows to retain or None for all rows.
        random_state:
            Sampling seed.

    Returns:
        Background DataFrame containing only the model feature columns,
        in model feature order.

    Raises:
        FileNotFoundError:
            If the configured background file does not exist.
        ValueError:
            If the data does not meet validation requirements.
    """

    background_path = Path(path)

    if not background_path.exists():
        raise FileNotFoundError(
            "Explanation background data was not found: "
            f"{background_path}"
        )

    background_data = pd.read_parquet(
        background_path
    )

    normalized_feature_names = (
        validate_background_dataframe(
            background_data=background_data,
            feature_names=feature_names,
        )
    )

    ordered_background_data = (
        background_data[
            normalized_feature_names
        ]
        .copy()
    )

    return sample_background_data(
        ordered_background_data,
        sample_size=sample_size,
        random_state=random_state,
    )
