"""Generic feature-impact calculations for model explanations.

This module converts model explanation values into a long-form DataFrame
containing one row per observation and feature.

The output supports:

- TreeSHAP explanations
- Positive and negative feature rankings
- Comparisons with a background population
- Arbitrary entity identifiers
- Optional observation dates
- Top-N or all-reason selection in downstream modules
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import shap


DEFAULT_ENTITY_ID_COLUMN = "entity_id"
DEFAULT_ATTRIBUTE_NAME_COLUMN = "attribute_name"
DEFAULT_ATTRIBUTE_VALUE_COLUMN = "attribute_value"
DEFAULT_ATTRIBUTE_IMPACT_COLUMN = "attribute_impact"


def validate_feature_names(
    observations: pd.DataFrame,
    feature_names: Sequence[str],
) -> list[str]:
    """Validate and normalize a feature-name collection.

    Args:
        observations:
            DataFrame containing the observations to explain.
        feature_names:
            Ordered model feature names.

    Returns:
        Feature names converted to a list.

    Raises:
        TypeError:
            If observations is not a DataFrame.
        ValueError:
            If feature names are empty, duplicated, or missing from observations.
    """

    if not isinstance(observations, pd.DataFrame):
        raise TypeError(
            "observations must be a pandas DataFrame"
        )

    normalized_feature_names = list(feature_names)

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
            observations.columns
        )
    )

    if missing_features:
        raise ValueError(
            "Observations are missing required features: "
            f"{missing_features}"
        )

    return normalized_feature_names


def validate_identifier_columns(
    observations: pd.DataFrame,
    entity_id_columns: Sequence[str] | None,
    date_column: str | None,
) -> list[str]:
    """Validate optional entity and date identifier columns.

    Args:
        observations:
            DataFrame containing observations.
        entity_id_columns:
            One or more columns that identify each observation.
        date_column:
            Optional observation date column.

    Returns:
        Ordered list of identifier columns.

    Raises:
        ValueError:
            If identifier columns are missing or observation keys are duplicated.
    """

    identifier_columns = list(
        entity_id_columns or []
    )

    if date_column is not None:
        identifier_columns.append(
            date_column
        )

    if len(identifier_columns) != len(
        set(identifier_columns)
    ):
        raise ValueError(
            "Identifier columns must not contain duplicates"
        )

    missing_columns = sorted(
        set(identifier_columns).difference(
            observations.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "Observations are missing identifier columns: "
            f"{missing_columns}"
        )

    if identifier_columns and observations[
        identifier_columns
    ].duplicated().any():
        raise ValueError(
            "Observation identifier combinations must be unique"
        )

    return identifier_columns


def validate_background_data(
    background_data: pd.DataFrame,
    feature_names: Sequence[str],
) -> None:
    """Validate the SHAP background population.

    Args:
        background_data:
            Reference population used by TreeSHAP.
        feature_names:
            Ordered model feature names.

    Raises:
        TypeError:
            If background_data is not a DataFrame.
        ValueError:
            If it is empty or missing required features.
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

    missing_features = sorted(
        set(feature_names).difference(
            background_data.columns
        )
    )

    if missing_features:
        raise ValueError(
            "Background data is missing required features: "
            f"{missing_features}"
        )


def normalize_shap_values(
    shap_output: Any,
    expected_rows: int,
    expected_features: int,
) -> np.ndarray:
    """Normalize SHAP output into a two-dimensional array.

    SHAP can return:

    - shap.Explanation
    - numpy arrays
    - lists of arrays
    - three-dimensional arrays with a class dimension

    For binary classification, the last class is treated as the modeled
    event class.

    Args:
        shap_output:
            Raw output from a SHAP explainer.
        expected_rows:
            Expected number of observations.
        expected_features:
            Expected number of model features.

    Returns:
        Array with shape:
            (expected_rows, expected_features)

    Raises:
        ValueError:
            If the result cannot be normalized to the expected shape.
    """

    if hasattr(
        shap_output,
        "values",
    ):
        values = shap_output.values
    else:
        values = shap_output

    if isinstance(
        values,
        list,
    ):
        if not values:
            raise ValueError(
                "SHAP returned an empty list"
            )

        values = values[-1]

    normalized_values = np.asarray(
        values
    )

    if normalized_values.ndim == 3:
        normalized_values = normalized_values[
            :,
            :,
            -1,
        ]

    if normalized_values.ndim == 1:
        normalized_values = normalized_values.reshape(
            1,
            -1,
        )

    expected_shape = (
        expected_rows,
        expected_features,
    )

    if normalized_values.shape != expected_shape:
        raise ValueError(
            "Unexpected SHAP output shape. "
            f"Expected {expected_shape}, "
            f"received {normalized_values.shape}."
        )

    return normalized_values.astype(
        float
    )


def create_tree_explainer(
    model: Any,
    background_data: pd.DataFrame,
    feature_names: Sequence[str],
) -> shap.TreeExplainer:
    """Create an interventional TreeSHAP explainer.

    Args:
        model:
            Supported tree-based model.
        background_data:
            Background population.
        feature_names:
            Ordered model feature names.

    Returns:
        Configured TreeExplainer.
    """

    return shap.TreeExplainer(
        model=model,
        data=background_data[
            list(feature_names)
        ],
        feature_names=list(
            feature_names
        ),
        feature_perturbation="interventional",
    )


def calculate_tree_shap_values(
    model: Any,
    observations: pd.DataFrame,
    feature_names: Sequence[str],
    background_data: pd.DataFrame,
    *,
    check_additivity: bool = False,
    explainer: Any | None = None,
) -> np.ndarray:
    """Calculate normalized TreeSHAP feature impacts.

    Args:
        model:
            Supported tree-based model.
        observations:
            Observations to explain.
        feature_names:
            Ordered model feature names.
        background_data:
            Background population.
        check_additivity:
            Whether SHAP should perform its additivity check.
        explainer:
            Optional prebuilt or test explainer.

    Returns:
        Two-dimensional SHAP value array.
    """

    normalized_feature_names = validate_feature_names(
        observations=observations,
        feature_names=feature_names,
    )

    validate_background_data(
        background_data=background_data,
        feature_names=normalized_feature_names,
    )

    active_explainer = (
        explainer
        if explainer is not None
        else create_tree_explainer(
            model=model,
            background_data=background_data,
            feature_names=normalized_feature_names,
        )
    )

    shap_output = active_explainer(
        observations[
            normalized_feature_names
        ],
        check_additivity=check_additivity,
    )

    return normalize_shap_values(
        shap_output=shap_output,
        expected_rows=len(
            observations
        ),
        expected_features=len(
            normalized_feature_names
        ),
    )


def build_feature_impact_dataframe(
    observations: pd.DataFrame,
    feature_names: Sequence[str],
    shap_values: np.ndarray,
    background_data: pd.DataFrame,
    *,
    entity_id_columns: Sequence[str] | None = None,
    date_column: str | None = None,
) -> pd.DataFrame:
    """Create a long-form feature-impact DataFrame.

    Args:
        observations:
            Original observations.
        feature_names:
            Ordered model feature names.
        shap_values:
            Two-dimensional array of local feature impacts.
        background_data:
            Reference population.
        entity_id_columns:
            Optional entity identifier columns.
        date_column:
            Optional date column.

    Returns:
        One row per observation and feature.
    """

    normalized_feature_names = validate_feature_names(
        observations=observations,
        feature_names=feature_names,
    )

    identifier_columns = validate_identifier_columns(
        observations=observations,
        entity_id_columns=entity_id_columns,
        date_column=date_column,
    )

    validate_background_data(
        background_data=background_data,
        feature_names=normalized_feature_names,
    )

    normalized_shap_values = normalize_shap_values(
        shap_output=shap_values,
        expected_rows=len(
            observations
        ),
        expected_features=len(
            normalized_feature_names
        ),
    )

    observation_frame = observations.copy()

    if not identifier_columns:
        observation_frame = observation_frame.assign(
            **{
                DEFAULT_ENTITY_ID_COLUMN: np.arange(
                    len(
                        observation_frame
                    )
                )
            }
        )

        identifier_columns = [
            DEFAULT_ENTITY_ID_COLUMN
        ]

    feature_value_frame = observation_frame.melt(
        id_vars=identifier_columns,
        value_vars=normalized_feature_names,
        var_name=DEFAULT_ATTRIBUTE_NAME_COLUMN,
        value_name=DEFAULT_ATTRIBUTE_VALUE_COLUMN,
    )

    shap_frame = pd.DataFrame(
        normalized_shap_values,
        columns=normalized_feature_names,
    )

    for identifier_column in identifier_columns:
        shap_frame[
            identifier_column
        ] = observation_frame[
            identifier_column
        ].values

    shap_long_frame = shap_frame.melt(
        id_vars=identifier_columns,
        value_vars=normalized_feature_names,
        var_name=DEFAULT_ATTRIBUTE_NAME_COLUMN,
        value_name=DEFAULT_ATTRIBUTE_IMPACT_COLUMN,
    )

    feature_impact_frame = feature_value_frame.merge(
        shap_long_frame,
        on=(
            identifier_columns
            + [
                DEFAULT_ATTRIBUTE_NAME_COLUMN
            ]
        ),
        how="inner",
        validate="one_to_one",
    )

    average_attribute_values = (
        background_data[
            normalized_feature_names
        ]
        .mean()
        .to_dict()
    )

    feature_impact_frame[
        "average_attribute_value"
    ] = feature_impact_frame[
        DEFAULT_ATTRIBUTE_NAME_COLUMN
    ].map(
        average_attribute_values
    )

    feature_impact_frame[
        "value_vs_typical"
    ] = np.where(
        feature_impact_frame[
            DEFAULT_ATTRIBUTE_VALUE_COLUMN
        ]
        > feature_impact_frame[
            "average_attribute_value"
        ],
        "more",
        np.where(
            feature_impact_frame[
                DEFAULT_ATTRIBUTE_VALUE_COLUMN
            ]
            < feature_impact_frame[
                "average_attribute_value"
            ],
            "less",
            "equal",
        ),
    )

    feature_impact_frame[
        "risk_vs_typical"
    ] = np.where(
        feature_impact_frame[
            DEFAULT_ATTRIBUTE_IMPACT_COLUMN
        ]
        > 0,
        "more",
        np.where(
            feature_impact_frame[
                DEFAULT_ATTRIBUTE_IMPACT_COLUMN
            ]
            < 0,
            "less",
            "neutral",
        ),
    )

    grouping_columns = identifier_columns

    feature_impact_frame[
        "ranked_impact_positive"
    ] = (
        feature_impact_frame.groupby(
            grouping_columns
        )[
            DEFAULT_ATTRIBUTE_IMPACT_COLUMN
        ]
        .rank(
            method="first",
            ascending=False,
        )
        .astype(
            int
        )
        - 1
    )

    feature_impact_frame[
        "ranked_impact_negative"
    ] = (
        feature_impact_frame.groupby(
            grouping_columns
        )[
            DEFAULT_ATTRIBUTE_IMPACT_COLUMN
        ]
        .rank(
            method="first",
            ascending=True,
        )
        .astype(
            int
        )
        - 1
    )

    feature_impact_frame[
        "absolute_impact"
    ] = feature_impact_frame[
        DEFAULT_ATTRIBUTE_IMPACT_COLUMN
    ].abs()

    feature_impact_frame[
        "ranked_impact_absolute"
    ] = (
        feature_impact_frame.groupby(
            grouping_columns
        )[
            "absolute_impact"
        ]
        .rank(
            method="first",
            ascending=False,
        )
        .astype(
            int
        )
        - 1
    )

    ordered_columns = (
        identifier_columns
        + [
            DEFAULT_ATTRIBUTE_NAME_COLUMN,
            DEFAULT_ATTRIBUTE_VALUE_COLUMN,
            "average_attribute_value",
            DEFAULT_ATTRIBUTE_IMPACT_COLUMN,
            "absolute_impact",
            "ranked_impact_positive",
            "ranked_impact_negative",
            "ranked_impact_absolute",
            "risk_vs_typical",
            "value_vs_typical",
        ]
    )

    return (
        feature_impact_frame[
            ordered_columns
        ]
        .sort_values(
            grouping_columns
            + [
                "ranked_impact_positive"
            ]
        )
        .reset_index(
            drop=True
        )
    )


def generate_tree_feature_impacts(
    model: Any,
    observations: pd.DataFrame,
    feature_names: Sequence[str],
    background_data: pd.DataFrame,
    *,
    entity_id_columns: Sequence[str] | None = None,
    date_column: str | None = None,
    check_additivity: bool = False,
    explainer: Any | None = None,
) -> pd.DataFrame:
    """Generate long-form TreeSHAP feature impacts.

    This is the main public entry point for tree-model explanations.

    Args:
        model:
            Tree-based model.
        observations:
            Observations to explain.
        feature_names:
            Ordered model feature names.
        background_data:
            SHAP background population.
        entity_id_columns:
            Optional entity identifier columns.
        date_column:
            Optional date column.
        check_additivity:
            Whether SHAP should perform its additivity check.
        explainer:
            Optional prebuilt or test explainer.

    Returns:
        Long-form feature-impact DataFrame.
    """

    shap_values = calculate_tree_shap_values(
        model=model,
        observations=observations,
        feature_names=feature_names,
        background_data=background_data,
        check_additivity=check_additivity,
        explainer=explainer,
    )

    return build_feature_impact_dataframe(
        observations=observations,
        feature_names=feature_names,
        shap_values=shap_values,
        background_data=background_data,
        entity_id_columns=entity_id_columns,
        date_column=date_column,
    )
