"""Tests for the generic TreeSHAP feature-impact engine."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from app.explainability.feature_impact import (
    build_feature_impact_dataframe,
    generate_tree_feature_impacts,
    normalize_shap_values,
    validate_background_data,
    validate_feature_names,
    validate_identifier_columns,
)


FEATURE_NAMES = [
    "feature_a",
    "feature_b",
    "feature_c",
]


@pytest.fixture
def observations() -> pd.DataFrame:
    """Return generic observations."""

    return pd.DataFrame(
        {
            "entity_id": [
                "entity-1",
                "entity-2",
            ],
            "observation_date": [
                "2026-07-30",
                "2026-07-31",
            ],
            "feature_a": [
                10.0,
                20.0,
            ],
            "feature_b": [
                4.0,
                8.0,
            ],
            "feature_c": [
                1.0,
                3.0,
            ],
        }
    )


@pytest.fixture
def background_data() -> pd.DataFrame:
    """Return a generic background population."""

    return pd.DataFrame(
        {
            "feature_a": [
                5.0,
                15.0,
                25.0,
            ],
            "feature_b": [
                2.0,
                6.0,
                10.0,
            ],
            "feature_c": [
                0.0,
                2.0,
                4.0,
            ],
        }
    )


@pytest.fixture
def shap_values() -> np.ndarray:
    """Return deterministic local feature impacts."""

    return np.array(
        [
            [
                0.40,
                -0.20,
                0.10,
            ],
            [
                -0.50,
                0.80,
                0.30,
            ],
        ]
    )


def test_validate_feature_names_accepts_valid_features(
    observations: pd.DataFrame,
) -> None:
    result = validate_feature_names(
        observations=observations,
        feature_names=FEATURE_NAMES,
    )

    assert result == FEATURE_NAMES


def test_validate_feature_names_rejects_empty_features(
    observations: pd.DataFrame,
) -> None:
    with pytest.raises(
        ValueError,
        match="feature_names cannot be empty",
    ):
        validate_feature_names(
            observations=observations,
            feature_names=[],
        )


def test_validate_feature_names_rejects_duplicate_features(
    observations: pd.DataFrame,
) -> None:
    with pytest.raises(
        ValueError,
        match="must not contain duplicates",
    ):
        validate_feature_names(
            observations=observations,
            feature_names=[
                "feature_a",
                "feature_a",
            ],
        )


def test_validate_feature_names_rejects_missing_features(
    observations: pd.DataFrame,
) -> None:
    with pytest.raises(
        ValueError,
        match="missing required features",
    ):
        validate_feature_names(
            observations=observations,
            feature_names=[
                "feature_a",
                "missing_feature",
            ],
        )


def test_validate_identifier_columns_accepts_unique_keys(
    observations: pd.DataFrame,
) -> None:
    result = validate_identifier_columns(
        observations=observations,
        entity_id_columns=[
            "entity_id",
        ],
        date_column="observation_date",
    )

    assert result == [
        "entity_id",
        "observation_date",
    ]


def test_validate_identifier_columns_rejects_duplicate_keys(
    observations: pd.DataFrame,
) -> None:
    duplicate_observations = pd.concat(
        [
            observations,
            observations.iloc[
                [
                    0
                ]
            ],
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="identifier combinations must be unique",
    ):
        validate_identifier_columns(
            observations=duplicate_observations,
            entity_id_columns=[
                "entity_id",
            ],
            date_column="observation_date",
        )


def test_validate_background_data_rejects_empty_data() -> None:
    empty_background = pd.DataFrame(
        columns=FEATURE_NAMES
    )

    with pytest.raises(
        ValueError,
        match="background_data cannot be empty",
    ):
        validate_background_data(
            background_data=empty_background,
            feature_names=FEATURE_NAMES,
        )


def test_normalize_shap_values_accepts_explanation_object() -> None:
    explanation = SimpleNamespace(
        values=np.array(
            [
                [
                    0.10,
                    0.20,
                    0.30,
                ],
                [
                    0.40,
                    0.50,
                    0.60,
                ],
            ]
        )
    )

    result = normalize_shap_values(
        shap_output=explanation,
        expected_rows=2,
        expected_features=3,
    )

    assert result.shape == (
        2,
        3,
    )


def test_normalize_shap_values_uses_last_class() -> None:
    explanation = SimpleNamespace(
        values=np.array(
            [
                [
                    [
                        0.01,
                        0.10,
                    ],
                    [
                        0.02,
                        0.20,
                    ],
                    [
                        0.03,
                        0.30,
                    ],
                ]
            ]
        )
    )

    result = normalize_shap_values(
        shap_output=explanation,
        expected_rows=1,
        expected_features=3,
    )

    assert result.tolist() == [
        [
            0.10,
            0.20,
            0.30,
        ]
    ]


def test_normalize_shap_values_rejects_wrong_shape() -> None:
    with pytest.raises(
        ValueError,
        match="Unexpected SHAP output shape",
    ):
        normalize_shap_values(
            shap_output=np.array(
                [
                    [
                        0.10,
                        0.20,
                    ]
                ]
            ),
            expected_rows=1,
            expected_features=3,
        )


def test_build_feature_impact_dataframe(
    observations: pd.DataFrame,
    background_data: pd.DataFrame,
    shap_values: np.ndarray,
) -> None:
    result = build_feature_impact_dataframe(
        observations=observations,
        feature_names=FEATURE_NAMES,
        shap_values=shap_values,
        background_data=background_data,
        entity_id_columns=[
            "entity_id",
        ],
        date_column="observation_date",
    )

    assert len(
        result
    ) == 6

    assert set(
        result[
            "attribute_name"
        ]
    ) == set(
        FEATURE_NAMES
    )

    entity_one = result[
        result[
            "entity_id"
        ]
        == "entity-1"
    ].sort_values(
        "ranked_impact_positive"
    )

    assert entity_one[
        "attribute_name"
    ].tolist() == [
        "feature_a",
        "feature_c",
        "feature_b",
    ]

    assert entity_one[
        "ranked_impact_positive"
    ].tolist() == [
        0,
        1,
        2,
    ]

    assert entity_one.iloc[
        0
    ][
        "risk_vs_typical"
    ] == "more"

    assert entity_one.iloc[
        2
    ][
        "risk_vs_typical"
    ] == "less"


def test_build_feature_impact_dataframe_adds_default_entity_id(
    background_data: pd.DataFrame,
) -> None:
    observations = pd.DataFrame(
        {
            "feature_a": [
                10.0,
            ],
            "feature_b": [
                4.0,
            ],
            "feature_c": [
                1.0,
            ],
        }
    )

    result = build_feature_impact_dataframe(
        observations=observations,
        feature_names=FEATURE_NAMES,
        shap_values=np.array(
            [
                [
                    0.20,
                    -0.10,
                    0.05,
                ]
            ]
        ),
        background_data=background_data,
    )

    assert "entity_id" in result.columns
    assert result[
        "entity_id"
    ].unique().tolist() == [
        0
    ]


def test_generate_tree_feature_impacts(
    observations: pd.DataFrame,
    background_data: pd.DataFrame,
    shap_values: np.ndarray,
) -> None:
    fake_explainer = lambda data, check_additivity: SimpleNamespace(
        values=shap_values
    )

    result = generate_tree_feature_impacts(
        model=object(),
        observations=observations,
        feature_names=FEATURE_NAMES,
        background_data=background_data,
        entity_id_columns=[
            "entity_id",
        ],
        date_column="observation_date",
        explainer=fake_explainer,
    )

    assert len(
        result
    ) == 6

    assert result[
        "attribute_impact"
    ].sum() == pytest.approx(
        0.90
    )
