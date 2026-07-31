"""Tests for the generic reason selection engine."""

import pandas as pd
import pytest

from app.explainability.feature_registry import (
    FeatureDefinition,
    FeatureRegistry,
)
from app.explainability.reason_selector import (
    infer_identifier_columns,
    select_reasons,
    validate_impact_dataframe,
    validate_reason_parameters,
)


@pytest.fixture
def impact_data() -> pd.DataFrame:
    """Return generic feature-impact data."""

    return pd.DataFrame(
        {
            "entity_id": [
                "entity-1",
                "entity-1",
                "entity-1",
                "entity-2",
                "entity-2",
                "entity-2",
            ],
            "attribute_name": [
                "feature_a",
                "feature_b",
                "feature_c",
                "feature_a",
                "feature_b",
                "feature_c",
            ],
            "attribute_value": [
                10.0,
                20.0,
                30.0,
                40.0,
                50.0,
                60.0,
            ],
            "average_attribute_value": [
                15.0,
                15.0,
                15.0,
                45.0,
                45.0,
                45.0,
            ],
            "attribute_impact": [
                0.80,
                0.20,
                -0.60,
                -0.90,
                0.40,
                0.10,
            ],
            "absolute_impact": [
                0.80,
                0.20,
                0.60,
                0.90,
                0.40,
                0.10,
            ],
            "ranked_impact_positive": [
                0,
                1,
                2,
                2,
                0,
                1,
            ],
            "ranked_impact_negative": [
                2,
                1,
                0,
                0,
                2,
                1,
            ],
            "ranked_impact_absolute": [
                0,
                2,
                1,
                0,
                1,
                2,
            ],
            "risk_vs_typical": [
                "more",
                "more",
                "less",
                "less",
                "more",
                "more",
            ],
            "value_vs_typical": [
                "less",
                "more",
                "more",
                "less",
                "more",
                "more",
            ],
        }
    )


@pytest.fixture
def feature_registry() -> FeatureRegistry:
    """Return generic feature metadata."""

    return FeatureRegistry(
        [
            FeatureDefinition(
                feature_name="feature_a",
                display_name="Feature A",
                reason_code="REASON_A",
                adverse_reason="Feature A increased modeled risk",
                favorable_reason="Feature A reduced modeled risk",
                metadata={
                    "business_priority": 2,
                },
            ),
            FeatureDefinition(
                feature_name="feature_b",
                display_name="Feature B",
                reason_code="REASON_B",
                adverse_reason="Feature B increased modeled risk",
                favorable_reason="Feature B reduced modeled risk",
                metadata={
                    "business_priority": 1,
                },
            ),
            FeatureDefinition(
                feature_name="feature_c",
                display_name="Feature C",
                reason_code="REASON_C",
                adverse_reason="Feature C increased modeled risk",
                favorable_reason="Feature C reduced modeled risk",
                compliance_review_required=True,
                metadata={
                    "business_priority": 3,
                },
            ),
        ]
    )


def test_validate_reason_parameters_accepts_valid_values() -> None:
    validate_reason_parameters(
        direction="adverse",
        ranking="impact",
        top_n=4,
        minimum_contribution=0.0,
    )


@pytest.mark.parametrize(
    "top_n",
    [
        0,
        -1,
    ],
)
def test_validate_reason_parameters_rejects_invalid_top_n(
    top_n: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="top_n must be at least 1",
    ):
        validate_reason_parameters(
            direction="adverse",
            ranking="impact",
            top_n=top_n,
            minimum_contribution=0.0,
        )


def test_validate_reason_parameters_accepts_all() -> None:
    validate_reason_parameters(
        direction="adverse",
        ranking="impact",
        top_n="all",
        minimum_contribution=0.0,
    )


def test_validate_reason_parameters_rejects_negative_minimum() -> None:
    with pytest.raises(
        ValueError,
        match="minimum_contribution must be non-negative",
    ):
        validate_reason_parameters(
            direction="adverse",
            ranking="impact",
            top_n=4,
            minimum_contribution=-0.01,
        )


def test_validate_impact_dataframe_rejects_missing_columns() -> None:
    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        validate_impact_dataframe(
            pd.DataFrame(
                {
                    "attribute_name": [
                        "feature_a"
                    ]
                }
            )
        )


def test_infer_identifier_columns(
    impact_data: pd.DataFrame,
) -> None:
    result = infer_identifier_columns(
        impact_data
    )

    assert result == [
        "entity_id"
    ]


def test_select_adverse_top_n(
    impact_data: pd.DataFrame,
) -> None:
    result = select_reasons(
        impact_data,
        direction="adverse",
        ranking="impact",
        top_n=1,
    )

    entity_one = result[
        result[
            "entity_id"
        ]
        == "entity-1"
    ]

    entity_two = result[
        result[
            "entity_id"
        ]
        == "entity-2"
    ]

    assert entity_one[
        "attribute_name"
    ].tolist() == [
        "feature_a"
    ]

    assert entity_two[
        "attribute_name"
    ].tolist() == [
        "feature_b"
    ]

    assert entity_one[
        "reason_rank"
    ].tolist() == [
        1
    ]


def test_select_adverse_all(
    impact_data: pd.DataFrame,
) -> None:
    result = select_reasons(
        impact_data,
        direction="adverse",
        ranking="impact",
        top_n="all",
    )

    assert len(
        result
    ) == 4

    entity_one = result[
        result[
            "entity_id"
        ]
        == "entity-1"
    ]

    assert entity_one[
        "attribute_name"
    ].tolist() == [
        "feature_a",
        "feature_b",
    ]


def test_select_favorable_reasons(
    impact_data: pd.DataFrame,
) -> None:
    result = select_reasons(
        impact_data,
        direction="favorable",
        ranking="impact",
        top_n="all",
    )

    assert result[
        "attribute_name"
    ].tolist() == [
        "feature_c",
        "feature_a",
    ]


def test_select_all_uses_absolute_ranking(
    impact_data: pd.DataFrame,
) -> None:
    result = select_reasons(
        impact_data,
        direction="all",
        ranking="impact",
        top_n=2,
    )

    entity_one = result[
        result[
            "entity_id"
        ]
        == "entity-1"
    ]

    assert entity_one[
        "attribute_name"
    ].tolist() == [
        "feature_a",
        "feature_c",
    ]


def test_select_reasons_applies_minimum_contribution(
    impact_data: pd.DataFrame,
) -> None:
    result = select_reasons(
        impact_data,
        direction="adverse",
        ranking="impact",
        top_n="all",
        minimum_contribution=0.30,
    )

    assert result[
        "attribute_name"
    ].tolist() == [
        "feature_a",
        "feature_b",
    ]

    assert result[
        "attribute_impact"
    ].tolist() == [
        0.80,
        0.40,
    ]


def test_select_reasons_enriches_registry_metadata(
    impact_data: pd.DataFrame,
    feature_registry: FeatureRegistry,
) -> None:
    result = select_reasons(
        impact_data,
        direction="adverse",
        ranking="impact",
        top_n=1,
        registry=feature_registry,
    )

    entity_one = result[
        result[
            "entity_id"
        ]
        == "entity-1"
    ].iloc[0]

    assert (
        entity_one[
            "display_name"
        ]
        == "Feature A"
    )

    assert (
        entity_one[
            "reason_code"
        ]
        == "REASON_A"
    )

    assert (
        entity_one[
            "adverse_reason"
        ]
        == "Feature A increased modeled risk"
    )


def test_select_reasons_flags_compliance_review(
    impact_data: pd.DataFrame,
    feature_registry: FeatureRegistry,
) -> None:
    result = select_reasons(
        impact_data,
        direction="all",
        ranking="absolute_impact",
        top_n="all",
        registry=feature_registry,
    )

    flagged = result[
        result[
            "attribute_name"
        ]
        == "feature_c"
    ]

    assert flagged[
        "compliance_review_required"
    ].all()


def test_business_priority_ranking(
    impact_data: pd.DataFrame,
    feature_registry: FeatureRegistry,
) -> None:
    result = select_reasons(
        impact_data,
        direction="adverse",
        ranking="business_priority",
        top_n="all",
        registry=feature_registry,
    )

    entity_one = result[
        result[
            "entity_id"
        ]
        == "entity-1"
    ]

    assert entity_one[
        "attribute_name"
    ].tolist() == [
        "feature_b",
        "feature_a",
    ]


def test_unregistered_features_use_defaults(
    impact_data: pd.DataFrame,
) -> None:
    result = select_reasons(
        impact_data,
        direction="adverse",
        ranking="impact",
        top_n=1,
        registry=FeatureRegistry(),
    )

    first = result.iloc[0]

    assert (
        first[
            "display_name"
        ]
        == first[
            "attribute_name"
        ]
    )

    assert first[
        "reason_code"
    ] is None

    assert bool(
    first[
        "compliance_review_required"
    ]
    ) is False


def test_select_reasons_rejects_missing_identifier(
    impact_data: pd.DataFrame,
) -> None:
    with pytest.raises(
        ValueError,
        match="missing identifier columns",
    ):
        select_reasons(
            impact_data,
            identifier_columns=[
                "missing_id"
            ],
        )

def test_select_adverse_reasons_excludes_zero_impacts(
    impact_data: pd.DataFrame,
) -> None:
    zero_impact_row = impact_data.iloc[
        [
            0
        ]
    ].copy()

    zero_impact_row[
        "attribute_name"
    ] = "zero_feature"

    zero_impact_row[
        "attribute_impact"
    ] = 0.0

    zero_impact_row[
        "absolute_impact"
    ] = 0.0

    zero_impact_row[
        "ranked_impact_positive"
    ] = 3

    zero_impact_row[
        "ranked_impact_negative"
    ] = 1

    zero_impact_row[
        "ranked_impact_absolute"
    ] = 3

    zero_impact_row[
        "risk_vs_typical"
    ] = "neutral"

    data_with_zero = pd.concat(
        [
            impact_data,
            zero_impact_row,
        ],
        ignore_index=True,
    )

    result = select_reasons(
        data_with_zero,
        direction="adverse",
        ranking="impact",
        top_n="all",
        minimum_contribution=0.0,
    )

    assert "zero_feature" not in result[
        "attribute_name"
    ].tolist()