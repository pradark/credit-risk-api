"""Tests for single-row explanation persistence."""

import datetime

import pandas as pd
import pytest

from app.explainability import (
    explanation_writer,
)


@pytest.fixture
def adverse_reasons() -> pd.DataFrame:
    """Return selected adverse reasons."""

    return pd.DataFrame(
        {
            "reason_rank": [
                1,
                2,
            ],
            "attribute_name": [
                "feature_a",
                "feature_b",
            ],
            "display_name": [
                "Feature A",
                "Feature B",
            ],
            "attribute_value": [
                10.0,
                20.0,
            ],
            "average_attribute_value": [
                5.0,
                15.0,
            ],
            "attribute_impact": [
                0.80,
                0.40,
            ],
            "absolute_impact": [
                0.80,
                0.40,
            ],
            "risk_vs_typical": [
                "more",
                "more",
            ],
            "value_vs_typical": [
                "more",
                "more",
            ],
            "reason_code": [
                "REASON_A",
                "REASON_B",
            ],
            "adverse_reason": [
                "Feature A increased modeled risk",
                "Feature B increased modeled risk",
            ],
            "compliance_review_required": [
                False,
                True,
            ],
        }
    )


@pytest.fixture
def all_feature_contributions() -> pd.DataFrame:
    """Return complete model feature contributions."""

    return pd.DataFrame(
        {
            "attribute_name": [
                "feature_a",
                "feature_b",
                "feature_c",
            ],
            "attribute_value": [
                10.0,
                20.0,
                30.0,
            ],
            "average_attribute_value": [
                5.0,
                15.0,
                35.0,
            ],
            "attribute_impact": [
                0.80,
                0.40,
                -0.20,
            ],
            "absolute_impact": [
                0.80,
                0.40,
                0.20,
            ],
            "ranked_impact_positive": [
                0,
                1,
                2,
            ],
            "ranked_impact_negative": [
                2,
                1,
                0,
            ],
            "ranked_impact_absolute": [
                0,
                1,
                2,
            ],
            "risk_vs_typical": [
                "more",
                "more",
                "less",
            ],
            "value_vs_typical": [
                "more",
                "more",
                "less",
            ],
        }
    )


def test_build_explanation_record_creates_one_row(
    adverse_reasons: pd.DataFrame,
    all_feature_contributions: pd.DataFrame,
) -> None:
    timestamp = datetime.datetime(
        2026,
        7,
        31,
        12,
        0,
        tzinfo=datetime.timezone.utc,
    )

    result = (
        explanation_writer
        .build_explanation_record(
            application_id="application-1",
            default_probability=0.81,
            decision_threshold=0.75,
            adverse_reasons=adverse_reasons,
            all_feature_contributions=(
                all_feature_contributions
            ),
            requested_top_n=2,
            prediction_id="prediction-1",
            prediction_timestamp=timestamp,
            model_version="model-v1",
        )
    )

    assert len(
        result
    ) == 1

    assert result.loc[
        0,
        "application_id",
    ] == "application-1"

    assert result.loc[
        0,
        "default_probability",
    ] == pytest.approx(
        0.81
    )

    assert result.loc[
        0,
        "decision",
    ] == "decline"

    assert result.loc[
        0,
        "requested_top_n",
    ] == "2"

    assert result.loc[
        0,
        "total_adverse_factors",
    ] == 2


def test_explanation_record_contains_nested_reasons(
    adverse_reasons: pd.DataFrame,
    all_feature_contributions: pd.DataFrame,
) -> None:
    result = (
        explanation_writer
        .build_explanation_record(
            application_id="application-1",
            default_probability=0.81,
            decision_threshold=0.75,
            adverse_reasons=adverse_reasons,
            all_feature_contributions=(
                all_feature_contributions
            ),
        )
    )

    reasons = result.loc[
        0,
        "adverse_action_reasons",
    ]

    assert len(
        reasons
    ) == 2

    assert reasons[
        0
    ][
        "reason_rank"
    ] == 1

    assert reasons[
        0
    ][
        "reason_code"
    ] == "REASON_A"

    assert reasons[
        1
    ][
        "attribute_name"
    ] == "feature_b"


def test_explanation_record_contains_all_contributions(
    adverse_reasons: pd.DataFrame,
    all_feature_contributions: pd.DataFrame,
) -> None:
    result = (
        explanation_writer
        .build_explanation_record(
            application_id="application-1",
            default_probability=0.81,
            decision_threshold=0.75,
            adverse_reasons=adverse_reasons,
            all_feature_contributions=(
                all_feature_contributions
            ),
            requested_top_n="all",
        )
    )

    contributions = result.loc[
        0,
        "all_feature_contributions",
    ]

    assert len(
        contributions
    ) == 3

    assert contributions[
        2
    ][
        "attribute_name"
    ] == "feature_c"


def test_explanation_record_builds_compliance_flags(
    adverse_reasons: pd.DataFrame,
    all_feature_contributions: pd.DataFrame,
) -> None:
    result = (
        explanation_writer
        .build_explanation_record(
            application_id="application-1",
            default_probability=0.81,
            decision_threshold=0.75,
            adverse_reasons=adverse_reasons,
            all_feature_contributions=(
                all_feature_contributions
            ),
        )
    )

    assert bool(
        result.loc[
            0,
            "compliance_review_required",
        ]
    ) is True

    flags = result.loc[
        0,
        "compliance_flags",
    ]

    assert len(
        flags
    ) == 1

    assert flags[
        0
    ][
        "attribute_name"
    ] == "feature_b"


def test_explanation_record_approves_below_threshold(
    adverse_reasons: pd.DataFrame,
    all_feature_contributions: pd.DataFrame,
) -> None:
    result = (
        explanation_writer
        .build_explanation_record(
            application_id="application-1",
            default_probability=0.20,
            decision_threshold=0.75,
            adverse_reasons=adverse_reasons.iloc[
                0:0
            ].copy(),
            all_feature_contributions=(
                all_feature_contributions
            ),
        )
    )

    assert result.loc[
        0,
        "decision",
    ] == "approve"


@pytest.mark.parametrize(
    "probability",
    [
        -0.01,
        1.01,
    ],
)
def test_rejects_invalid_probability(
    probability: float,
    adverse_reasons: pd.DataFrame,
    all_feature_contributions: pd.DataFrame,
) -> None:
    with pytest.raises(
        ValueError,
        match=(
            "default_probability must be "
            "between 0 and 1"
        ),
    ):
        (
            explanation_writer
            .build_explanation_record(
                application_id="application-1",
                default_probability=probability,
                decision_threshold=0.75,
                adverse_reasons=adverse_reasons,
                all_feature_contributions=(
                    all_feature_contributions
                ),
            )
        )


def test_build_explanation_s3_key() -> None:
    timestamp = datetime.datetime(
        2026,
        7,
        31,
        12,
        0,
        tzinfo=datetime.timezone.utc,
    )

    result = (
        explanation_writer
        .build_explanation_s3_key(
            application_id="application-1",
            prediction_id="prediction-1",
            prediction_timestamp=timestamp,
        )
    )

    assert result == (
        "explanations/"
        "dt=2026-07-31/"
        "application_id=application-1/"
        "explanation_prediction-1.parquet"
    )


def test_write_explanation_to_s3(
    monkeypatch,
    adverse_reasons: pd.DataFrame,
    all_feature_contributions: pd.DataFrame,
) -> None:
    captured: dict[
        str,
        object
    ] = {}

    def fake_write(
        dataframe: pd.DataFrame,
        bucket: str,
        key: str,
    ) -> None:
        captured[
            "dataframe"
        ] = dataframe

        captured[
            "bucket"
        ] = bucket

        captured[
            "key"
        ] = key

    monkeypatch.setattr(
        explanation_writer,
        "write_parquet_to_s3",
        fake_write,
    )

    timestamp = datetime.datetime(
        2026,
        7,
        31,
        12,
        0,
        tzinfo=datetime.timezone.utc,
    )

    result, key = (
        explanation_writer
        .write_explanation_to_s3(
            application_id="application-1",
            default_probability=0.81,
            decision_threshold=0.75,
            adverse_reasons=adverse_reasons,
            all_feature_contributions=(
                all_feature_contributions
            ),
            requested_top_n=2,
            prediction_id="prediction-1",
            prediction_timestamp=timestamp,
            bucket="test-bucket",
        )
    )

    assert len(
        result
    ) == 1

    assert captured[
        "bucket"
    ] == "test-bucket"

    assert captured[
        "key"
    ] == key

    assert key == (
        "explanations/"
        "dt=2026-07-31/"
        "application_id=application-1/"
        "explanation_prediction-1.parquet"
    )

def test_explanation_record_uses_none_for_empty_metadata(
    adverse_reasons: pd.DataFrame,
    all_feature_contributions: pd.DataFrame,
) -> None:
    result = explanation_writer.build_explanation_record(
        application_id="application-1",
        default_probability=0.81,
        decision_threshold=0.75,
        adverse_reasons=adverse_reasons,
        all_feature_contributions=(
            all_feature_contributions
        ),
        additional_metadata={},
    )

    assert result.loc[
        0,
        "additional_metadata",
    ] is None