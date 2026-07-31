"""Tests for the end-to-end explanation service."""

import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from app.explainability import explanation_service
from app.explainability.feature_registry import (
    FeatureDefinition,
    FeatureRegistry,
)


FEATURE_NAMES = [
    "feature_a",
    "feature_b",
    "feature_c",
]


VALID_FEATURES = {
    "feature_a": 10.0,
    "feature_b": 20.0,
    "feature_c": 30.0,
}


@pytest.fixture
def background_data() -> pd.DataFrame:
    """Return a generic background population."""

    return pd.DataFrame(
        {
            "feature_a": [
                5.0,
                10.0,
                15.0,
            ],
            "feature_b": [
                10.0,
                20.0,
                30.0,
            ],
            "feature_c": [
                25.0,
                30.0,
                35.0,
            ],
        }
    )


@pytest.fixture
def registry() -> FeatureRegistry:
    """Return generic model feature metadata."""

    return FeatureRegistry(
        [
            FeatureDefinition(
                feature_name="feature_a",
                display_name="Feature A",
                reason_code="REASON_A",
                adverse_reason=(
                    "Feature A increased modeled risk"
                ),
            ),
            FeatureDefinition(
                feature_name="feature_b",
                display_name="Feature B",
                reason_code="REASON_B",
                adverse_reason=(
                    "Feature B increased modeled risk"
                ),
                compliance_review_required=True,
            ),
            FeatureDefinition(
                feature_name="feature_c",
                display_name="Feature C",
                reason_code="REASON_C",
                adverse_reason=(
                    "Feature C increased modeled risk"
                ),
            ),
        ]
    )


@pytest.fixture
def fake_explainer():
    """Return deterministic SHAP contributions."""

    shap_values = np.array(
        [
            [
                0.80,
                0.40,
                -0.20,
            ]
        ]
    )

    return lambda data, check_additivity: SimpleNamespace(
        values=shap_values
    )


def test_validate_feature_mapping_preserves_order() -> None:
    result = explanation_service.validate_feature_mapping(
        features={
            "feature_c": 30.0,
            "feature_a": 10.0,
            "feature_b": 20.0,
        },
        feature_names=FEATURE_NAMES,
    )

    assert list(result) == FEATURE_NAMES


def test_validate_feature_mapping_rejects_missing_feature() -> None:
    with pytest.raises(
        ValueError,
        match="Missing required features",
    ):
        explanation_service.validate_feature_mapping(
            features={
                "feature_a": 10.0,
            },
            feature_names=FEATURE_NAMES,
        )


def test_create_observation_dataframe() -> None:
    result = explanation_service.create_observation_dataframe(
        application_id="application-1",
        features=VALID_FEATURES,
        feature_names=FEATURE_NAMES,
    )

    assert len(result) == 1

    assert result.loc[
        0,
        "application_id",
    ] == "application-1"

    assert (
        result[
            FEATURE_NAMES
        ]
        .iloc[0]
        .to_dict()
        == VALID_FEATURES
    )


@pytest.mark.parametrize(
    (
        "probability",
        "threshold",
        "expected",
    ),
    [
        (
            0.80,
            0.75,
            "decline",
        ),
        (
            0.75,
            0.75,
            "decline",
        ),
        (
            0.20,
            0.75,
            "approve",
        ),
    ],
)
def test_determine_decision(
    probability: float,
    threshold: float,
    expected: str,
) -> None:
    result = explanation_service.determine_decision(
        default_probability=probability,
        decision_threshold=threshold,
    )

    assert result == expected


def test_explain_prediction_returns_ranked_reasons(
    background_data: pd.DataFrame,
    registry: FeatureRegistry,
    fake_explainer,
) -> None:
    timestamp = datetime.datetime(
        2026,
        7,
        31,
        12,
        0,
        tzinfo=datetime.timezone.utc,
    )

    result = explanation_service.explain_prediction(
        application_id="application-1",
        features=VALID_FEATURES,
        model=object(),
        predictor=lambda features: 0.81,
        feature_names=FEATURE_NAMES,
        background_data=background_data,
        decision_threshold=0.75,
        registry=registry,
        top_n=2,
        prediction_id="prediction-1",
        prediction_timestamp=timestamp,
        explainer=fake_explainer,
    )

    assert result[
        "prediction_id"
    ] == "prediction-1"

    assert result[
        "default_probability"
    ] == pytest.approx(
        0.81
    )

    assert result[
        "decision"
    ] == "decline"

    assert result[
        "total_selected_reasons"
    ] == 2

    assert [
        reason[
            "attribute_name"
        ]
        for reason in result[
            "reasons"
        ]
    ] == [
        "feature_a",
        "feature_b",
    ]

    assert result[
        "s3_key"
    ] is None


def test_explain_prediction_supports_all_reasons(
    background_data: pd.DataFrame,
    fake_explainer,
) -> None:
    result = explanation_service.explain_prediction(
        application_id="application-1",
        features=VALID_FEATURES,
        model=object(),
        predictor=lambda features: 0.81,
        feature_names=FEATURE_NAMES,
        background_data=background_data,
        decision_threshold=0.75,
        top_n="all",
        explainer=fake_explainer,
    )

    assert result[
        "requested_top_n"
    ] == "all"

    assert result[
        "total_selected_reasons"
    ] == 2


def test_explain_prediction_can_persist_to_s3(
    monkeypatch,
    background_data: pd.DataFrame,
    registry: FeatureRegistry,
    fake_explainer,
) -> None:
    captured: dict[
        str,
        object,
    ] = {}

    original_builder = (
        explanation_service
        .build_explanation_record
    )

    def fake_write(
        **kwargs,
    ):
        captured.update(
            kwargs
        )

        record = original_builder(
            **{
                key: value
                for key, value
                in kwargs.items()
                if key not in {
                    "bucket",
                    "prefix",
                }
            }
        )

        return (
            record,
            (
                "explanations/"
                "dt=2026-07-31/"
                "application_id=application-1/"
                "explanation_prediction-1.parquet"
            ),
        )

    monkeypatch.setattr(
        explanation_service,
        "write_explanation_to_s3",
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

    result = explanation_service.explain_prediction(
        application_id="application-1",
        features=VALID_FEATURES,
        model=object(),
        predictor=lambda features: 0.81,
        feature_names=FEATURE_NAMES,
        background_data=background_data,
        decision_threshold=0.75,
        registry=registry,
        top_n=2,
        prediction_id="prediction-1",
        prediction_timestamp=timestamp,
        persistence="s3",
        s3_bucket="test-bucket",
        explainer=fake_explainer,
    )

    assert captured[
        "bucket"
    ] == "test-bucket"

    assert result[
        "s3_key"
    ] == (
        "explanations/"
        "dt=2026-07-31/"
        "application_id=application-1/"
        "explanation_prediction-1.parquet"
    )


def test_explain_prediction_flags_compliance_review(
    background_data: pd.DataFrame,
    registry: FeatureRegistry,
    fake_explainer,
) -> None:
    result = explanation_service.explain_prediction(
        application_id="application-1",
        features=VALID_FEATURES,
        model=object(),
        predictor=lambda features: 0.81,
        feature_names=FEATURE_NAMES,
        background_data=background_data,
        decision_threshold=0.75,
        registry=registry,
        top_n=2,
        explainer=fake_explainer,
    )

    assert result[
        "compliance_review_required"
    ] is True

    assert len(
        result[
            "compliance_flags"
        ]
    ) == 1


def test_explain_prediction_rejects_invalid_persistence(
    background_data: pd.DataFrame,
    fake_explainer,
) -> None:
    with pytest.raises(
        ValueError,
        match=(
            "persistence must be "
            "'none' or 's3'"
        ),
    ):
        explanation_service.explain_prediction(
            application_id="application-1",
            features=VALID_FEATURES,
            model=object(),
            predictor=lambda features: 0.81,
            feature_names=FEATURE_NAMES,
            background_data=background_data,
            decision_threshold=0.75,
            persistence="database",
            explainer=fake_explainer,
        )
