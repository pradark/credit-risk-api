"""Unit tests for the model performance monitoring workflow."""

import pandas as pd
import pytest

from app.monitoring.performance_monitor import (
    add_monitoring_metadata,
    build_performance_s3_keys,
    join_predictions_and_outcomes,
    validate_minimum_sample_size,
    validate_outcome_data,
    validate_prediction_data,
)


def create_valid_predictions() -> pd.DataFrame:
    """Create valid prediction data for tests."""
    return pd.DataFrame(
        {
            "application_id": [
                "app-001",
                "app-002",
                "app-003",
                "app-004",
            ],
            "predicted_probability": [
                0.10,
                0.30,
                0.70,
                0.90,
            ],
        }
    )


def create_valid_outcomes() -> pd.DataFrame:
    """Create valid outcome data for tests."""
    return pd.DataFrame(
        {
            "application_id": [
                "app-001",
                "app-002",
                "app-003",
                "app-004",
            ],
            "actual_default": [
                0,
                0,
                1,
                1,
            ],
        }
    )


def test_build_performance_s3_keys() -> None:
    """S3 keys should use the requested monitoring date."""
    result = build_performance_s3_keys(
        "2026-07-31"
    )

    assert result == {
        "prediction_key": (
            "predictions/dt=2026-07-31/"
            "predictions.parquet"
        ),
        "outcome_key": (
            "outcomes/dt=2026-07-31/"
            "outcomes.parquet"
        ),
        "output_key": (
            "monitoring/performance/"
            "model_performance_2026-07-31.parquet"
        ),
    }


def test_validate_prediction_data_accepts_valid_data() -> None:
    """Valid prediction data should not raise an exception."""
    predictions = create_valid_predictions()

    validate_prediction_data(
        predictions
    )


def test_validate_prediction_data_rejects_missing_columns() -> None:
    """Prediction data must contain required columns."""
    predictions = pd.DataFrame(
        {
            "application_id": [
                "app-001",
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        validate_prediction_data(
            predictions
        )


def test_validate_prediction_data_rejects_empty_data() -> None:
    """Prediction data cannot be empty."""
    predictions = pd.DataFrame(
        columns=[
            "application_id",
            "predicted_probability",
        ]
    )

    with pytest.raises(
        ValueError,
        match="Prediction data is empty",
    ):
        validate_prediction_data(
            predictions
        )


def test_validate_prediction_data_rejects_missing_ids() -> None:
    """Prediction application identifiers cannot be missing."""
    predictions = create_valid_predictions()
    predictions.loc[
        0,
        "application_id",
    ] = None

    with pytest.raises(
        ValueError,
        match="missing application_id",
    ):
        validate_prediction_data(
            predictions
        )


def test_validate_prediction_data_rejects_duplicate_ids() -> None:
    """Prediction application identifiers must be unique."""
    predictions = create_valid_predictions()
    predictions.loc[
        1,
        "application_id",
    ] = predictions.loc[
        0,
        "application_id",
    ]

    with pytest.raises(
        ValueError,
        match="duplicate application_id",
    ):
        validate_prediction_data(
            predictions
        )


def test_validate_prediction_data_rejects_missing_probabilities() -> None:
    """Predicted probabilities cannot be missing."""
    predictions = create_valid_predictions()
    predictions.loc[
        0,
        "predicted_probability",
    ] = None

    with pytest.raises(
        ValueError,
        match="missing predicted_probability",
    ):
        validate_prediction_data(
            predictions
        )


@pytest.mark.parametrize(
    "invalid_probability",
    [
        -0.01,
        1.01,
    ],
)
def test_validate_prediction_data_rejects_invalid_probabilities(
    invalid_probability: float,
) -> None:
    """Predicted probabilities must remain between zero and one."""
    predictions = create_valid_predictions()
    predictions.loc[
        0,
        "predicted_probability",
    ] = invalid_probability

    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        validate_prediction_data(
            predictions
        )


def test_validate_outcome_data_accepts_valid_data() -> None:
    """Valid outcome data should not raise an exception."""
    outcomes = create_valid_outcomes()

    validate_outcome_data(
        outcomes
    )


def test_validate_outcome_data_rejects_missing_columns() -> None:
    """Outcome data must contain required columns."""
    outcomes = pd.DataFrame(
        {
            "application_id": [
                "app-001",
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        validate_outcome_data(
            outcomes
        )


def test_validate_outcome_data_rejects_empty_data() -> None:
    """Outcome data cannot be empty."""
    outcomes = pd.DataFrame(
        columns=[
            "application_id",
            "actual_default",
        ]
    )

    with pytest.raises(
        ValueError,
        match="Outcome data is empty",
    ):
        validate_outcome_data(
            outcomes
        )


def test_validate_outcome_data_rejects_missing_ids() -> None:
    """Outcome application identifiers cannot be missing."""
    outcomes = create_valid_outcomes()
    outcomes.loc[
        0,
        "application_id",
    ] = None

    with pytest.raises(
        ValueError,
        match="missing application_id",
    ):
        validate_outcome_data(
            outcomes
        )


def test_validate_outcome_data_rejects_duplicate_ids() -> None:
    """Outcome application identifiers must be unique."""
    outcomes = create_valid_outcomes()
    outcomes.loc[
        1,
        "application_id",
    ] = outcomes.loc[
        0,
        "application_id",
    ]

    with pytest.raises(
        ValueError,
        match="duplicate application_id",
    ):
        validate_outcome_data(
            outcomes
        )


def test_validate_outcome_data_rejects_missing_outcomes() -> None:
    """Observed outcomes cannot be missing."""
    outcomes = create_valid_outcomes()
    outcomes.loc[
        0,
        "actual_default",
    ] = None

    with pytest.raises(
        ValueError,
        match="missing actual_default",
    ):
        validate_outcome_data(
            outcomes
        )


def test_validate_outcome_data_rejects_non_binary_values() -> None:
    """Observed outcomes must contain only zero and one."""
    outcomes = create_valid_outcomes()
    outcomes.loc[
        0,
        "actual_default",
    ] = 2

    with pytest.raises(
        ValueError,
        match="only 0 and 1",
    ):
        validate_outcome_data(
            outcomes
        )


def test_join_predictions_and_outcomes() -> None:
    """Predictions and outcomes should join on application_id."""
    predictions = create_valid_predictions()
    outcomes = create_valid_outcomes()

    result = join_predictions_and_outcomes(
        predictions=predictions,
        outcomes=outcomes,
    )

    assert list(
        result.columns
    ) == [
        "application_id",
        "predicted_probability",
        "actual_default",
    ]

    assert len(result) == 4

    assert result[
        "application_id"
    ].tolist() == [
        "app-001",
        "app-002",
        "app-003",
        "app-004",
    ]


def test_join_predictions_and_outcomes_uses_inner_join() -> None:
    """Only applications present in both datasets should remain."""
    predictions = create_valid_predictions()

    outcomes = pd.DataFrame(
        {
            "application_id": [
                "app-001",
                "app-003",
            ],
            "actual_default": [
                0,
                1,
            ],
        }
    )

    result = join_predictions_and_outcomes(
        predictions=predictions,
        outcomes=outcomes,
    )

    assert len(result) == 2

    assert result[
        "application_id"
    ].tolist() == [
        "app-001",
        "app-003",
    ]


def test_join_predictions_and_outcomes_rejects_no_matches() -> None:
    """The join should fail when no application identifiers match."""
    predictions = create_valid_predictions()

    outcomes = pd.DataFrame(
        {
            "application_id": [
                "other-001",
                "other-002",
            ],
            "actual_default": [
                0,
                1,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="No matching application_id",
    ):
        join_predictions_and_outcomes(
            predictions=predictions,
            outcomes=outcomes,
        )


def test_validate_minimum_sample_size_accepts_enough_data() -> None:
    """A dataset meeting the minimum size should pass validation."""
    performance_data = pd.DataFrame(
        {
            "application_id": [
                "app-001",
                "app-002",
                "app-003",
            ],
        }
    )

    validate_minimum_sample_size(
        performance_data=performance_data,
        minimum_samples=3,
    )


def test_validate_minimum_sample_size_rejects_too_few_rows() -> None:
    """A dataset below the minimum size should fail validation."""
    performance_data = pd.DataFrame(
        {
            "application_id": [
                "app-001",
                "app-002",
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="Not enough matched samples",
    ):
        validate_minimum_sample_size(
            performance_data=performance_data,
            minimum_samples=3,
        )


@pytest.mark.parametrize(
    "minimum_samples",
    [
        0,
        -1,
    ],
)
def test_validate_minimum_sample_size_rejects_invalid_minimum(
    minimum_samples: int,
) -> None:
    """The configured minimum sample size must be positive."""
    performance_data = pd.DataFrame(
        {
            "application_id": [
                "app-001",
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="must be at least 1",
    ):
        validate_minimum_sample_size(
            performance_data=performance_data,
            minimum_samples=minimum_samples,
        )


def test_add_monitoring_metadata() -> None:
    """Monitoring metadata should be added without changing metrics."""
    metrics = pd.DataFrame(
        {
            "auc": [
                0.85,
            ],
            "ks": [
                0.60,
            ],
        }
    )

    result = add_monitoring_metadata(
        metrics=metrics,
        run_date="2026-07-31",
        prediction_key=(
            "predictions/dt=2026-07-31/"
            "predictions.parquet"
        ),
        outcome_key=(
            "outcomes/dt=2026-07-31/"
            "outcomes.parquet"
        ),
        matched_sample_size=250,
    )

    assert result.loc[
        0,
        "auc",
    ] == 0.85

    assert result.loc[
        0,
        "ks",
    ] == 0.60

    assert result.loc[
        0,
        "run_date",
    ] == "2026-07-31"

    assert result.loc[
        0,
        "prediction_dataset",
    ] == (
        "predictions/dt=2026-07-31/"
        "predictions.parquet"
    )

    assert result.loc[
        0,
        "outcome_dataset",
    ] == (
        "outcomes/dt=2026-07-31/"
        "outcomes.parquet"
    )

    assert result.loc[
        0,
        "matched_sample_size",
    ] == 250

    assert result.loc[
        0,
        "model_version",
    ] == "credit-risk-model-v1"

    assert result.loc[
        0,
        "monitoring_type",
    ] == "model_performance"
