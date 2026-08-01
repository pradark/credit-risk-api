"""Tests for the AWS monitoring pipeline."""

from datetime import date
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from app.monitoring.pipeline import (
    join_predictions_and_outcomes,
    run_monitoring_pipeline,
    validate_pipeline_data,
)

# Shorthand used in all run_monitoring_pipeline calls
_SMALL_SAMPLE_KWARGS = {
    "minimum_samples": 1,
}


@pytest.fixture
def predictions() -> pd.DataFrame:
    """Return sample prediction data."""

    return pd.DataFrame(
        {
            "application_id": [
                f"APP-{i:04d}"
                for i in range(1, 11)
            ],
            "predicted_probability": [
                0.05,
                0.10,
                0.15,
                0.20,
                0.30,
                0.45,
                0.55,
                0.65,
                0.80,
                0.90,
            ],
        }
    )


@pytest.fixture
def outcomes() -> pd.DataFrame:
    """Return sample outcome data."""

    return pd.DataFrame(
        {
            "application_id": [
                f"APP-{i:04d}"
                for i in range(1, 11)
            ],
            "actual_default": [
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                1,
                1,
                1,
            ],
        }
    )


@pytest.fixture
def reference_data() -> pd.DataFrame:
    """Return small reference data for PSI."""

    import numpy as np

    rng = np.random.default_rng(
        42
    )

    return pd.DataFrame(
        {
            "income": rng.normal(
                50000,
                10000,
                50,
            ),
            "fico": rng.normal(
                700,
                50,
                50,
            ),
        }
    )


# ---------------------------------------------------------------------------
# validate_pipeline_data
# ---------------------------------------------------------------------------

def test_validate_pipeline_data_accepts_valid_input(
    predictions: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> None:
    """Valid predictions and outcomes should not raise."""

    validate_pipeline_data(
        predictions=predictions,
        outcomes=outcomes,
    )


def test_validate_pipeline_data_rejects_empty_predictions(
    outcomes: pd.DataFrame,
) -> None:
    """Empty predictions should raise ValueError."""

    with pytest.raises(
        ValueError,
        match="empty",
    ):
        validate_pipeline_data(
            predictions=pd.DataFrame(),
            outcomes=outcomes,
        )


def test_validate_pipeline_data_rejects_missing_columns(
    outcomes: pd.DataFrame,
) -> None:
    """Missing required columns in predictions should raise."""

    bad_predictions = pd.DataFrame(
        {
            "application_id": [
                "APP-0001",
            ]
        }
    )

    with pytest.raises(
        ValueError,
        match="Predictions missing",
    ):
        validate_pipeline_data(
            predictions=bad_predictions,
            outcomes=outcomes,
        )


# ---------------------------------------------------------------------------
# join_predictions_and_outcomes
# ---------------------------------------------------------------------------

def test_join_predictions_and_outcomes(
    predictions: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> None:
    """Joined data should contain predicted_probability and actual_default."""

    joined = join_predictions_and_outcomes(
        predictions=predictions,
        outcomes=outcomes,
        minimum_samples=1,
    )

    assert "predicted_probability" in joined.columns
    assert "actual_default" in joined.columns
    assert len(joined) == 10


def test_join_raises_when_no_matches() -> None:
    """No matching IDs should raise ValueError."""

    predictions = pd.DataFrame(
        {
            "application_id": ["A"],
            "predicted_probability": [0.5],
        }
    )

    outcomes = pd.DataFrame(
        {
            "application_id": ["B"],
            "actual_default": [0],
        }
    )

    with pytest.raises(
        ValueError,
        match="No matching",
    ):
        join_predictions_and_outcomes(
            predictions=predictions,
            outcomes=outcomes,
            minimum_samples=1,
        )


def test_join_raises_when_too_few_samples(
    predictions: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> None:
    """Below minimum sample threshold should raise ValueError."""

    with pytest.raises(
        ValueError,
        match="Too few",
    ):
        join_predictions_and_outcomes(
            predictions=predictions,
            outcomes=outcomes,
            minimum_samples=1000,
        )


# ---------------------------------------------------------------------------
# run_monitoring_pipeline
# ---------------------------------------------------------------------------

def test_pipeline_returns_success_status(
    predictions: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> None:
    """A successful pipeline run should return status='success'."""

    s3_client = Mock()
    cw_client = Mock()

    result = run_monitoring_pipeline(
        minimum_samples=1,
        predictions=predictions,
        outcomes=outcomes,
        model_version="v1",
        report_date=date(2026, 8, 1),
        environment="test",
        write_to_s3=True,
        publish_to_cloudwatch=True,
        run_id="test-run-001",
        s3_client=s3_client,
        cloudwatch_client=cw_client,
    )

    assert result[
        "status"
    ] == "success"


def test_pipeline_returns_run_id(
    predictions: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> None:
    """The result should include the run_id."""

    result = run_monitoring_pipeline(
        minimum_samples=1,
        predictions=predictions,
        outcomes=outcomes,
        model_version="v1",
        report_date=date(2026, 8, 1),
        write_to_s3=False,
        publish_to_cloudwatch=False,
        run_id="my-run-id",
    )

    assert result[
        "run_id"
    ] == "my-run-id"


def test_pipeline_calculates_performance_metrics(
    predictions: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> None:
    """Performance metrics DataFrame should be returned."""

    result = run_monitoring_pipeline(
        minimum_samples=1,
        predictions=predictions,
        outcomes=outcomes,
        model_version="v1",
        report_date=date(2026, 8, 1),
        write_to_s3=False,
        publish_to_cloudwatch=False,
    )

    perf = result[
        "performance_metrics"
    ]

    assert isinstance(
        perf,
        pd.DataFrame,
    )

    assert len(perf) == 1

    assert perf.iloc[0][
        "record_count"
    ] == 10


def test_pipeline_calculates_calibration_error(
    predictions: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> None:
    """ECE and MCE should be non-negative floats."""

    result = run_monitoring_pipeline(
        minimum_samples=1,
        predictions=predictions,
        outcomes=outcomes,
        write_to_s3=False,
        publish_to_cloudwatch=False,
    )

    assert isinstance(
        result[
            "expected_calibration_error"
        ],
        float,
    )

    assert isinstance(
        result[
            "maximum_calibration_error"
        ],
        float,
    )

    assert result[
        "expected_calibration_error"
    ] >= 0

    assert result[
        "maximum_calibration_error"
    ] >= 0


def test_pipeline_calculates_psi_when_reference_provided(
    predictions: pd.DataFrame,
    outcomes: pd.DataFrame,
    reference_data: pd.DataFrame,
) -> None:
    """PSI should be calculated when reference data is provided."""

    # Add feature columns to predictions matching reference
    predictions_with_features = predictions.copy()
    predictions_with_features[
        "income"
    ] = reference_data[
        "income"
    ].values[:10]
    predictions_with_features[
        "fico"
    ] = reference_data[
        "fico"
    ].values[:10]

    result = run_monitoring_pipeline(
        minimum_samples=1,
        predictions=predictions_with_features,
        outcomes=outcomes,
        reference_data=reference_data,
        write_to_s3=False,
        publish_to_cloudwatch=False,
    )

    assert result[
        "maximum_psi"
    ] is not None

    assert isinstance(
        result[
            "maximum_psi"
        ],
        float,
    )


def test_pipeline_skips_psi_when_no_reference(
    predictions: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> None:
    """PSI should be zero when no reference data is provided."""

    result = run_monitoring_pipeline(
        minimum_samples=1,
        predictions=predictions,
        outcomes=outcomes,
        reference_data=None,
        write_to_s3=False,
        publish_to_cloudwatch=False,
    )

    assert result[
        "maximum_psi"
    ] == 0.0


def test_pipeline_writes_to_s3(
    predictions: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> None:
    """write_to_s3=True should call put_object for each dataset."""

    s3_client = Mock()

    result = run_monitoring_pipeline(
        minimum_samples=1,
        predictions=predictions,
        outcomes=outcomes,
        write_to_s3=True,
        publish_to_cloudwatch=False,
        s3_client=s3_client,
    )

    assert result[
        "s3_dataset_results"
    ] is not None

    # At minimum: performance_metrics, calibration_metrics, pipeline_runs
    assert s3_client.put_object.call_count >= 3


def test_pipeline_does_not_write_html(
    predictions: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> None:
    """The pipeline must never write HTML files to S3."""

    s3_client = Mock()

    run_monitoring_pipeline(
        minimum_samples=1,
        predictions=predictions,
        outcomes=outcomes,
        write_to_s3=True,
        publish_to_cloudwatch=False,
        s3_client=s3_client,
    )

    s3_keys = [
        call.kwargs.get(
            "Key",
            "",
        )
        for call in s3_client.put_object.call_args_list
    ]

    assert not any(
        ".html" in key
        for key in s3_keys
    )


def test_pipeline_publishes_cloudwatch_metrics(
    predictions: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> None:
    """publish_to_cloudwatch=True should call put_metric_data."""

    cw_client = Mock()

    result = run_monitoring_pipeline(
        minimum_samples=1,
        predictions=predictions,
        outcomes=outcomes,
        write_to_s3=False,
        publish_to_cloudwatch=True,
        cloudwatch_client=cw_client,
    )

    assert cw_client.put_metric_data.call_count >= 1


def test_pipeline_does_not_call_generate_html_report(
    predictions: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> None:
    """The pipeline must never call generate_html_report()."""

    with patch(
        "app.monitoring.report.generate_html_report"
    ) as mock_html:
        run_monitoring_pipeline(
        minimum_samples=1,
            predictions=predictions,
            outcomes=outcomes,
            write_to_s3=False,
            publish_to_cloudwatch=False,
        )

        mock_html.assert_not_called()


def test_pipeline_returns_record_count(
    predictions: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> None:
    """Record count should equal the number of matched rows."""

    result = run_monitoring_pipeline(
        minimum_samples=1,
        predictions=predictions,
        outcomes=outcomes,
        write_to_s3=False,
        publish_to_cloudwatch=False,
    )

    assert result[
        "record_count"
    ] == 10


def test_pipeline_stamps_model_version(
    predictions: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> None:
    """model_version should appear in performance metrics."""

    result = run_monitoring_pipeline(
        minimum_samples=1,
        predictions=predictions,
        outcomes=outcomes,
        model_version="test-model-v99",
        write_to_s3=False,
        publish_to_cloudwatch=False,
    )

    assert result[
        "model_version"
    ] == "test-model-v99"

    perf = result[
        "performance_metrics"
    ]

    assert perf.iloc[0][
        "model_version"
    ] == "test-model-v99"


def test_pipeline_no_auto_recalibration(
    predictions: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> None:
    """Predicted probabilities must not change after pipeline runs."""

    original_probs = predictions[
        "predicted_probability"
    ].tolist()

    run_monitoring_pipeline(
        minimum_samples=1,
        predictions=predictions,
        outcomes=outcomes,
        write_to_s3=False,
        publish_to_cloudwatch=False,
    )

    # Predictions DataFrame must be unmodified
    assert predictions[
        "predicted_probability"
    ].tolist() == original_probs
