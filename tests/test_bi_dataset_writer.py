"""Tests for the S3 Parquet BI dataset writer."""

from datetime import date
from unittest.mock import Mock, call

import pandas as pd
import pytest

from app.monitoring.bi_dataset_writer import (
    VALID_DATASET_NAMES,
    build_partitioned_s3_key,
    dataframe_to_parquet_bytes,
    validate_dataframe,
    validate_dataset_name,
    write_dataframe_to_s3,
    write_monitoring_datasets,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Return a small sample DataFrame."""

    return pd.DataFrame(
        {
            "auc": [0.84],
            "ks": [0.51],
            "record_count": [100],
        }
    )


# ---------------------------------------------------------------------------
# validate_dataset_name
# ---------------------------------------------------------------------------

def test_valid_dataset_names_are_accepted() -> None:
    """All approved dataset names should be accepted."""

    for name in VALID_DATASET_NAMES:
        validate_dataset_name(
            name
        )


def test_invalid_dataset_name_is_rejected() -> None:
    """An unrecognised dataset name should raise ValueError."""

    with pytest.raises(
        ValueError,
        match="Invalid dataset name",
    ):
        validate_dataset_name(
            "html_report"
        )


# ---------------------------------------------------------------------------
# validate_dataframe
# ---------------------------------------------------------------------------

def test_validate_dataframe_accepts_non_empty_df(
    sample_df: pd.DataFrame,
) -> None:
    """A non-empty DataFrame should be accepted."""

    validate_dataframe(
        sample_df,
        "performance_metrics",
    )


def test_validate_dataframe_rejects_non_dataframe() -> None:
    """Non-DataFrame input must be rejected."""

    with pytest.raises(
        TypeError,
        match="must be a pandas DataFrame",
    ):
        validate_dataframe(
            [],
            "performance_metrics",
        )


def test_validate_dataframe_rejects_empty_df() -> None:
    """An empty DataFrame must be rejected."""

    with pytest.raises(
        ValueError,
        match="empty",
    ):
        validate_dataframe(
            pd.DataFrame(),
            "performance_metrics",
        )


# ---------------------------------------------------------------------------
# build_partitioned_s3_key
# ---------------------------------------------------------------------------

def test_build_partitioned_s3_key_format() -> None:
    """The S3 key should follow Hive-style partition format."""

    key = build_partitioned_s3_key(
        dataset_name="performance_metrics",
        model_version="credit-risk-model-v1",
        report_date=date(
            2026,
            8,
            1,
        ),
        analytics_prefix="analytics",
        run_id="test-run-001",
    )

    assert key.startswith(
        "analytics/performance_metrics/"
        "model_version=credit-risk-model-v1/"
        "report_date=2026-08-01/"
    )

    assert key.endswith(
        "part-test-run-001.parquet"
    )


def test_build_partitioned_s3_key_strips_prefix_slashes() -> None:
    """Leading and trailing slashes in the prefix should be stripped."""

    key = build_partitioned_s3_key(
        dataset_name="psi_metrics",
        model_version="v1",
        report_date=date(2026, 1, 1),
        analytics_prefix="/analytics/",
        run_id="r1",
    )

    assert key.startswith(
        "analytics/psi_metrics/"
    )


def test_build_partitioned_s3_key_uses_today_when_no_date(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When report_date is None the key should use today's date."""

    import datetime as dt
    from app.monitoring import bi_dataset_writer

    class FakeDate(dt.date):
        @classmethod
        def today(cls) -> "FakeDate":
            return cls(
                2026,
                8,
                15,
            )

    monkeypatch.setattr(
        bi_dataset_writer,
        "date",
        FakeDate,
    )

    key = build_partitioned_s3_key(
        dataset_name="pipeline_runs",
        model_version="v1",
        run_id="r1",
    )

    assert "report_date=2026-08-15" in key


def test_build_partitioned_s3_key_generates_uuid_run_id() -> None:
    """When run_id is None a UUID should be generated."""

    key = build_partitioned_s3_key(
        dataset_name="pipeline_runs",
        model_version="v1",
        report_date=date(2026, 1, 1),
    )

    # UUID v4 is 36 characters (8-4-4-4-12 with hyphens)
    part = key.split(
        "part-"
    )[-1].replace(
        ".parquet",
        "",
    )

    assert len(part) == 36


# ---------------------------------------------------------------------------
# dataframe_to_parquet_bytes
# ---------------------------------------------------------------------------

def test_dataframe_to_parquet_bytes_produces_valid_parquet(
    sample_df: pd.DataFrame,
) -> None:
    """Serialized bytes should be readable back as a DataFrame."""

    parquet_bytes = dataframe_to_parquet_bytes(
        sample_df
    )

    from io import BytesIO

    result = pd.read_parquet(
        BytesIO(
            parquet_bytes
        )
    )

    assert list(
        result.columns
    ) == list(
        sample_df.columns
    )

    assert len(
        result
    ) == len(
        sample_df
    )


# ---------------------------------------------------------------------------
# write_dataframe_to_s3
# ---------------------------------------------------------------------------

def test_write_dataframe_to_s3_calls_put_object(
    sample_df: pd.DataFrame,
) -> None:
    """The function should call put_object exactly once."""

    client = Mock()

    result = write_dataframe_to_s3(
        dataframe=sample_df,
        dataset_name="performance_metrics",
        bucket="test-bucket",
        model_version="v1",
        report_date=date(2026, 8, 1),
        analytics_prefix="analytics",
        run_id="run-001",
        s3_client=client,
    )

    client.put_object.assert_called_once()

    call_kwargs = client.put_object.call_args.kwargs

    assert call_kwargs[
        "Bucket"
    ] == "test-bucket"

    assert "analytics/performance_metrics/" in call_kwargs[
        "Key"
    ]

    assert call_kwargs[
        "ContentType"
    ] == "application/octet-stream"


def test_write_dataframe_to_s3_returns_s3_uri(
    sample_df: pd.DataFrame,
) -> None:
    """The result should contain a complete S3 URI."""

    client = Mock()

    result = write_dataframe_to_s3(
        dataframe=sample_df,
        dataset_name="performance_metrics",
        bucket="my-bucket",
        model_version="v1",
        report_date=date(2026, 8, 1),
        s3_client=client,
    )

    assert result[
        "s3_uri"
    ].startswith(
        "s3://my-bucket/"
    )

    assert result[
        "dataset_name"
    ] == "performance_metrics"

    assert result[
        "row_count"
    ] == "1"


def test_write_dataframe_to_s3_rejects_invalid_name(
    sample_df: pd.DataFrame,
) -> None:
    """An invalid dataset name must be rejected before any S3 call."""

    client = Mock()

    with pytest.raises(
        ValueError,
        match="Invalid dataset name",
    ):
        write_dataframe_to_s3(
            dataframe=sample_df,
            dataset_name="monitoring_report_html",
            s3_client=client,
        )

    client.put_object.assert_not_called()


def test_write_dataframe_to_s3_includes_metadata(
    sample_df: pd.DataFrame,
) -> None:
    """S3 object metadata should include dataset-name and model-version."""

    client = Mock()

    write_dataframe_to_s3(
        dataframe=sample_df,
        dataset_name="psi_metrics",
        bucket="b",
        model_version="v2",
        report_date=date(2026, 1, 1),
        environment="production",
        run_id="r1",
        s3_client=client,
    )

    metadata = client.put_object.call_args.kwargs[
        "Metadata"
    ]

    assert metadata[
        "dataset-name"
    ] == "psi_metrics"

    assert metadata[
        "model-version"
    ] == "v2"

    assert metadata[
        "environment"
    ] == "production"


# ---------------------------------------------------------------------------
# write_monitoring_datasets
# ---------------------------------------------------------------------------

def test_write_monitoring_datasets_writes_all_datasets() -> None:
    """All supplied datasets should be written to S3."""

    client = Mock()

    datasets = {
        "performance_metrics": pd.DataFrame(
            [{"auc": 0.84}]
        ),
        "psi_metrics": pd.DataFrame(
            [{"feature": "income", "psi": 0.02}]
        ),
    }

    result = write_monitoring_datasets(
        datasets=datasets,
        bucket="b",
        model_version="v1",
        report_date=date(2026, 8, 1),
        run_id="r1",
        s3_client=client,
    )

    assert result[
        "dataset_count"
    ] == 2

    assert result[
        "success"
    ] is True

    assert client.put_object.call_count == 2


def test_write_monitoring_datasets_shares_run_id() -> None:
    """All datasets in one call should share the same run_id."""

    client = Mock()

    datasets = {
        "performance_metrics": pd.DataFrame(
            [{"auc": 0.84}]
        ),
        "calibration_metrics": pd.DataFrame(
            [{"calibration_band": 1, "record_count": 10}]
        ),
    }

    result = write_monitoring_datasets(
        datasets=datasets,
        bucket="b",
        model_version="v1",
        report_date=date(2026, 8, 1),
        run_id="shared-run",
        s3_client=client,
    )

    assert result[
        "run_id"
    ] == "shared-run"

    # Both S3 keys should include the shared run_id
    keys = [
        call.kwargs[
            "Key"
        ]
        for call in client.put_object.call_args_list
    ]

    assert all(
        "shared-run" in key
        for key in keys
    )


def test_write_monitoring_datasets_rejects_empty_mapping() -> None:
    """An empty datasets mapping must be rejected."""

    with pytest.raises(
        ValueError,
        match="empty",
    ):
        write_monitoring_datasets(
            datasets={},
            s3_client=Mock(),
        )


def test_write_monitoring_datasets_captures_errors() -> None:
    """Write errors for individual datasets should be captured in errors dict."""

    client = Mock()

    client.put_object.side_effect = Exception(
        "S3 connection failed"
    )

    datasets = {
        "performance_metrics": pd.DataFrame(
            [{"auc": 0.84}]
        ),
    }

    result = write_monitoring_datasets(
        datasets=datasets,
        s3_client=client,
    )

    assert result[
        "success"
    ] is False

    assert "performance_metrics" in result[
        "errors"
    ]

    assert "S3 connection failed" in result[
        "errors"
    ][
        "performance_metrics"
    ]


def test_write_monitoring_datasets_uses_shared_client() -> None:
    """A single S3 client should be reused across all datasets."""

    client = Mock()

    datasets = {
        "performance_metrics": pd.DataFrame(
            [{"auc": 0.84}]
        ),
        "psi_metrics": pd.DataFrame(
            [{"feature": "income", "psi": 0.01}]
        ),
        "pipeline_runs": pd.DataFrame(
            [{"run_id": "r1", "status": "success"}]
        ),
    }

    write_monitoring_datasets(
        datasets=datasets,
        bucket="b",
        model_version="v1",
        report_date=date(2026, 8, 1),
        run_id="r1",
        s3_client=client,
    )

    assert client.put_object.call_count == 3
