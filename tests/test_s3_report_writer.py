"""Tests for S3 monitoring report uploads."""

from datetime import date
from pathlib import Path
from unittest.mock import Mock

import pytest

from app.monitoring.s3_report_writer import (
    build_artifact_s3_keys,
    build_report_s3_prefix,
    collect_monitoring_artifacts,
    infer_content_type,
    normalize_s3_prefix,
    upload_monitoring_artifacts,
    validate_artifact_path,
    validate_bucket_name,
    validate_model_version,
    validate_report_date,
)


@pytest.fixture
def report_artifacts(
    tmp_path: Path,
) -> dict[str, Path]:
    """Create local monitoring files for S3 upload tests."""

    artifacts = {
        "performance_metrics_path": (
            tmp_path
            / "performance_metrics.csv"
        ),
        "calibration_table_path": (
            tmp_path
            / "calibration_table.csv"
        ),
        "roc_curve_path": (
            tmp_path
            / "roc_curve.png"
        ),
        "expected_vs_actual_path": (
            tmp_path
            / "expected_vs_actual.png"
        ),
    }

    for path in artifacts.values():
        path.write_bytes(
            b"test-content"
        )

    return artifacts


@pytest.fixture
def html_report_path(
    tmp_path: Path,
) -> Path:
    """Create a local HTML monitoring report."""

    path = (
        tmp_path
        / "monitoring_report.html"
    )

    path.write_text(
        "<html><body>Report</body></html>",
        encoding="utf-8",
    )

    return path


def test_validate_bucket_name_accepts_valid_value() -> None:
    """A non-empty bucket name should be accepted."""

    validate_bucket_name(
        "credit-risk-monitoring"
    )


def test_validate_bucket_name_rejects_non_string() -> None:
    """The bucket name must be a string."""

    with pytest.raises(
        TypeError,
        match="bucket must be a string",
    ):
        validate_bucket_name(
            123
        )


def test_validate_bucket_name_rejects_empty_value() -> None:
    """An empty bucket name should be rejected."""

    with pytest.raises(
        ValueError,
        match="bucket cannot be empty",
    ):
        validate_bucket_name(
            " "
        )


def test_validate_model_version_accepts_valid_value() -> None:
    """A non-empty model version should be accepted."""

    validate_model_version(
        "credit-risk-model-v1"
    )


def test_validate_model_version_rejects_empty_value() -> None:
    """An empty model version should be rejected."""

    with pytest.raises(
        ValueError,
        match="model_version cannot be empty",
    ):
        validate_model_version(
            ""
        )


def test_validate_report_date_rejects_non_date() -> None:
    """The report date must be a date object."""

    with pytest.raises(
        TypeError,
        match="report_date must be a date",
    ):
        validate_report_date(
            "2026-08-01"
        )


def test_normalize_s3_prefix() -> None:
    """Leading and trailing separators should be removed."""

    result = normalize_s3_prefix(
        "/monitoring-reports/"
    )

    assert result == (
        "monitoring-reports"
    )


def test_normalize_s3_prefix_rejects_empty_value() -> None:
    """An empty S3 prefix should be rejected."""

    with pytest.raises(
        ValueError,
        match="prefix cannot be empty",
    ):
        normalize_s3_prefix(
            "///"
        )


def test_build_report_s3_prefix() -> None:
    """The S3 prefix should contain model and date partitions."""

    result = build_report_s3_prefix(
        prefix="monitoring-reports",
        model_version="model-v2",
        report_date=date(
            2026,
            8,
            1,
        ),
    )

    assert result == (
        "monitoring-reports/"
        "model_version=model-v2/"
        "dt=2026-08-01"
    )


@pytest.mark.parametrize(
    (
        "filename",
        "expected_content_type",
    ),
    [
        (
            "performance.csv",
            "text/csv",
        ),
        (
            "report.html",
            "text/html",
        ),
        (
            "curve.png",
            "image/png",
        ),
        (
            "payload.json",
            "application/json",
        ),
        (
            "data.parquet",
            "application/octet-stream",
        ),
        (
            "unknown.bin",
            "application/octet-stream",
        ),
    ],
)
def test_infer_content_type(
    filename: str,
    expected_content_type: str,
) -> None:
    """Content type should be inferred from the extension."""

    result = infer_content_type(
        filename
    )

    assert result == (
        expected_content_type
    )


def test_validate_artifact_path_accepts_file(
    tmp_path: Path,
) -> None:
    """An existing file should be returned as a Path."""

    artifact = (
        tmp_path
        / "metrics.csv"
    )

    artifact.write_text(
        "auc,ks\n0.84,0.51\n",
        encoding="utf-8",
    )

    result = validate_artifact_path(
        artifact
    )

    assert result == artifact


def test_validate_artifact_path_rejects_missing_file(
    tmp_path: Path,
) -> None:
    """A missing artifact should be rejected."""

    missing_path = (
        tmp_path
        / "missing.csv"
    )

    with pytest.raises(
        FileNotFoundError,
        match="does not exist",
    ):
        validate_artifact_path(
            missing_path
        )


def test_validate_artifact_path_rejects_directory(
    tmp_path: Path,
) -> None:
    """An artifact path must point to a file."""

    with pytest.raises(
        ValueError,
        match="is not a file",
    ):
        validate_artifact_path(
            tmp_path
        )


def test_collect_monitoring_artifacts(
    report_artifacts: dict[
        str,
        Path,
    ],
    html_report_path: Path,
) -> None:
    """All dashboard and HTML artifacts should be collected."""

    result = collect_monitoring_artifacts(
        dashboard=report_artifacts,
        report_path=html_report_path,
    )

    assert set(
        result
    ) == {
        "performance_metrics.csv",
        "calibration_table.csv",
        "roc_curve.png",
        "expected_vs_actual.png",
        "monitoring_report.html",
    }

    assert result[
        "monitoring_report.html"
    ] == html_report_path


def test_collect_monitoring_artifacts_rejects_missing_key(
    report_artifacts: dict[
        str,
        Path,
    ],
    html_report_path: Path,
) -> None:
    """Missing dashboard paths should be rejected."""

    dashboard = dict(
        report_artifacts
    )

    dashboard.pop(
        "roc_curve_path"
    )

    with pytest.raises(
        ValueError,
        match=(
            "Dashboard is missing "
            "required artifact path"
        ),
    ):
        collect_monitoring_artifacts(
            dashboard=dashboard,
            report_path=html_report_path,
        )


def test_build_artifact_s3_keys(
    report_artifacts: dict[
        str,
        Path,
    ],
) -> None:
    """Every artifact should receive a partitioned S3 key."""

    artifacts = {
        "performance_metrics.csv": (
            report_artifacts[
                "performance_metrics_path"
            ]
        ),
        "roc_curve.png": (
            report_artifacts[
                "roc_curve_path"
            ]
        ),
    }

    result = build_artifact_s3_keys(
        artifacts=artifacts,
        prefix="monitoring-reports",
        model_version="model-v2",
        report_date=date(
            2026,
            8,
            1,
        ),
    )

    assert result == {
        "performance_metrics.csv": (
            "monitoring-reports/"
            "model_version=model-v2/"
            "dt=2026-08-01/"
            "performance_metrics.csv"
        ),
        "roc_curve.png": (
            "monitoring-reports/"
            "model_version=model-v2/"
            "dt=2026-08-01/"
            "roc_curve.png"
        ),
    }


def test_build_artifact_s3_keys_rejects_empty_artifacts() -> None:
    """An empty artifact mapping should be rejected."""

    with pytest.raises(
        ValueError,
        match="artifacts cannot be empty",
    ):
        build_artifact_s3_keys(
            artifacts={}
        )


def test_upload_monitoring_artifacts(
    report_artifacts: dict[
        str,
        Path,
    ],
    html_report_path: Path,
) -> None:
    """All five report artifacts should be uploaded."""

    client = Mock()

    result = upload_monitoring_artifacts(
        dashboard=report_artifacts,
        report_path=html_report_path,
        bucket=(
            "credit-risk-monitoring-test"
        ),
        prefix="monitoring-reports",
        model_version="model-v2",
        report_date=date(
            2026,
            8,
            1,
        ),
        s3_client=client,
    )

    assert result[
        "bucket"
    ] == (
        "credit-risk-monitoring-test"
    )

    assert result[
        "prefix"
    ] == (
        "monitoring-reports/"
        "model_version=model-v2/"
        "dt=2026-08-01"
    )

    assert result[
        "artifact_count"
    ] == 5

    assert (
        client
        .upload_file
        .call_count
        == 5
    )

    assert set(
        result[
            "artifacts"
        ]
    ) == {
        "performance_metrics.csv",
        "calibration_table.csv",
        "roc_curve.png",
        "expected_vs_actual.png",
        "monitoring_report.html",
    }


def test_upload_uses_expected_bucket_and_keys(
    report_artifacts: dict[
        str,
        Path,
    ],
    html_report_path: Path,
) -> None:
    """S3 calls should use the requested bucket and partition."""

    client = Mock()

    upload_monitoring_artifacts(
        dashboard=report_artifacts,
        report_path=html_report_path,
        bucket="test-bucket",
        prefix="reports",
        model_version="model-v3",
        report_date=date(
            2026,
            8,
            2,
        ),
        s3_client=client,
    )

    calls = (
        client
        .upload_file
        .call_args_list
    )

    uploaded_keys = {
        call.kwargs[
            "Key"
        ]
        for call in calls
    }

    assert (
        "reports/"
        "model_version=model-v3/"
        "dt=2026-08-02/"
        "performance_metrics.csv"
    ) in uploaded_keys

    assert all(
        call.kwargs[
            "Bucket"
        ]
        == "test-bucket"
        for call in calls
    )


def test_upload_sets_content_types(
    report_artifacts: dict[
        str,
        Path,
    ],
    html_report_path: Path,
) -> None:
    """Each uploaded artifact should receive a content type."""

    client = Mock()

    upload_monitoring_artifacts(
        dashboard=report_artifacts,
        report_path=html_report_path,
        bucket="test-bucket",
        report_date=date(
            2026,
            8,
            1,
        ),
        s3_client=client,
    )

    content_types = {
        Path(
            call.kwargs[
                "Filename"
            ]
        ).suffix: (
            call.kwargs[
                "ExtraArgs"
            ][
                "ContentType"
            ]
        )
        for call in (
            client
            .upload_file
            .call_args_list
        )
    }

    assert content_types[
        ".csv"
    ] == "text/csv"

    assert content_types[
        ".png"
    ] == "image/png"

    assert content_types[
        ".html"
    ] == "text/html"


def test_upload_adds_metadata(
    report_artifacts: dict[
        str,
        Path,
    ],
    html_report_path: Path,
) -> None:
    """Additional metadata should be included with every upload."""

    client = Mock()

    upload_monitoring_artifacts(
        dashboard=report_artifacts,
        report_path=html_report_path,
        bucket="test-bucket",
        report_date=date(
            2026,
            8,
            1,
        ),
        additional_metadata={
            "environment": "test",
            "portfolio": "personal-loans",
        },
        s3_client=client,
    )

    for call in (
        client
        .upload_file
        .call_args_list
    ):
        assert call.kwargs[
            "ExtraArgs"
        ][
            "Metadata"
        ] == {
            "environment": "test",
            "portfolio": "personal-loans",
        }


def test_upload_returns_s3_uris(
    report_artifacts: dict[
        str,
        Path,
    ],
    html_report_path: Path,
) -> None:
    """The upload result should contain complete S3 URIs."""

    client = Mock()

    result = upload_monitoring_artifacts(
        dashboard=report_artifacts,
        report_path=html_report_path,
        bucket="test-bucket",
        prefix="reports",
        model_version="model-v2",
        report_date=date(
            2026,
            8,
            1,
        ),
        s3_client=client,
    )

    report_result = result[
        "artifacts"
    ][
        "monitoring_report.html"
    ]

    assert report_result[
        "s3_uri"
    ] == (
        "s3://test-bucket/"
        "reports/"
        "model_version=model-v2/"
        "dt=2026-08-01/"
        "monitoring_report.html"
    )