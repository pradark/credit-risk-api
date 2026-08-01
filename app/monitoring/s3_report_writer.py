"""Upload model monitoring artifacts to Amazon S3."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any

import boto3

from app.config import (
    AWS_REGION,
    MODEL_VERSION,
    S3_BUCKET,
)


DEFAULT_REPORT_PREFIX = "monitoring-reports"

DASHBOARD_ARTIFACT_MAPPING = {
    "performance_metrics_path": (
        "performance_metrics.csv"
    ),
    "calibration_table_path": (
        "calibration_table.csv"
    ),
    "roc_curve_path": (
        "roc_curve.png"
    ),
    "expected_vs_actual_path": (
        "expected_vs_actual.png"
    ),
}

CONTENT_TYPES = {
    ".csv": "text/csv",
    ".html": "text/html",
    ".png": "image/png",
    ".json": "application/json",
    ".parquet": "application/octet-stream",
}


def validate_bucket_name(
    bucket: str,
) -> None:
    """Validate the target S3 bucket name."""

    if not isinstance(
        bucket,
        str,
    ):
        raise TypeError(
            "bucket must be a string."
        )

    if not bucket.strip():
        raise ValueError(
            "bucket cannot be empty."
        )


def validate_model_version(
    model_version: str,
) -> None:
    """Validate the model version used in S3 keys."""

    if not isinstance(
        model_version,
        str,
    ):
        raise TypeError(
            "model_version must be a string."
        )

    if not model_version.strip():
        raise ValueError(
            "model_version cannot be empty."
        )


def normalize_s3_prefix(
    prefix: str,
) -> str:
    """Normalize an S3 key prefix."""

    if not isinstance(
        prefix,
        str,
    ):
        raise TypeError(
            "prefix must be a string."
        )

    normalized_prefix = prefix.strip(
        "/ "
    )

    if not normalized_prefix:
        raise ValueError(
            "prefix cannot be empty."
        )

    return normalized_prefix


def validate_report_date(
    report_date: date,
) -> None:
    """Validate the report partition date."""

    if not isinstance(
        report_date,
        date,
    ):
        raise TypeError(
            "report_date must be a date."
        )


def build_report_s3_prefix(
    prefix: str = DEFAULT_REPORT_PREFIX,
    model_version: str = MODEL_VERSION,
    report_date: date | None = None,
) -> str:
    """Build the partitioned S3 prefix for a monitoring report."""

    normalized_prefix = normalize_s3_prefix(
        prefix
    )

    validate_model_version(
        model_version
    )

    resolved_date = (
        report_date
        if report_date is not None
        else date.today()
    )

    validate_report_date(
        resolved_date
    )

    return (
        f"{normalized_prefix}/"
        f"model_version={model_version}/"
        f"dt={resolved_date.isoformat()}"
    )


def infer_content_type(
    file_path: str | Path,
) -> str:
    """Infer an S3 content type from a file extension."""

    path = Path(
        file_path
    )

    return CONTENT_TYPES.get(
        path.suffix.lower(),
        "application/octet-stream",
    )


def validate_artifact_path(
    artifact_path: str | Path,
) -> Path:
    """Validate that a monitoring artifact exists and is a file."""

    path = Path(
        artifact_path
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Monitoring artifact does not exist: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Monitoring artifact is not a file: {path}"
        )

    return path


def collect_monitoring_artifacts(
    dashboard: Mapping[str, object],
    report_path: str | Path,
) -> dict[str, Path]:
    """Collect generated monitoring artifact paths."""

    if not isinstance(
        dashboard,
        Mapping,
    ):
        raise TypeError(
            "dashboard must be a mapping."
        )

    artifacts: dict[str, Path] = {}

    for (
        dashboard_key,
        output_filename,
    ) in DASHBOARD_ARTIFACT_MAPPING.items():
        if dashboard_key not in dashboard:
            raise ValueError(
                "Dashboard is missing required artifact "
                f"path: {dashboard_key}"
            )

        artifacts[
            output_filename
        ] = validate_artifact_path(
            dashboard[
                dashboard_key
            ]
        )

    artifacts[
        "monitoring_report.html"
    ] = validate_artifact_path(
        report_path
    )

    return artifacts


def build_artifact_s3_keys(
    artifacts: Mapping[str, Path],
    prefix: str = DEFAULT_REPORT_PREFIX,
    model_version: str = MODEL_VERSION,
    report_date: date | None = None,
) -> dict[str, str]:
    """Build an S3 object key for every monitoring artifact."""

    if not isinstance(
        artifacts,
        Mapping,
    ):
        raise TypeError(
            "artifacts must be a mapping."
        )

    if not artifacts:
        raise ValueError(
            "artifacts cannot be empty."
        )

    report_prefix = build_report_s3_prefix(
        prefix=prefix,
        model_version=model_version,
        report_date=report_date,
    )

    return {
        artifact_name: (
            f"{report_prefix}/{artifact_name}"
        )
        for artifact_name in artifacts
    }


def upload_monitoring_artifacts(
    dashboard: Mapping[str, object],
    report_path: str | Path,
    bucket: str = S3_BUCKET,
    prefix: str = DEFAULT_REPORT_PREFIX,
    model_version: str = MODEL_VERSION,
    report_date: date | None = None,
    additional_metadata: Mapping[
        str,
        str,
    ]
    | None = None,
    s3_client: Any | None = None,
) -> dict[str, object]:
    """Upload generated monitoring artifacts to Amazon S3.

    The dashboard must contain paths returned by
    generate_monitoring_dashboard(). The HTML report path is passed
    separately because generate_html_report() returns it after the
    dashboard dictionary has already been created.
    """

    validate_bucket_name(
        bucket
    )

    artifacts = collect_monitoring_artifacts(
        dashboard=dashboard,
        report_path=report_path,
    )

    object_keys = build_artifact_s3_keys(
        artifacts=artifacts,
        prefix=prefix,
        model_version=model_version,
        report_date=report_date,
    )

    client = (
        s3_client
        if s3_client is not None
        else boto3.client(
            "s3",
            region_name=AWS_REGION,
        )
    )

    normalized_metadata = {
        str(
            key
        ): str(
            value
        )
        for key, value in (
            additional_metadata
            or {}
        ).items()
    }

    uploaded_artifacts: dict[
        str,
        dict[str, str],
    ] = {}

    for (
        artifact_name,
        local_path,
    ) in artifacts.items():
        object_key = object_keys[
            artifact_name
        ]

        extra_args: dict[
            str,
            object,
        ] = {
            "ContentType": infer_content_type(
                local_path
            ),
        }

        if normalized_metadata:
            extra_args[
                "Metadata"
            ] = normalized_metadata

        client.upload_file(
            Filename=str(
                local_path
            ),
            Bucket=bucket,
            Key=object_key,
            ExtraArgs=extra_args,
        )

        uploaded_artifacts[
            artifact_name
        ] = {
            "local_path": str(
                local_path
            ),
            "s3_key": object_key,
            "s3_uri": (
                f"s3://{bucket}/{object_key}"
            ),
        }

    return {
        "bucket": bucket,
        "prefix": build_report_s3_prefix(
            prefix=prefix,
            model_version=model_version,
            report_date=report_date,
        ),
        "artifact_count": len(
            uploaded_artifacts
        ),
        "artifacts": uploaded_artifacts,
    }