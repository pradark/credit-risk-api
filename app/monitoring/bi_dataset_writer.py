"""Write partitioned Parquet analytical datasets to Amazon S3.

This module is the production output layer for the monitoring pipeline.
All monitoring results are written as partitioned Parquet datasets to S3
for consumption by AWS Glue, Amazon Athena, and Amazon QuickSight.

No HTML files are written. No CSV files are written to S3 as primary
monitoring outputs.
"""

from __future__ import annotations

import uuid
from datetime import date
from io import BytesIO
from typing import Any

import boto3
import pandas as pd

from app.config import (
    AWS_REGION,
    ENVIRONMENT,
    MODEL_VERSION,
    S3_ANALYTICS_PREFIX,
    S3_BUCKET,
)


VALID_DATASET_NAMES = frozenset(
    {
        "performance_metrics",
        "calibration_metrics",
        "psi_metrics",
        "segment_metrics",
        "adverse_reason_summary",
        "pipeline_runs",
    }
)


def validate_dataset_name(
    dataset_name: str,
) -> None:
    """Validate the analytical dataset name.

    Parameters
    ----------
    dataset_name:
        One of the supported monitoring dataset names.

    Raises
    ------
    ValueError
        If the name is not one of the approved dataset names.
    """

    if dataset_name not in VALID_DATASET_NAMES:
        raise ValueError(
            f"Invalid dataset name: '{dataset_name}'. "
            f"Must be one of: {sorted(VALID_DATASET_NAMES)}"
        )


def validate_dataframe(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> None:
    """Validate a DataFrame before writing to S3.

    Parameters
    ----------
    dataframe:
        The analytical dataset to validate.
    dataset_name:
        The target dataset name (used in error messages).

    Raises
    ------
    TypeError
        If dataframe is not a pandas DataFrame.
    ValueError
        If the DataFrame is empty.
    """

    if not isinstance(
        dataframe,
        pd.DataFrame,
    ):
        raise TypeError(
            f"{dataset_name}: data must be a pandas DataFrame."
        )

    if dataframe.empty:
        raise ValueError(
            f"{dataset_name}: DataFrame is empty and cannot be written."
        )


def build_partitioned_s3_key(
    dataset_name: str,
    model_version: str = MODEL_VERSION,
    report_date: date | None = None,
    analytics_prefix: str = S3_ANALYTICS_PREFIX,
    run_id: str | None = None,
) -> str:
    """Build a partitioned S3 object key for a monitoring dataset.

    The key format follows Hive-style partitioning for Glue/Athena:

        <analytics_prefix>/<dataset_name>/
            model_version=<version>/
            report_date=<YYYY-MM-DD>/
            part-<uuid>.parquet

    Parameters
    ----------
    dataset_name:
        Approved monitoring dataset name.
    model_version:
        Model version string used in the partition key.
    report_date:
        Report date; defaults to today if None.
    analytics_prefix:
        Root S3 prefix for analytical datasets.
    run_id:
        Optional run identifier; a UUID is generated if None.

    Returns
    -------
    str
        Fully qualified partitioned S3 key.
    """

    validate_dataset_name(
        dataset_name
    )

    resolved_date = (
        report_date
        if report_date is not None
        else date.today()
    )

    resolved_run_id = (
        run_id
        if run_id is not None
        else str(
            uuid.uuid4()
        )
    )

    prefix = analytics_prefix.strip(
        "/ "
    )

    return (
        f"{prefix}/{dataset_name}/"
        f"model_version={model_version}/"
        f"report_date={resolved_date.isoformat()}/"
        f"part-{resolved_run_id}.parquet"
    )


def dataframe_to_parquet_bytes(
    dataframe: pd.DataFrame,
) -> bytes:
    """Serialize a DataFrame to Parquet bytes in memory.

    Parameters
    ----------
    dataframe:
        DataFrame to serialize.

    Returns
    -------
    bytes
        Parquet-encoded bytes ready for S3 upload.
    """

    buffer = BytesIO()

    dataframe.to_parquet(
        buffer,
        index=False,
        engine="pyarrow",
    )

    return buffer.getvalue()


def write_dataframe_to_s3(
    dataframe: pd.DataFrame,
    dataset_name: str,
    bucket: str = S3_BUCKET,
    model_version: str = MODEL_VERSION,
    report_date: date | None = None,
    analytics_prefix: str = S3_ANALYTICS_PREFIX,
    environment: str = ENVIRONMENT,
    run_id: str | None = None,
    s3_client: Any | None = None,
) -> dict[str, str]:
    """Write a single analytical DataFrame to S3 as Parquet.

    Parameters
    ----------
    dataframe:
        The monitoring dataset to write.
    dataset_name:
        Approved analytical dataset name.
    bucket:
        Target S3 bucket.
    model_version:
        Model version string.
    report_date:
        Report date partition; defaults to today.
    analytics_prefix:
        Root prefix for analytical datasets.
    environment:
        Deployment environment tag.
    run_id:
        Pipeline run identifier; a UUID is generated if None.
    s3_client:
        Injected boto3 S3 client for testing. A real client is
        created if None.

    Returns
    -------
    dict[str, str]
        Dictionary with dataset_name, s3_key, s3_uri, bucket, and
        row_count.
    """

    validate_dataset_name(
        dataset_name
    )

    validate_dataframe(
        dataframe,
        dataset_name,
    )

    resolved_run_id = (
        run_id
        if run_id is not None
        else str(
            uuid.uuid4()
        )
    )

    s3_key = build_partitioned_s3_key(
        dataset_name=dataset_name,
        model_version=model_version,
        report_date=report_date,
        analytics_prefix=analytics_prefix,
        run_id=resolved_run_id,
    )

    parquet_bytes = dataframe_to_parquet_bytes(
        dataframe
    )

    client = (
        s3_client
        if s3_client is not None
        else boto3.client(
            "s3",
            region_name=AWS_REGION,
        )
    )

    client.put_object(
        Bucket=bucket,
        Key=s3_key,
        Body=parquet_bytes,
        ContentType="application/octet-stream",
        Metadata={
            "dataset-name": dataset_name,
            "model-version": model_version,
            "environment": environment,
            "run-id": resolved_run_id,
        },
    )

    return {
        "dataset_name": dataset_name,
        "s3_key": s3_key,
        "s3_uri": f"s3://{bucket}/{s3_key}",
        "bucket": bucket,
        "row_count": str(
            len(
                dataframe
            )
        ),
    }


def write_monitoring_datasets(
    datasets: dict[str, pd.DataFrame],
    bucket: str = S3_BUCKET,
    model_version: str = MODEL_VERSION,
    report_date: date | None = None,
    analytics_prefix: str = S3_ANALYTICS_PREFIX,
    environment: str = ENVIRONMENT,
    run_id: str | None = None,
    s3_client: Any | None = None,
) -> dict[str, object]:
    """Write all monitoring analytical datasets to S3.

    Iterates over each dataset in the supplied mapping, validates it,
    and uploads it to the appropriate partitioned S3 path. All datasets
    share the same run_id so they can be correlated in Athena.

    Parameters
    ----------
    datasets:
        Mapping of dataset_name -> DataFrame. Dataset names must be
        members of VALID_DATASET_NAMES.
    bucket:
        Target S3 bucket.
    model_version:
        Model version string.
    report_date:
        Report date partition; defaults to today.
    analytics_prefix:
        Root prefix for analytical datasets.
    environment:
        Deployment environment tag.
    run_id:
        Shared run identifier for all datasets; generated if None.
    s3_client:
        Injected boto3 S3 client. A real client is created if None.

    Returns
    -------
    dict[str, object]
        Summary with run_id, dataset_count, results mapping, and any
        errors.
    """

    if not datasets:
        raise ValueError(
            "datasets mapping is empty. "
            "At least one dataset must be provided."
        )

    resolved_run_id = (
        run_id
        if run_id is not None
        else str(
            uuid.uuid4()
        )
    )

    client = (
        s3_client
        if s3_client is not None
        else boto3.client(
            "s3",
            region_name=AWS_REGION,
        )
    )

    results: dict[str, dict[str, str]] = {}
    errors: dict[str, str] = {}

    for dataset_name, dataframe in datasets.items():
        try:
            result = write_dataframe_to_s3(
                dataframe=dataframe,
                dataset_name=dataset_name,
                bucket=bucket,
                model_version=model_version,
                report_date=report_date,
                analytics_prefix=analytics_prefix,
                environment=environment,
                run_id=resolved_run_id,
                s3_client=client,
            )

            results[
                dataset_name
            ] = result

        except Exception as exc:
            errors[
                dataset_name
            ] = str(
                exc
            )

    return {
        "run_id": resolved_run_id,
        "dataset_count": len(
            results
        ),
        "results": results,
        "errors": errors,
        "success": len(
            errors
        ) == 0,
    }
