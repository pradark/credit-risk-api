"""AWS Glue Data Catalog definitions for credit risk monitoring datasets.

Creates and maintains external Glue tables that point to S3 Parquet
datasets written by the monitoring pipeline. Tables are partitioned
by model_version and report_date for efficient Athena querying.

All operations are idempotent — calling them repeatedly is safe.
"""

from __future__ import annotations

from typing import Any

import boto3

from app.config import (
    AWS_REGION,
    GLUE_DATABASE_NAME,
    S3_ANALYTICS_PREFIX,
    S3_BUCKET,
)


# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------

PERFORMANCE_METRICS_COLUMNS = [
    ("record_count", "int"),
    ("default_count", "int"),
    ("non_default_count", "int"),
    ("auc", "double"),
    ("ks", "double"),
    ("gini", "double"),
    ("bad_rate", "double"),
    ("average_predicted_pd", "double"),
    ("prediction_standard_deviation", "double"),
    ("minimum_predicted_pd", "double"),
    ("maximum_predicted_pd", "double"),
    ("accuracy", "double"),
    ("balanced_accuracy", "double"),
    ("precision", "double"),
    ("recall", "double"),
    ("specificity", "double"),
    ("f1", "double"),
    ("false_positive_rate", "double"),
    ("false_negative_rate", "double"),
    ("predicted_positive_rate", "double"),
    ("brier_score", "double"),
    ("log_loss", "double"),
    ("calibration_error", "double"),
    ("true_positive", "int"),
    ("false_positive", "int"),
    ("true_negative", "int"),
    ("false_negative", "int"),
    ("classification_threshold", "double"),
    ("run_id", "string"),
    ("environment", "string"),
]

CALIBRATION_METRICS_COLUMNS = [
    ("calibration_band", "int"),
    ("record_count", "int"),
    ("default_count", "int"),
    ("minimum_predicted_pd", "double"),
    ("maximum_predicted_pd", "double"),
    ("average_predicted_pd", "double"),
    ("actual_default_rate", "double"),
    ("population_percentage", "double"),
    ("calibration_gap", "double"),
    ("absolute_calibration_gap", "double"),
    ("environment", "string"),
]

PSI_METRICS_COLUMNS = [
    ("feature", "string"),
    ("psi", "double"),
    ("status", "string"),
    ("environment", "string"),
]

SEGMENT_METRICS_COLUMNS = [
    ("segment_name", "string"),
    ("segment_value", "string"),
    ("record_count", "int"),
    ("default_count", "int"),
    ("bad_rate", "double"),
    ("average_predicted_pd", "double"),
    ("calibration_gap", "double"),
    ("environment", "string"),
]

ADVERSE_REASON_SUMMARY_COLUMNS = [
    ("reason_code", "string"),
    ("adverse_reason", "string"),
    ("selection_count", "int"),
    ("selection_rate", "double"),
    ("average_contribution", "double"),
    ("environment", "string"),
]

PIPELINE_RUNS_COLUMNS = [
    ("run_id", "string"),
    ("record_count", "int"),
    ("expected_calibration_error", "double"),
    ("maximum_calibration_error", "double"),
    ("maximum_psi", "double"),
    ("warning_feature_count", "int"),
    ("alert_feature_count", "int"),
    ("status", "string"),
    ("error", "string"),
    ("environment", "string"),
]

# Standard Hive-style partitions used by all tables
STANDARD_PARTITIONS = [
    {
        "Name": "model_version",
        "Type": "string",
        "Comment": "Model version identifier",
    },
    {
        "Name": "report_date",
        "Type": "date",
        "Comment": "Monitoring report date",
    },
]

# Table registry: name -> (columns, s3_prefix_suffix)
TABLE_REGISTRY: dict[str, tuple[list[tuple[str, str]], str]] = {
    "performance_metrics": (
        PERFORMANCE_METRICS_COLUMNS,
        "performance_metrics",
    ),
    "calibration_metrics": (
        CALIBRATION_METRICS_COLUMNS,
        "calibration_metrics",
    ),
    "psi_metrics": (
        PSI_METRICS_COLUMNS,
        "psi_metrics",
    ),
    "segment_metrics": (
        SEGMENT_METRICS_COLUMNS,
        "segment_metrics",
    ),
    "adverse_reason_summary": (
        ADVERSE_REASON_SUMMARY_COLUMNS,
        "adverse_reason_summary",
    ),
    "pipeline_runs": (
        PIPELINE_RUNS_COLUMNS,
        "pipeline_runs",
    ),
}


def _columns_to_glue(
    columns: list[tuple[str, str]],
) -> list[dict[str, str]]:
    """Convert (name, type) tuples to Glue column dicts."""

    return [
        {
            "Name": name,
            "Type": col_type,
        }
        for name, col_type in columns
    ]


def create_glue_database(
    database_name: str = GLUE_DATABASE_NAME,
    glue_client: Any | None = None,
) -> dict[str, str]:
    """Create the Glue database if it does not exist.

    Parameters
    ----------
    database_name:
        Name of the Glue database to create.
    glue_client:
        Injected Glue client for testing.

    Returns
    -------
    dict[str, str]
        Result with database_name and status.
    """

    client = (
        glue_client
        if glue_client is not None
        else boto3.client(
            "glue",
            region_name=AWS_REGION,
        )
    )

    try:
        client.get_database(
            Name=database_name
        )

        return {
            "database_name": database_name,
            "status": "already_exists",
        }

    except client.exceptions.EntityNotFoundException:
        client.create_database(
            DatabaseInput={
                "Name": database_name,
                "Description": (
                    "Credit risk model monitoring analytical datasets. "
                    "Partitioned Parquet tables for Athena and QuickSight."
                ),
            }
        )

        return {
            "database_name": database_name,
            "status": "created",
        }


def build_table_input(
    table_name: str,
    columns: list[tuple[str, str]],
    s3_location: str,
) -> dict[str, Any]:
    """Build a Glue CreateTable / UpdateTable input structure.

    Parameters
    ----------
    table_name:
        Glue table name.
    columns:
        List of (column_name, glue_type) tuples.
    s3_location:
        S3 URI for the table root location.

    Returns
    -------
    dict
        Glue TableInput dictionary.
    """

    return {
        "Name": table_name,
        "Description": (
            f"Credit risk monitoring: {table_name}. "
            "Partitioned Parquet. Managed by the monitoring pipeline."
        ),
        "StorageDescriptor": {
            "Columns": _columns_to_glue(
                columns
            ),
            "Location": s3_location,
            "InputFormat": (
                "org.apache.hadoop.mapred.TextInputFormat"
            ),
            "OutputFormat": (
                "org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat"
            ),
            "SerdeInfo": {
                "SerializationLibrary": (
                    "org.apache.hadoop.hive.ql.io.parquet.serde"
                    ".ParquetHiveSerDe"
                ),
                "Parameters": {
                    "serialization.format": "1",
                },
            },
            "Compressed": False,
            "StoredAsSubDirectories": False,
        },
        "PartitionKeys": STANDARD_PARTITIONS,
        "TableType": "EXTERNAL_TABLE",
        "Parameters": {
            "EXTERNAL": "TRUE",
            "parquet.compression": "SNAPPY",
            "classification": "parquet",
        },
    }


def create_or_update_table(
    table_name: str,
    columns: list[tuple[str, str]],
    s3_location: str,
    database_name: str = GLUE_DATABASE_NAME,
    glue_client: Any | None = None,
) -> dict[str, str]:
    """Create or update a Glue external table.

    If the table already exists it is updated in place (schema sync).
    This operation is idempotent.

    Parameters
    ----------
    table_name:
        Glue table name.
    columns:
        Column definitions as (name, type) tuples.
    s3_location:
        S3 URI for the table root.
    database_name:
        Glue database name.
    glue_client:
        Injected client for testing.

    Returns
    -------
    dict[str, str]
        Result with table_name and status.
    """

    client = (
        glue_client
        if glue_client is not None
        else boto3.client(
            "glue",
            region_name=AWS_REGION,
        )
    )

    table_input = build_table_input(
        table_name=table_name,
        columns=columns,
        s3_location=s3_location,
    )

    try:
        client.get_table(
            DatabaseName=database_name,
            Name=table_name,
        )

        client.update_table(
            DatabaseName=database_name,
            TableInput=table_input,
        )

        return {
            "table_name": table_name,
            "status": "updated",
        }

    except client.exceptions.EntityNotFoundException:
        client.create_table(
            DatabaseName=database_name,
            TableInput=table_input,
        )

        return {
            "table_name": table_name,
            "status": "created",
        }


def add_partition(
    table_name: str,
    model_version: str,
    report_date: str,
    database_name: str = GLUE_DATABASE_NAME,
    bucket: str = S3_BUCKET,
    analytics_prefix: str = S3_ANALYTICS_PREFIX,
    glue_client: Any | None = None,
) -> dict[str, str]:
    """Register a Hive-style partition in the Glue catalog.

    Parameters
    ----------
    table_name:
        Glue table name.
    model_version:
        Partition value for model_version.
    report_date:
        Partition value for report_date (YYYY-MM-DD).
    database_name:
        Glue database name.
    bucket:
        S3 bucket.
    analytics_prefix:
        S3 analytics prefix.
    glue_client:
        Injected client for testing.

    Returns
    -------
    dict[str, str]
        Result with table_name, partition, and status.
    """

    client = (
        glue_client
        if glue_client is not None
        else boto3.client(
            "glue",
            region_name=AWS_REGION,
        )
    )

    prefix = analytics_prefix.strip(
        "/ "
    )

    partition_location = (
        f"s3://{bucket}/{prefix}/{table_name}/"
        f"model_version={model_version}/"
        f"report_date={report_date}/"
    )

    partition_input = {
        "Values": [
            model_version,
            report_date,
        ],
        "StorageDescriptor": {
            "Location": partition_location,
            "InputFormat": (
                "org.apache.hadoop.mapred.TextInputFormat"
            ),
            "OutputFormat": (
                "org.apache.hadoop.hive.ql.io"
                ".HiveIgnoreKeyTextOutputFormat"
            ),
            "SerdeInfo": {
                "SerializationLibrary": (
                    "org.apache.hadoop.hive.ql.io.parquet.serde"
                    ".ParquetHiveSerDe"
                ),
            },
        },
    }

    try:
        client.batch_create_partition(
            DatabaseName=database_name,
            TableName=table_name,
            PartitionInputList=[
                partition_input
            ],
        )

        return {
            "table_name": table_name,
            "partition": f"model_version={model_version}/report_date={report_date}",
            "status": "created",
        }

    except client.exceptions.AlreadyExistsException:
        return {
            "table_name": table_name,
            "partition": f"model_version={model_version}/report_date={report_date}",
            "status": "already_exists",
        }


def create_all_tables(
    database_name: str = GLUE_DATABASE_NAME,
    bucket: str = S3_BUCKET,
    analytics_prefix: str = S3_ANALYTICS_PREFIX,
    glue_client: Any | None = None,
) -> dict[str, object]:
    """Create or update all monitoring Glue tables.

    Parameters
    ----------
    database_name:
        Glue database name.
    bucket:
        S3 bucket for table locations.
    analytics_prefix:
        S3 prefix for analytical datasets.
    glue_client:
        Injected client for testing.

    Returns
    -------
    dict[str, object]
        Summary with database_name, table_count, and table_results.
    """

    client = (
        glue_client
        if glue_client is not None
        else boto3.client(
            "glue",
            region_name=AWS_REGION,
        )
    )

    db_result = create_glue_database(
        database_name=database_name,
        glue_client=client,
    )

    prefix = analytics_prefix.strip(
        "/ "
    )

    table_results: dict[str, dict[str, str]] = {}

    for table_name, (columns, s3_suffix) in TABLE_REGISTRY.items():
        s3_location = (
            f"s3://{bucket}/{prefix}/{s3_suffix}/"
        )

        table_results[
            table_name
        ] = create_or_update_table(
            table_name=table_name,
            columns=columns,
            s3_location=s3_location,
            database_name=database_name,
            glue_client=client,
        )

    return {
        "database_name": database_name,
        "database_status": db_result[
            "status"
        ],
        "table_count": len(
            table_results
        ),
        "table_results": table_results,
    }
