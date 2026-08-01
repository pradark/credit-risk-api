"""Amazon Athena query execution and validation utilities.

Provides functions for starting Athena queries, polling until
completion, and validating monitoring datasets with row-count and
latest-partition checks.

All functions accept an injected Athena client for unit testing
without live AWS calls.
"""

from __future__ import annotations

import time
from typing import Any

import boto3

from app.config import (
    ATHENA_OUTPUT_LOCATION,
    ATHENA_WORKGROUP,
    AWS_REGION,
    GLUE_DATABASE_NAME,
)


QUERY_POLL_INTERVAL_SECONDS = 2
QUERY_TIMEOUT_SECONDS = 300

TERMINAL_STATES = frozenset(
    {
        "SUCCEEDED",
        "FAILED",
        "CANCELLED",
    }
)


def start_query(
    query: str,
    database: str = GLUE_DATABASE_NAME,
    output_location: str = ATHENA_OUTPUT_LOCATION,
    workgroup: str = ATHENA_WORKGROUP,
    athena_client: Any | None = None,
) -> str:
    """Start an Athena query and return the execution ID.

    Parameters
    ----------
    query:
        SQL query string.
    database:
        Glue database name.
    output_location:
        S3 URI for Athena result output.
    workgroup:
        Athena workgroup name.
    athena_client:
        Injected client for testing.

    Returns
    -------
    str
        Athena query execution ID.
    """

    client = (
        athena_client
        if athena_client is not None
        else boto3.client(
            "athena",
            region_name=AWS_REGION,
        )
    )

    response = client.start_query_execution(
        QueryString=query,
        QueryExecutionContext={
            "Database": database,
        },
        ResultConfiguration={
            "OutputLocation": output_location,
        },
        WorkGroup=workgroup,
    )

    return response[
        "QueryExecutionId"
    ]


def poll_query_status(
    execution_id: str,
    poll_interval: float = QUERY_POLL_INTERVAL_SECONDS,
    timeout: float = QUERY_TIMEOUT_SECONDS,
    athena_client: Any | None = None,
) -> dict[str, Any]:
    """Poll an Athena query until it reaches a terminal state.

    Parameters
    ----------
    execution_id:
        Athena query execution ID.
    poll_interval:
        Seconds between status checks.
    timeout:
        Maximum seconds to wait before raising a TimeoutError.
    athena_client:
        Injected client for testing.

    Returns
    -------
    dict[str, Any]
        GetQueryExecution response for the completed query.

    Raises
    ------
    TimeoutError
        If the query does not complete within timeout seconds.
    RuntimeError
        If the query fails or is cancelled.
    """

    client = (
        athena_client
        if athena_client is not None
        else boto3.client(
            "athena",
            region_name=AWS_REGION,
        )
    )

    elapsed = 0.0

    while elapsed < timeout:
        response = client.get_query_execution(
            QueryExecutionId=execution_id
        )

        status = response[
            "QueryExecution"
        ][
            "Status"
        ][
            "State"
        ]

        if status == "SUCCEEDED":
            return response

        if status == "FAILED":
            reason = (
                response[
                    "QueryExecution"
                ][
                    "Status"
                ]
                .get(
                    "StateChangeReason",
                    "Unknown reason",
                )
            )

            raise RuntimeError(
                f"Athena query {execution_id} FAILED: {reason}"
            )

        if status == "CANCELLED":
            raise RuntimeError(
                f"Athena query {execution_id} was CANCELLED."
            )

        time.sleep(
            poll_interval
        )

        elapsed += poll_interval

    raise TimeoutError(
        f"Athena query {execution_id} did not complete "
        f"within {timeout} seconds."
    )


def run_query(
    query: str,
    database: str = GLUE_DATABASE_NAME,
    output_location: str = ATHENA_OUTPUT_LOCATION,
    workgroup: str = ATHENA_WORKGROUP,
    poll_interval: float = QUERY_POLL_INTERVAL_SECONDS,
    timeout: float = QUERY_TIMEOUT_SECONDS,
    athena_client: Any | None = None,
) -> dict[str, Any]:
    """Start a query and poll until completion.

    Convenience wrapper combining start_query and poll_query_status.

    Parameters
    ----------
    query:
        SQL query string.
    database:
        Glue database name.
    output_location:
        S3 URI for Athena result output.
    workgroup:
        Athena workgroup name.
    poll_interval:
        Seconds between status checks.
    timeout:
        Maximum seconds to wait.
    athena_client:
        Injected client for testing.

    Returns
    -------
    dict[str, Any]
        Completed GetQueryExecution response.
    """

    execution_id = start_query(
        query=query,
        database=database,
        output_location=output_location,
        workgroup=workgroup,
        athena_client=athena_client,
    )

    return poll_query_status(
        execution_id=execution_id,
        poll_interval=poll_interval,
        timeout=timeout,
        athena_client=athena_client,
    )


def validate_table_exists(
    table_name: str,
    database: str = GLUE_DATABASE_NAME,
    output_location: str = ATHENA_OUTPUT_LOCATION,
    workgroup: str = ATHENA_WORKGROUP,
    athena_client: Any | None = None,
) -> dict[str, object]:
    """Validate that a Glue/Athena table exists by querying it.

    Uses a LIMIT 1 query to confirm the table is reachable.

    Parameters
    ----------
    table_name:
        Name of the table to validate.
    database:
        Glue database name.
    output_location:
        S3 URI for Athena result output.
    workgroup:
        Athena workgroup name.
    athena_client:
        Injected client for testing.

    Returns
    -------
    dict[str, object]
        Result with table_name and exists (bool).
    """

    query = f"SELECT 1 FROM {table_name} LIMIT 1"

    try:
        run_query(
            query=query,
            database=database,
            output_location=output_location,
            workgroup=workgroup,
            athena_client=athena_client,
        )

        return {
            "table_name": table_name,
            "exists": True,
        }

    except Exception as exc:
        return {
            "table_name": table_name,
            "exists": False,
            "error": str(
                exc
            ),
        }


def run_row_count_check(
    table_name: str,
    database: str = GLUE_DATABASE_NAME,
    output_location: str = ATHENA_OUTPUT_LOCATION,
    workgroup: str = ATHENA_WORKGROUP,
    athena_client: Any | None = None,
) -> dict[str, object]:
    """Run a COUNT(*) query against a monitoring table.

    Parameters
    ----------
    table_name:
        Glue table name.
    database:
        Glue database name.
    output_location:
        S3 URI for Athena result output.
    workgroup:
        Athena workgroup name.
    athena_client:
        Injected client for testing.

    Returns
    -------
    dict[str, object]
        Result with table_name and execution_id.
    """

    query = f"SELECT COUNT(*) FROM {table_name}"

    execution_response = run_query(
        query=query,
        database=database,
        output_location=output_location,
        workgroup=workgroup,
        athena_client=athena_client,
    )

    return {
        "table_name": table_name,
        "execution_id": execution_response[
            "QueryExecution"
        ][
            "QueryExecutionId"
        ],
        "status": "succeeded",
    }


def run_latest_partition_check(
    table_name: str,
    database: str = GLUE_DATABASE_NAME,
    output_location: str = ATHENA_OUTPUT_LOCATION,
    workgroup: str = ATHENA_WORKGROUP,
    athena_client: Any | None = None,
) -> dict[str, object]:
    """Query the latest partition available in a monitoring table.

    Parameters
    ----------
    table_name:
        Glue table name.
    database:
        Glue database name.
    output_location:
        S3 URI for Athena result output.
    workgroup:
        Athena workgroup name.
    athena_client:
        Injected client for testing.

    Returns
    -------
    dict[str, object]
        Result with table_name, execution_id, and status.
    """

    query = (
        f"SELECT model_version, report_date "
        f"FROM {table_name} "
        f"ORDER BY report_date DESC "
        f"LIMIT 1"
    )

    execution_response = run_query(
        query=query,
        database=database,
        output_location=output_location,
        workgroup=workgroup,
        athena_client=athena_client,
    )

    return {
        "table_name": table_name,
        "execution_id": execution_response[
            "QueryExecution"
        ][
            "QueryExecutionId"
        ],
        "status": "succeeded",
    }


# ---------------------------------------------------------------------------
# Standard example queries (documented for QuickSight/Athena use)
# ---------------------------------------------------------------------------

EXAMPLE_QUERIES: dict[str, str] = {
    "performance_count": (
        "SELECT COUNT(*) FROM performance_metrics"
    ),
    "performance_trend": (
        "SELECT model_version, report_date, auc, ks, gini "
        "FROM performance_metrics "
        "ORDER BY report_date DESC"
    ),
    "calibration_by_band": (
        "SELECT calibration_band, average_predicted_pd, "
        "actual_default_rate "
        "FROM calibration_metrics "
        "WHERE model_version = '<version>' "
        "ORDER BY report_date DESC, calibration_band"
    ),
    "psi_latest": (
        "SELECT feature, psi, status "
        "FROM psi_metrics "
        "ORDER BY report_date DESC, psi DESC"
    ),
    "segment_performance": (
        "SELECT segment_name, segment_value, bad_rate, "
        "average_predicted_pd, calibration_gap "
        "FROM segment_metrics "
        "ORDER BY report_date DESC, segment_name, bad_rate DESC"
    ),
    "adverse_reason_frequency": (
        "SELECT reason_code, adverse_reason, selection_count, "
        "selection_rate, average_contribution "
        "FROM adverse_reason_summary "
        "ORDER BY report_date DESC, selection_count DESC"
    ),
    "pipeline_history": (
        "SELECT run_id, report_date, record_count, status, "
        "maximum_psi, warning_feature_count, alert_feature_count "
        "FROM pipeline_runs "
        "ORDER BY report_date DESC"
    ),
}
