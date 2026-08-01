"""Amazon QuickSight asset definitions for credit risk monitoring.

Provides functions to create or update QuickSight data sources,
datasets, and dashboard definitions that connect to the Athena/Glue
monitoring tables.

IMPORTANT DEPLOYMENT NOTES
---------------------------
QuickSight resource creation requires an AWS account with QuickSight
Enterprise or Standard edition enabled, with appropriate IAM
permissions and a QuickSight namespace. Account IDs, namespace names,
and permission ARNs vary by AWS account and must be supplied as
parameters or environment variables.

QuickSight creation is therefore parameterized — this module provides
the asset specifications and helper functions. Actual creation requires
valid AWS credentials and a configured QuickSight account.

No HTML dashboards are created. All dashboards use QuickSight's native
visual layer backed by Athena SQL over S3 Parquet.
"""

from __future__ import annotations

from typing import Any

import boto3

from app.config import (
    ATHENA_WORKGROUP,
    AWS_ACCOUNT_ID,
    AWS_REGION,
    GLUE_DATABASE_NAME,
    QUICKSIGHT_DATA_SET_PREFIX,
    QUICKSIGHT_DATA_SOURCE_ID,
)


# ---------------------------------------------------------------------------
# Data source definition
# ---------------------------------------------------------------------------

def build_athena_data_source_input(
    data_source_id: str = QUICKSIGHT_DATA_SOURCE_ID,
    database_name: str = GLUE_DATABASE_NAME,
    workgroup: str = ATHENA_WORKGROUP,
) -> dict[str, Any]:
    """Build the QuickSight Athena data source parameters.

    Parameters
    ----------
    data_source_id:
        Unique identifier for the QuickSight data source.
    database_name:
        Glue database name (Athena catalog).
    workgroup:
        Athena workgroup name.

    Returns
    -------
    dict[str, Any]
        Data source parameter dict for CreateDataSource /
        UpdateDataSource.
    """

    return {
        "DataSourceId": data_source_id,
        "Name": "CreditRiskAthenaDataSource",
        "Type": "ATHENA",
        "DataSourceParameters": {
            "AthenaParameters": {
                "WorkGroup": workgroup,
                "RoleArn": "",  # Set by caller from IAM config
            }
        },
        "SslProperties": {
            "DisableSsl": False,
        },
    }


# ---------------------------------------------------------------------------
# Dataset definitions
# ---------------------------------------------------------------------------

# Map of QuickSight dataset name -> SQL query
DATASET_QUERIES: dict[str, str] = {
    "performance_metrics": (
        "SELECT * FROM {database}.performance_metrics"
    ),
    "calibration_metrics": (
        "SELECT * FROM {database}.calibration_metrics"
    ),
    "psi_metrics": (
        "SELECT * FROM {database}.psi_metrics"
    ),
    "segment_metrics": (
        "SELECT * FROM {database}.segment_metrics"
    ),
    "adverse_reason_summary": (
        "SELECT * FROM {database}.adverse_reason_summary"
    ),
    "pipeline_runs": (
        "SELECT * FROM {database}.pipeline_runs"
    ),
}


def build_dataset_input(
    dataset_suffix: str,
    sql_query: str,
    data_source_id: str = QUICKSIGHT_DATA_SOURCE_ID,
    dataset_prefix: str = QUICKSIGHT_DATA_SET_PREFIX,
    database_name: str = GLUE_DATABASE_NAME,
    import_mode: str = "DIRECT_QUERY",
) -> dict[str, Any]:
    """Build a QuickSight dataset input for CreateDataSet / UpdateDataSet.

    Parameters
    ----------
    dataset_suffix:
        Suffix appended to dataset_prefix to form the dataset ID.
    sql_query:
        SQL query template. Use {database} as a placeholder for the
        Glue database name.
    data_source_id:
        QuickSight data source ID to use.
    dataset_prefix:
        Prefix for dataset IDs and names.
    database_name:
        Glue database name for SQL template substitution.
    import_mode:
        DIRECT_QUERY or SPICE.

    Returns
    -------
    dict[str, Any]
        Dataset parameter dict for CreateDataSet / UpdateDataSet.
    """

    dataset_id = f"{dataset_prefix}-{dataset_suffix}"
    dataset_name = f"{dataset_prefix}-{dataset_suffix}".replace(
        "-",
        " ",
    ).title()

    resolved_sql = sql_query.format(
        database=database_name
    )

    return {
        "DataSetId": dataset_id,
        "Name": dataset_name,
        "ImportMode": import_mode,
        "PhysicalTableMap": {
            f"{dataset_suffix}_physical": {
                "CustomSql": {
                    "DataSourceArn": data_source_id,
                    "Name": dataset_suffix,
                    "SqlQuery": resolved_sql,
                    "Columns": [],  # Auto-inferred from SQL
                }
            }
        },
    }


def build_all_dataset_inputs(
    data_source_id: str = QUICKSIGHT_DATA_SOURCE_ID,
    dataset_prefix: str = QUICKSIGHT_DATA_SET_PREFIX,
    database_name: str = GLUE_DATABASE_NAME,
    import_mode: str = "DIRECT_QUERY",
) -> list[dict[str, Any]]:
    """Build dataset inputs for all monitoring tables.

    Parameters
    ----------
    data_source_id:
        QuickSight Athena data source ID.
    dataset_prefix:
        Prefix for dataset IDs.
    database_name:
        Glue database name.
    import_mode:
        DIRECT_QUERY or SPICE.

    Returns
    -------
    list[dict[str, Any]]
        List of dataset inputs for all monitoring tables.
    """

    return [
        build_dataset_input(
            dataset_suffix=suffix,
            sql_query=query,
            data_source_id=data_source_id,
            dataset_prefix=dataset_prefix,
            database_name=database_name,
            import_mode=import_mode,
        )
        for suffix, query in DATASET_QUERIES.items()
    ]


# ---------------------------------------------------------------------------
# Dashboard specifications (documented definitions)
# ---------------------------------------------------------------------------

DASHBOARD_SPECS: dict[str, dict[str, Any]] = {
    "executive_model_health": {
        "name": "Credit Risk - Executive Model Health",
        "description": (
            "Top-level KPIs: AUC, KS, Gini, bad rate, average predicted "
            "PD, ECE, MCE, maximum PSI, and record count. "
            "Filters: model_version, report_date."
        ),
        "primary_dataset": "performance_metrics",
        "visuals": [
            "AUC KPI",
            "KS KPI",
            "Gini KPI",
            "BadRate KPI",
            "AveragePredictedPD KPI",
            "ECE KPI",
            "MCE KPI",
            "MaxPSI KPI",
            "RecordCount KPI",
            "AUC trend line chart by report_date",
            "KS trend line chart by report_date",
        ],
        "filters": [
            "model_version (filter)",
            "report_date (date range)",
        ],
    },
    "calibration": {
        "name": "Credit Risk - Calibration",
        "description": (
            "Expected vs actual default rate by calibration band. "
            "ECE and MCE trend. Model version comparison."
        ),
        "primary_dataset": "calibration_metrics",
        "visuals": [
            "Bar chart: average_predicted_pd vs actual_default_rate "
            "by calibration_band",
            "Bar chart: calibration_gap by calibration_band",
            "Line chart: ECE over report_date",
            "Line chart: MCE over report_date",
            "Table: full calibration band detail",
        ],
        "filters": [
            "model_version (multi-select for version comparison)",
            "report_date (date range)",
        ],
        "governance_note": (
            "Calibration monitoring does not trigger automatic "
            "recalibration. All model changes require governance approval."
        ),
    },
    "drift": {
        "name": "Credit Risk - Drift",
        "description": (
            "PSI by feature. Stable, warning, and alert counts. "
            "PSI trend by feature."
        ),
        "primary_dataset": "psi_metrics",
        "visuals": [
            "Bar chart: psi by feature, coloured by status",
            "KPI: stable feature count",
            "KPI: warning feature count",
            "KPI: alert feature count",
            "Line chart: psi trend by feature over report_date",
            "Table: full PSI detail",
        ],
        "filters": [
            "model_version",
            "report_date (date range)",
            "status (stable / warning / alert)",
        ],
    },
    "segment_performance": {
        "name": "Credit Risk - Segment Performance",
        "description": (
            "Volume, bad rate, average PD, and calibration gap "
            "by segment. Supports score band, FICO band, loan amount, "
            "income band, and any configured segment."
        ),
        "primary_dataset": "segment_metrics",
        "visuals": [
            "Bar chart: record_count by segment_value",
            "Bar chart: bad_rate by segment_value",
            "Bar chart: average_predicted_pd by segment_value",
            "Bar chart: calibration_gap by segment_value",
            "Table: full segment detail",
        ],
        "filters": [
            "segment_name (dropdown)",
            "model_version",
            "report_date (date range)",
        ],
    },
    "adverse_reasons": {
        "name": "Credit Risk - Adverse Reasons",
        "description": (
            "Adverse reason frequency, selection rate, average SHAP "
            "contribution, trend by date, and segment comparison."
        ),
        "primary_dataset": "adverse_reason_summary",
        "visuals": [
            "Bar chart: selection_count by reason_code",
            "Bar chart: selection_rate by reason_code",
            "Bar chart: average_contribution by reason_code",
            "Line chart: selection_count trend by reason_code over report_date",
            "Table: full adverse reason detail",
        ],
        "filters": [
            "model_version",
            "report_date (date range)",
        ],
        "compliance_note": (
            "Adverse action reason wording is demonstration content. "
            "All adverse action notices require legal and compliance "
            "approval before use in production."
        ),
    },
    "governance": {
        "name": "Credit Risk - Governance",
        "description": (
            "Model version, governance status, approval reference, "
            "approval date, deployment date, and threshold breaches."
        ),
        "primary_dataset": "pipeline_runs",
        "visuals": [
            "Table: pipeline run history with status",
            "KPI: latest run status",
            "Table: threshold breach events from CloudWatch alarms",
        ],
        "filters": [
            "model_version",
            "report_date (date range)",
            "status (success / failed)",
        ],
        "governance_note": (
            "The monitoring pipeline does not change governance status "
            "automatically. Model version changes, threshold updates, and "
            "governance status transitions require explicit approval and "
            "a controlled deployment process."
        ),
    },
}


def create_athena_data_source(
    account_id: str = AWS_ACCOUNT_ID,
    data_source_id: str = QUICKSIGHT_DATA_SOURCE_ID,
    database_name: str = GLUE_DATABASE_NAME,
    workgroup: str = ATHENA_WORKGROUP,
    principal_arn: str | None = None,
    quicksight_client: Any | None = None,
) -> dict[str, Any]:
    """Create or update the QuickSight Athena data source.

    Parameters
    ----------
    account_id:
        AWS account ID.
    data_source_id:
        Unique data source identifier.
    database_name:
        Glue database name.
    workgroup:
        Athena workgroup.
    principal_arn:
        IAM ARN of the QuickSight principal (user or group).
    quicksight_client:
        Injected client for testing.

    Returns
    -------
    dict[str, Any]
        API response or result summary.
    """

    if not account_id:
        raise ValueError(
            "AWS_ACCOUNT_ID must be set to create QuickSight resources. "
            "Set the AWS_ACCOUNT_ID environment variable."
        )

    client = (
        quicksight_client
        if quicksight_client is not None
        else boto3.client(
            "quicksight",
            region_name=AWS_REGION,
        )
    )

    params: dict[str, Any] = {
        "AwsAccountId": account_id,
        "DataSourceId": data_source_id,
        "Name": "CreditRiskAthenaDataSource",
        "Type": "ATHENA",
        "DataSourceParameters": {
            "AthenaParameters": {
                "WorkGroup": workgroup,
            }
        },
        "SslProperties": {
            "DisableSsl": False,
        },
    }

    if principal_arn:
        params[
            "Permissions"
        ] = [
            {
                "Principal": principal_arn,
                "Actions": [
                    "quicksight:DescribeDataSource",
                    "quicksight:DescribeDataSourcePermissions",
                    "quicksight:PassDataSource",
                    "quicksight:UpdateDataSource",
                    "quicksight:DeleteDataSource",
                    "quicksight:UpdateDataSourcePermissions",
                ],
            }
        ]

    try:
        response = client.create_data_source(
            **params
        )

        return {
            "data_source_id": data_source_id,
            "status": "created",
            "arn": response.get(
                "Arn",
                "",
            ),
        }

    except client.exceptions.ResourceExistsException:
        client.update_data_source(
            AwsAccountId=account_id,
            DataSourceId=data_source_id,
            Name="CreditRiskAthenaDataSource",
            DataSourceParameters={
                "AthenaParameters": {
                    "WorkGroup": workgroup,
                }
            },
        )

        return {
            "data_source_id": data_source_id,
            "status": "updated",
        }


def get_dashboard_specifications() -> dict[str, dict[str, Any]]:
    """Return the full QuickSight dashboard specification catalogue.

    Returns the documented dashboard specs for all six recommended
    dashboards. These specifications describe the recommended visuals,
    filters, and datasets. Actual QuickSight dashboard creation
    requires the QuickSight API or console and varies by account.

    Returns
    -------
    dict[str, dict[str, Any]]
        Dashboard specifications indexed by dashboard key.
    """

    return DASHBOARD_SPECS
