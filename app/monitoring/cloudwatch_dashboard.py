"""CloudWatch dashboard and alarm definitions for credit risk monitoring.

Provides functions to create or update the CloudWatch monitoring
dashboard and configure metric alarms. All alarm thresholds are
configurable through environment variables.

Alarms must never trigger automatic model changes, retraining,
recalibration, or threshold updates. They are evidence-gathering and
alerting mechanisms only.
"""

from __future__ import annotations

import json
from typing import Any

import boto3

from app.config import (
    AUC_ALERT_THRESHOLD,
    AUC_WARNING_THRESHOLD,
    AWS_REGION,
    CALIBRATION_ALERT_THRESHOLD,
    CALIBRATION_WARNING_THRESHOLD,
    CLOUDWATCH_MONITORING_NAMESPACE,
    CLOUDWATCH_RUNTIME_NAMESPACE,
    ENVIRONMENT,
    KS_ALERT_THRESHOLD,
    KS_WARNING_THRESHOLD,
    MODEL_VERSION,
    PSI_ALERT_THRESHOLD,
    PSI_WARNING_THRESHOLD,
)


DASHBOARD_NAME = "CreditRiskModelMonitoring"


def _model_metric(
    metric_name: str,
    model_version: str,
    environment: str,
    namespace: str = CLOUDWATCH_MONITORING_NAMESPACE,
) -> list[Any]:
    """Build a CloudWatch metrics widget metric entry."""

    return [
        namespace,
        metric_name,
        "ModelVersion",
        model_version,
        "Environment",
        environment,
    ]


def _runtime_metric(
    metric_name: str,
    namespace: str = CLOUDWATCH_RUNTIME_NAMESPACE,
) -> list[Any]:
    """Build a CloudWatch runtime widget metric entry."""

    return [
        namespace,
        metric_name,
    ]


def build_dashboard_body(
    model_version: str = MODEL_VERSION,
    environment: str = ENVIRONMENT,
    region: str = AWS_REGION,
) -> str:
    """Build the CloudWatch dashboard JSON body.

    The dashboard includes:
    - AUC, KS, Gini trend
    - Bad rate vs average predicted PD
    - ECE and MCE trend
    - Maximum PSI and feature count widgets
    - Prediction volume and default count
    - Prediction and explanation error rates
    - Pipeline failure and duration

    Parameters
    ----------
    model_version:
        Model version for dimension filtering.
    environment:
        Environment for dimension filtering.
    region:
        AWS region for CloudWatch metrics.

    Returns
    -------
    str
        JSON dashboard body suitable for PutDashboard.
    """

    def metric(name: str) -> list[Any]:
        return _model_metric(
            name,
            model_version,
            environment,
        )

    def rt_metric(name: str) -> list[Any]:
        return _runtime_metric(
            name
        )

    widgets = [
        # Row 1: Discriminatory power
        {
            "type": "metric",
            "x": 0,
            "y": 0,
            "width": 8,
            "height": 6,
            "properties": {
                "title": "AUC Trend",
                "metrics": [
                    metric("AUC"),
                ],
                "view": "timeSeries",
                "stat": "Average",
                "period": 86400,
                "yAxis": {
                    "left": {
                        "min": 0,
                        "max": 1,
                    }
                },
                "annotations": {
                    "horizontal": [
                        {
                            "label": "Warning",
                            "value": AUC_WARNING_THRESHOLD,
                            "color": "#ff7f0e",
                        },
                        {
                            "label": "Alert",
                            "value": AUC_ALERT_THRESHOLD,
                            "color": "#d62728",
                        },
                    ]
                },
            },
        },
        {
            "type": "metric",
            "x": 8,
            "y": 0,
            "width": 8,
            "height": 6,
            "properties": {
                "title": "KS Trend",
                "metrics": [
                    metric("KS"),
                ],
                "view": "timeSeries",
                "stat": "Average",
                "period": 86400,
                "yAxis": {
                    "left": {
                        "min": 0,
                        "max": 1,
                    }
                },
                "annotations": {
                    "horizontal": [
                        {
                            "label": "Warning",
                            "value": KS_WARNING_THRESHOLD,
                            "color": "#ff7f0e",
                        },
                        {
                            "label": "Alert",
                            "value": KS_ALERT_THRESHOLD,
                            "color": "#d62728",
                        },
                    ]
                },
            },
        },
        {
            "type": "metric",
            "x": 16,
            "y": 0,
            "width": 8,
            "height": 6,
            "properties": {
                "title": "Gini Trend",
                "metrics": [
                    metric("Gini"),
                ],
                "view": "timeSeries",
                "stat": "Average",
                "period": 86400,
                "yAxis": {
                    "left": {
                        "min": 0,
                        "max": 1,
                    }
                },
            },
        },
        # Row 2: Calibration
        {
            "type": "metric",
            "x": 0,
            "y": 6,
            "width": 8,
            "height": 6,
            "properties": {
                "title": "Bad Rate vs Average Predicted PD",
                "metrics": [
                    metric("BadRate"),
                    metric("AveragePredictedPD"),
                ],
                "view": "timeSeries",
                "stat": "Average",
                "period": 86400,
            },
        },
        {
            "type": "metric",
            "x": 8,
            "y": 6,
            "width": 8,
            "height": 6,
            "properties": {
                "title": "Expected Calibration Error (ECE)",
                "metrics": [
                    metric("ExpectedCalibrationError"),
                ],
                "view": "timeSeries",
                "stat": "Average",
                "period": 86400,
                "annotations": {
                    "horizontal": [
                        {
                            "label": "Warning",
                            "value": CALIBRATION_WARNING_THRESHOLD,
                            "color": "#ff7f0e",
                        },
                        {
                            "label": "Alert",
                            "value": CALIBRATION_ALERT_THRESHOLD,
                            "color": "#d62728",
                        },
                    ]
                },
            },
        },
        {
            "type": "metric",
            "x": 16,
            "y": 6,
            "width": 8,
            "height": 6,
            "properties": {
                "title": "Maximum Calibration Error (MCE)",
                "metrics": [
                    metric("MaximumCalibrationError"),
                ],
                "view": "timeSeries",
                "stat": "Average",
                "period": 86400,
            },
        },
        # Row 3: Drift
        {
            "type": "metric",
            "x": 0,
            "y": 12,
            "width": 8,
            "height": 6,
            "properties": {
                "title": "Maximum PSI",
                "metrics": [
                    metric("MaximumPSI"),
                ],
                "view": "timeSeries",
                "stat": "Maximum",
                "period": 86400,
                "annotations": {
                    "horizontal": [
                        {
                            "label": "Warning",
                            "value": PSI_WARNING_THRESHOLD,
                            "color": "#ff7f0e",
                        },
                        {
                            "label": "Alert",
                            "value": PSI_ALERT_THRESHOLD,
                            "color": "#d62728",
                        },
                    ]
                },
            },
        },
        {
            "type": "metric",
            "x": 8,
            "y": 12,
            "width": 8,
            "height": 6,
            "properties": {
                "title": "Feature Drift Counts",
                "metrics": [
                    metric("WarningFeatureCount"),
                    metric("AlertFeatureCount"),
                ],
                "view": "timeSeries",
                "stat": "Maximum",
                "period": 86400,
            },
        },
        # Row 4: Volume
        {
            "type": "metric",
            "x": 16,
            "y": 12,
            "width": 8,
            "height": 6,
            "properties": {
                "title": "Prediction Volume",
                "metrics": [
                    rt_metric("PredictionCount"),
                ],
                "view": "timeSeries",
                "stat": "Sum",
                "period": 86400,
            },
        },
        {
            "type": "metric",
            "x": 0,
            "y": 18,
            "width": 8,
            "height": 6,
            "properties": {
                "title": "Default Count",
                "metrics": [
                    metric("DefaultCount"),
                ],
                "view": "timeSeries",
                "stat": "Sum",
                "period": 86400,
            },
        },
        # Row 5: Errors and pipeline
        {
            "type": "metric",
            "x": 8,
            "y": 18,
            "width": 8,
            "height": 6,
            "properties": {
                "title": "Prediction & Explanation Errors",
                "metrics": [
                    rt_metric("PredictionErrorCount"),
                    rt_metric("ExplanationErrorCount"),
                ],
                "view": "timeSeries",
                "stat": "Sum",
                "period": 3600,
            },
        },
        {
            "type": "metric",
            "x": 16,
            "y": 18,
            "width": 8,
            "height": 6,
            "properties": {
                "title": "Pipeline Failures",
                "metrics": [
                    metric("PipelineFailure"),
                ],
                "view": "timeSeries",
                "stat": "Sum",
                "period": 86400,
            },
        },
    ]

    return json.dumps(
        {
            "widgets": widgets,
        }
    )


def put_dashboard(
    dashboard_name: str = DASHBOARD_NAME,
    model_version: str = MODEL_VERSION,
    environment: str = ENVIRONMENT,
    region: str = AWS_REGION,
    cloudwatch_client: Any | None = None,
) -> dict[str, str]:
    """Create or update the CloudWatch monitoring dashboard.

    Parameters
    ----------
    dashboard_name:
        Name of the CloudWatch dashboard.
    model_version:
        Model version for widget dimension filtering.
    environment:
        Environment for widget dimension filtering.
    region:
        AWS region.
    cloudwatch_client:
        Injected client for testing.

    Returns
    -------
    dict[str, str]
        Result with dashboard_name and status.
    """

    client = (
        cloudwatch_client
        if cloudwatch_client is not None
        else boto3.client(
            "cloudwatch",
            region_name=region,
        )
    )

    body = build_dashboard_body(
        model_version=model_version,
        environment=environment,
        region=region,
    )

    client.put_dashboard(
        DashboardName=dashboard_name,
        DashboardBody=body,
    )

    return {
        "dashboard_name": dashboard_name,
        "status": "created_or_updated",
    }


# ---------------------------------------------------------------------------
# Alarm definitions
# ---------------------------------------------------------------------------

def build_alarm_definitions(
    model_version: str = MODEL_VERSION,
    environment: str = ENVIRONMENT,
    sns_topic_arn: str | None = None,
) -> list[dict[str, Any]]:
    """Build CloudWatch alarm configurations.

    Returns a list of alarm parameter dicts ready to pass to
    put_metric_alarm. Alarms must never trigger automatic model
    changes.

    Parameters
    ----------
    model_version:
        Dimension value for ModelVersion.
    environment:
        Dimension value for Environment.
    sns_topic_arn:
        Optional SNS topic ARN for alarm notifications.

    Returns
    -------
    list[dict[str, Any]]
        Alarm parameter dictionaries.
    """

    alarm_actions = (
        [
            sns_topic_arn,
        ]
        if sns_topic_arn
        else []
    )

    dimensions = [
        {
            "Name": "ModelVersion",
            "Value": model_version,
        },
        {
            "Name": "Environment",
            "Value": environment,
        },
    ]

    monitoring_ns = CLOUDWATCH_MONITORING_NAMESPACE
    runtime_ns = CLOUDWATCH_RUNTIME_NAMESPACE

    alarms = [
        # AUC alarms
        {
            "AlarmName": f"CreditRisk-AUC-Warning-{model_version}-{environment}",
            "AlarmDescription": (
                "AUC below warning threshold. "
                "Investigate model performance. "
                "Do not automatically retrain or redeploy."
            ),
            "Namespace": monitoring_ns,
            "MetricName": "AUC",
            "Dimensions": dimensions,
            "Period": 86400,
            "EvaluationPeriods": 1,
            "Threshold": AUC_WARNING_THRESHOLD,
            "ComparisonOperator": "LessThanThreshold",
            "Statistic": "Average",
            "TreatMissingData": "notBreaching",
            "AlarmActions": alarm_actions,
        },
        {
            "AlarmName": f"CreditRisk-AUC-Alert-{model_version}-{environment}",
            "AlarmDescription": (
                "AUC below alert threshold. "
                "Escalate for governance review. "
                "Do not automatically retrain or redeploy."
            ),
            "Namespace": monitoring_ns,
            "MetricName": "AUC",
            "Dimensions": dimensions,
            "Period": 86400,
            "EvaluationPeriods": 1,
            "Threshold": AUC_ALERT_THRESHOLD,
            "ComparisonOperator": "LessThanThreshold",
            "Statistic": "Average",
            "TreatMissingData": "notBreaching",
            "AlarmActions": alarm_actions,
        },
        # KS alarms
        {
            "AlarmName": f"CreditRisk-KS-Warning-{model_version}-{environment}",
            "AlarmDescription": (
                "KS below warning threshold. "
                "Review discriminatory power."
            ),
            "Namespace": monitoring_ns,
            "MetricName": "KS",
            "Dimensions": dimensions,
            "Period": 86400,
            "EvaluationPeriods": 1,
            "Threshold": KS_WARNING_THRESHOLD,
            "ComparisonOperator": "LessThanThreshold",
            "Statistic": "Average",
            "TreatMissingData": "notBreaching",
            "AlarmActions": alarm_actions,
        },
        {
            "AlarmName": f"CreditRisk-KS-Alert-{model_version}-{environment}",
            "AlarmDescription": (
                "KS below alert threshold. "
                "Escalate for governance review."
            ),
            "Namespace": monitoring_ns,
            "MetricName": "KS",
            "Dimensions": dimensions,
            "Period": 86400,
            "EvaluationPeriods": 1,
            "Threshold": KS_ALERT_THRESHOLD,
            "ComparisonOperator": "LessThanThreshold",
            "Statistic": "Average",
            "TreatMissingData": "notBreaching",
            "AlarmActions": alarm_actions,
        },
        # ECE alarms
        {
            "AlarmName": f"CreditRisk-ECE-Warning-{model_version}-{environment}",
            "AlarmDescription": (
                "Expected Calibration Error above warning threshold. "
                "Review calibration. "
                "Do not automatically recalibrate."
            ),
            "Namespace": monitoring_ns,
            "MetricName": "ExpectedCalibrationError",
            "Dimensions": dimensions,
            "Period": 86400,
            "EvaluationPeriods": 1,
            "Threshold": CALIBRATION_WARNING_THRESHOLD,
            "ComparisonOperator": "GreaterThanThreshold",
            "Statistic": "Average",
            "TreatMissingData": "notBreaching",
            "AlarmActions": alarm_actions,
        },
        {
            "AlarmName": f"CreditRisk-ECE-Alert-{model_version}-{environment}",
            "AlarmDescription": (
                "Expected Calibration Error above alert threshold. "
                "Escalate for governance review. "
                "Do not automatically recalibrate."
            ),
            "Namespace": monitoring_ns,
            "MetricName": "ExpectedCalibrationError",
            "Dimensions": dimensions,
            "Period": 86400,
            "EvaluationPeriods": 1,
            "Threshold": CALIBRATION_ALERT_THRESHOLD,
            "ComparisonOperator": "GreaterThanThreshold",
            "Statistic": "Average",
            "TreatMissingData": "notBreaching",
            "AlarmActions": alarm_actions,
        },
        # MCE alarm
        {
            "AlarmName": f"CreditRisk-MCE-Alert-{model_version}-{environment}",
            "AlarmDescription": (
                "Maximum Calibration Error above threshold. "
                "Investigate worst-performing score band."
            ),
            "Namespace": monitoring_ns,
            "MetricName": "MaximumCalibrationError",
            "Dimensions": dimensions,
            "Period": 86400,
            "EvaluationPeriods": 1,
            "Threshold": CALIBRATION_ALERT_THRESHOLD,
            "ComparisonOperator": "GreaterThanThreshold",
            "Statistic": "Maximum",
            "TreatMissingData": "notBreaching",
            "AlarmActions": alarm_actions,
        },
        # PSI alarms
        {
            "AlarmName": f"CreditRisk-PSI-Warning-{model_version}-{environment}",
            "AlarmDescription": (
                "Maximum PSI above warning threshold. "
                "Feature drift detected."
            ),
            "Namespace": monitoring_ns,
            "MetricName": "MaximumPSI",
            "Dimensions": dimensions,
            "Period": 86400,
            "EvaluationPeriods": 1,
            "Threshold": PSI_WARNING_THRESHOLD,
            "ComparisonOperator": "GreaterThanThreshold",
            "Statistic": "Maximum",
            "TreatMissingData": "notBreaching",
            "AlarmActions": alarm_actions,
        },
        {
            "AlarmName": f"CreditRisk-PSI-Alert-{model_version}-{environment}",
            "AlarmDescription": (
                "Maximum PSI above alert threshold. "
                "Significant feature drift. Escalate."
            ),
            "Namespace": monitoring_ns,
            "MetricName": "MaximumPSI",
            "Dimensions": dimensions,
            "Period": 86400,
            "EvaluationPeriods": 1,
            "Threshold": PSI_ALERT_THRESHOLD,
            "ComparisonOperator": "GreaterThanThreshold",
            "Statistic": "Maximum",
            "TreatMissingData": "notBreaching",
            "AlarmActions": alarm_actions,
        },
        # Prediction errors
        {
            "AlarmName": f"CreditRisk-PredictionErrors-{environment}",
            "AlarmDescription": (
                "Prediction error rate elevated. "
                "Review API logs."
            ),
            "Namespace": runtime_ns,
            "MetricName": "PredictionErrorCount",
            "Period": 3600,
            "EvaluationPeriods": 1,
            "Threshold": 10.0,
            "ComparisonOperator": "GreaterThanThreshold",
            "Statistic": "Sum",
            "TreatMissingData": "notBreaching",
            "AlarmActions": alarm_actions,
        },
        # Pipeline failure
        {
            "AlarmName": f"CreditRisk-PipelineFailure-{model_version}-{environment}",
            "AlarmDescription": (
                "Monitoring pipeline failure detected. "
                "Review pipeline logs."
            ),
            "Namespace": monitoring_ns,
            "MetricName": "PipelineFailure",
            "Dimensions": dimensions,
            "Period": 86400,
            "EvaluationPeriods": 1,
            "Threshold": 0,
            "ComparisonOperator": "GreaterThanThreshold",
            "Statistic": "Sum",
            "TreatMissingData": "notBreaching",
            "AlarmActions": alarm_actions,
        },
    ]

    return alarms


def put_alarms(
    model_version: str = MODEL_VERSION,
    environment: str = ENVIRONMENT,
    sns_topic_arn: str | None = None,
    cloudwatch_client: Any | None = None,
) -> dict[str, object]:
    """Create or update all CloudWatch monitoring alarms.

    Parameters
    ----------
    model_version:
        Model version for alarm dimensions.
    environment:
        Environment for alarm dimensions.
    sns_topic_arn:
        Optional SNS topic ARN for notifications.
    cloudwatch_client:
        Injected client for testing.

    Returns
    -------
    dict[str, object]
        Summary with alarm_count and alarm_names.
    """

    client = (
        cloudwatch_client
        if cloudwatch_client is not None
        else boto3.client(
            "cloudwatch",
            region_name=AWS_REGION,
        )
    )

    alarm_definitions = build_alarm_definitions(
        model_version=model_version,
        environment=environment,
        sns_topic_arn=sns_topic_arn,
    )

    alarm_names = []

    for alarm in alarm_definitions:
        # Remove empty AlarmActions lists before calling AWS
        if (
            "AlarmActions" in alarm
            and not alarm[
                "AlarmActions"
            ]
        ):
            alarm_copy = {
                k: v
                for k, v in alarm.items()
                if k != "AlarmActions"
            }

        else:
            alarm_copy = alarm

        client.put_metric_alarm(
            **alarm_copy
        )

        alarm_names.append(
            alarm[
                "AlarmName"
            ]
        )

    return {
        "alarm_count": len(
            alarm_names
        ),
        "alarm_names": alarm_names,
    }
