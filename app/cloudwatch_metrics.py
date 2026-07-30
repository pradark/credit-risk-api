import boto3
import logging

from app.config import AWS_REGION


logger = logging.getLogger("credit-risk-api")


cloudwatch = boto3.client(
    "cloudwatch",
    region_name=AWS_REGION
)


NAMESPACE = "CreditRiskAPI"


def put_metric(
    name: str,
    value: float,
    unit: str = "Count"
):
    """
    Publish a custom CloudWatch metric.

    Parameters:
        name: Metric name
        value: Metric value
        unit: CloudWatch unit type
    """

    try:

        cloudwatch.put_metric_data(
            Namespace=NAMESPACE,
            MetricData=[
                {
                    "MetricName": name,
                    "Value": value,
                    "Unit": unit
                }
            ]
        )

        logger.info(
            f"Published CloudWatch metric: {name}={value}"
        )

    except Exception as e:

        logger.error(
            f"Failed to publish CloudWatch metric {name}: {e}"
        )