import boto3
import logging

from botocore.exceptions import NoCredentialsError, ClientError

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


    except NoCredentialsError:

        logger.warning(
            f"Skipping CloudWatch metric {name}: AWS credentials not available"
        )


    except ClientError as e:

        logger.error(
            f"CloudWatch API error for {name}: {e}"
        )


    except Exception as e:

        logger.exception(
            f"Unexpected CloudWatch metric failure for {name}: {e}"
        )