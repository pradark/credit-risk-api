import logging
import boto3
import watchtower

from app.config import AWS_REGION


logger = logging.getLogger("credit-risk-api")


def setup_logging():

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s "
        "%(levelname)s "
        "%(name)s "
        "%(message)s"
    )


    # Console logging
    console_handler = logging.StreamHandler()

    console_handler.setFormatter(
        formatter
    )


    logger.addHandler(
        console_handler
    )


    # AWS CloudWatch logging
    cloudwatch_handler = watchtower.CloudWatchLogHandler(
        log_group="credit-risk-api",
        stream_name="api",
        boto3_client=boto3.client(
            "logs",
            region_name=AWS_REGION
        )
    )


    cloudwatch_handler.setFormatter(
        formatter
    )


    logger.addHandler(
        cloudwatch_handler
    )

