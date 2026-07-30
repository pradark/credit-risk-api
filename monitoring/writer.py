import boto3
import pandas as pd

from io import BytesIO
from datetime import datetime
import logging

from app.config import (
    S3_BUCKET,
    AWS_REGION,
    S3_MONITORING_PREFIX
)


logger = logging.getLogger(__name__)


s3 = boto3.client(
    "s3",
    region_name=AWS_REGION
)



def write_metrics(metrics: dict) -> str:
    """
    Write monitoring metrics to S3 as parquet.

    Returns:
        S3 object key
    """


    df = pd.DataFrame(
        [metrics]
    )


    buffer = BytesIO()


    df.to_parquet(
        buffer,
        index=False
    )


    timestamp = datetime.utcnow().strftime(
        "%Y-%m-%d"
    )


    key = (
        f"{S3_MONITORING_PREFIX}/"
        f"metrics_{timestamp}.parquet"
    )


    s3.put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=buffer.getvalue()
    )


    logger.info(
        f"Monitoring metrics written to {key}"
    )


    return key