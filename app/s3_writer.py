import boto3
import pandas as pd

from datetime import datetime, timezone
from io import BytesIO

from app.config import (
    S3_BUCKET,
    AWS_REGION
)


s3 = boto3.client(
    "s3",
    region_name=AWS_REGION
)


def write_parquet(
    data: dict,
    prefix: str
):

    today = datetime.now(
        timezone.utc
    ).strftime("%Y-%m-%d")


    key = (
        f"{prefix}/dt={today}/"
        f"data.parquet"
    )


    df = pd.DataFrame(
        [data]
    )


    buffer = BytesIO()


    df.to_parquet(
        buffer,
        index=False
    )


    buffer.seek(0)


    s3.put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=buffer.getvalue()
    )


    return key