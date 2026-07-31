import boto3
import pandas as pd
from io import BytesIO


s3 = boto3.client("s3")


def read_parquet_from_s3(
    bucket,
    key
):

    response = s3.get_object(
        Bucket=bucket,
        Key=key
    )

    return pd.read_parquet(
        BytesIO(
            response["Body"].read()
        )
    )


def write_parquet_to_s3(
    df,
    bucket,
    key
):

    buffer = BytesIO()

    df.to_parquet(
        buffer,
        index=False
    )

    buffer.seek(0)


    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=buffer
    )
