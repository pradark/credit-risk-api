import boto3
import pandas as pd
import logging

from io import BytesIO

from app.config import (
    S3_BUCKET,
    AWS_REGION
)


logger = logging.getLogger(__name__)


s3 = boto3.client(
    "s3",
    region_name=AWS_REGION
)


def read_latest_parquet(prefix: str) -> pd.DataFrame:
    """
    Read the latest parquet file from an S3 prefix.

    Example:
        s3://bucket/features/
        s3://bucket/predictions/

    Returns:
        pandas DataFrame
    """

    logger.info(
        f"Searching for parquet files under s3://{S3_BUCKET}/{prefix}"
    )

    response = s3.list_objects_v2(
        Bucket=S3_BUCKET,
        Prefix=prefix
    )


    files = [
        obj["Key"]
        for obj in response.get("Contents", [])
        if obj["Key"].endswith(".parquet")
    ]


    if not files:
        raise FileNotFoundError(
            f"No parquet files found under prefix: {prefix}"
        )


    # Assumes filenames contain sortable timestamps
    latest_file = sorted(files)[-1]


    logger.info(
        f"Reading latest parquet file: {latest_file}"
    )


    obj = s3.get_object(
        Bucket=S3_BUCKET,
        Key=latest_file
    )


    df = pd.read_parquet(
        BytesIO(
            obj["Body"].read()
        )
    )


    logger.info(
        f"Loaded {len(df)} rows from {latest_file}"
    )


    return df



def read_features() -> pd.DataFrame:
    """
    Read latest feature dataset.
    """

    return read_latest_parquet(
        "features/"
    )



def read_predictions() -> pd.DataFrame:
    """
    Read latest prediction dataset.
    """

    return read_latest_parquet(
        "predictions/"
    )