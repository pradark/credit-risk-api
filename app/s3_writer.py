import boto3
import pandas as pd
import logging

from datetime import datetime, timezone
from io import BytesIO

from botocore.exceptions import NoCredentialsError, ClientError

from app.config import (
    S3_BUCKET,
    AWS_REGION
)


logger = logging.getLogger("credit-risk-api")


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


    try:

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


        logger.info(
            f"Successfully wrote parquet to S3: {key}"
        )


        return key


    except NoCredentialsError:

        logger.warning(
            f"Skipping S3 write {key}: AWS credentials not available"
        )

        return None


    except ClientError as e:

        logger.error(
            f"S3 error writing {key}: {e}"
        )

        return None


    except Exception as e:

        logger.exception(
            f"Unexpected S3 failure writing {key}: {e}"
        )

        return None