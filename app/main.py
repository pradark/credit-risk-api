from fastapi import FastAPI, HTTPException

from app.schemas import CreditRequest
from app.model import predict_probability
from app.logging_config import setup_logging, logger
from app.metadata import load_metadata
from app.s3_writer import write_parquet
from app.cloudwatch_metrics import put_metric


setup_logging()


app = FastAPI(
    title="Credit Risk Prediction API"
)


@app.get("/health")
def health():

    logger.info(
        "Health check requested"
    )

    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(request: CreditRequest):

    logger.info(
        "Prediction request received"
    )

    try:

        features = request.model_dump()


        probability = predict_probability(
            features
        )


        # CloudWatch Metrics

        put_metric(
            "PredictionCount",
            1
        )


        put_metric(
            "DefaultProbability",
            probability,
            "None"
        )


        if probability >= 0.75:

            put_metric(
                "HighRiskPredictionCount",
                1
            )


        # Write features to S3

        write_parquet(
            features,
            "features"
        )


        # Write predictions to S3

        prediction_record = {
            **features,
            "default_probability": probability
        }


        write_parquet(
            prediction_record,
            "predictions"
        )


        logger.info(
            f"Prediction probability={probability}"
        )


        return {
            "default_probability": probability
        }


    except Exception as e:

        logger.error(
            f"Prediction failed: {e}"
        )


        put_metric(
            "PredictionErrorCount",
            1
        )


        raise HTTPException(
            status_code=500,
            detail="Prediction failed"
        )



@app.get("/model-info")
def model_info():

    logger.info(
        "Model information requested"
    )

    return load_metadata()