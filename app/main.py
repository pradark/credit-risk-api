from fastapi import FastAPI

from app.schemas import CreditRequest
from app.model import predict_probability
from app.logging_config import setup_logging, logger
from app.metadata import load_metadata
from app.s3_writer import write_parquet


setup_logging()


app = FastAPI(
    title="Credit Risk Prediction API"
)


@app.get("/health")
def health():

    logger.info("Health check requested")

    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(request: CreditRequest):

    logger.info(
        "Prediction request received"
    )

    features = request.model_dump()

    probability = predict_probability(
        features
    )

    write_parquet(
        features,
        "features"
    )

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


@app.get("/model-info")
def model_info():

    logger.info(
        "Model information requested"
    )

    return load_metadata()