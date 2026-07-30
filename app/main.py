from fastapi import FastAPI

from app.schemas import CreditRequest
from app.model import predict_probability
from app.logging_config import setup_logging, logger
from app.metadata import load_metadata


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

    probability = predict_probability(
        request.model_dump()
    )

    logger.info(
        f"Prediction probability={probability}"
    )

    return {
        "default_probability": probability
    }


@app.get("/model-info")
def model_info():

    logger.info("Model information requested")

    return load_metadata()