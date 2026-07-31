"""FastAPI application for credit-risk prediction and explainability."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException

from app.cloudwatch_metrics import put_metric
from app.explainability.background_loader import (
    load_background_data,
)
from app.explainability.explanation_service import (
    explain_prediction,
)
from app.explainability.model_config import (
    EXPLANATION_DECISION_THRESHOLD,
    EXPLANATION_S3_PREFIX,
    FEATURE_REGISTRY,
    MODEL_FEATURE_NAMES,
)
from app.logging_config import logger, setup_logging
from app.metadata import load_metadata
from app.model import model, predict_probability
from app.s3_writer import write_parquet
from app.schemas import CreditRequest, ExplainRequest


setup_logging()


app = FastAPI(
    title="Credit Risk Prediction API",
)


@lru_cache(maxsize=1)
def get_explanation_background():
    """Load and cache the SHAP background population."""

    return load_background_data()


@app.get("/health")
def health():
    """Return application health status."""

    logger.info(
        "Health check requested"
    )

    return {
        "status": "healthy",
    }


@app.post("/predict")
def predict(
    request: CreditRequest,
):
    """Return the predicted default probability."""

    logger.info(
        "Prediction request received"
    )

    try:
        features = request.model_dump()

        probability = predict_probability(
            features
        )

        put_metric(
            "PredictionCount",
            1,
        )

        put_metric(
            "DefaultProbability",
            probability,
            "None",
        )

        if probability >= 0.75:
            put_metric(
                "HighRiskPredictionCount",
                1,
            )

        write_parquet(
            features,
            "features",
        )

        prediction_record = {
            **features,
            "default_probability": probability,
        }

        write_parquet(
            prediction_record,
            "predictions",
        )

        logger.info(
            f"Prediction probability={probability}"
        )

        return {
            "default_probability": probability,
        }

    except Exception as exc:
        logger.error(
            f"Prediction failed: {exc}"
        )

        put_metric(
            "PredictionErrorCount",
            1,
        )

        raise HTTPException(
            status_code=500,
            detail="Prediction failed",
        ) from exc


@app.post("/explain")
def explain(
    request: ExplainRequest,
):
    """Return the prediction and ranked local explanation reasons."""

    logger.info(
        "Explanation request received "
        f"application_id={request.application_id}"
    )

    try:
        request_data = request.model_dump()

        application_id = request_data.pop(
            "application_id"
        )

        top_n = request_data.pop(
            "top_n"
        )

        persist = request_data.pop(
            "persist"
        )

        minimum_contribution = request_data.pop(
            "minimum_contribution"
        )

        features: dict[
            str,
            Any,
        ] = {
            feature_name: request_data[
                feature_name
            ]
            for feature_name
            in MODEL_FEATURE_NAMES
        }

        result = explain_prediction(
            application_id=application_id,
            features=features,
            model=model,
            predictor=predict_probability,
            feature_names=MODEL_FEATURE_NAMES,
            background_data=get_explanation_background(),
            decision_threshold=(
                EXPLANATION_DECISION_THRESHOLD
            ),
            registry=FEATURE_REGISTRY,
            top_n=top_n,
            direction="adverse",
            ranking="impact",
            minimum_contribution=(
                minimum_contribution
            ),
            persistence=(
                "s3"
                if persist
                else "none"
            ),
            s3_prefix=EXPLANATION_S3_PREFIX,
        )

        put_metric(
            "ExplanationCount",
            1,
        )

        if result[
            "compliance_review_required"
        ]:
            put_metric(
                "ExplanationComplianceReviewCount",
                1,
            )

        logger.info(
            "Explanation completed "
            f"application_id={application_id} "
            f"decision={result['decision']}"
        )

        return result

    except Exception as exc:
        logger.error(
            "Explanation failed "
            f"application_id={request.application_id}: "
            f"{exc}"
        )

        put_metric(
            "ExplanationErrorCount",
            1,
        )

        raise HTTPException(
            status_code=500,
            detail="Explanation failed",
        ) from exc


@app.get("/model-info")
def model_info():
    """Return model metadata."""

    logger.info(
        "Model information requested"
    )

    return load_metadata()