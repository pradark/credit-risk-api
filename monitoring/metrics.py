import pandas as pd
import logging


logger = logging.getLogger(__name__)


def calculate_metrics(
    features: pd.DataFrame,
    predictions: pd.DataFrame
) -> dict:
    """
    Calculate credit risk monitoring metrics.

    Returns:
        Dictionary containing production monitoring metrics.
    """

    metrics = {}


    # -------------------------
    # Prediction metrics
    # -------------------------

    if not predictions.empty:

        pd_values = predictions[
            "default_probability"
        ]


        metrics["prediction_count"] = int(
            len(predictions)
        )


        metrics["avg_default_probability"] = float(
            pd_values.mean()
        )


        metrics["min_default_probability"] = float(
            pd_values.min()
        )


        metrics["max_default_probability"] = float(
            pd_values.max()
        )


        # PD risk buckets

        metrics["pd_low_count"] = int(
            (pd_values < 0.05).sum()
        )


        metrics["pd_medium_count"] = int(
            (
                (pd_values >= 0.05)
                &
                (pd_values < 0.10)
            ).sum()
        )


        metrics["pd_high_count"] = int(
            (
                (pd_values >= 0.10)
                &
                (pd_values < 0.20)
            ).sum()
        )


        metrics["pd_very_high_count"] = int(
            (pd_values >= 0.20).sum()
        )


    else:

        metrics["prediction_count"] = 0



    # -------------------------
    # Feature metrics
    # -------------------------

    feature_metrics = [
        "fico",
        "income",
        "loan_amount",
        "utilization"
    ]


    for column in feature_metrics:

        if column in features.columns:

            metrics[
                f"avg_{column}"
            ] = float(
                features[column].mean()
            )


    logger.info(
        f"Calculated monitoring metrics: {metrics}"
    )


    return metrics