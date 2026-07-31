"""Model performance metric calculations for credit risk monitoring."""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve


DEFAULT_ACTUAL_COLUMN: Final[str] = "actual_default"
DEFAULT_PROBABILITY_COLUMN: Final[str] = "predicted_probability"
DEFAULT_CLASSIFICATION_THRESHOLD: Final[float] = 0.50


def validate_performance_data(
    data: pd.DataFrame,
    actual_column: str = DEFAULT_ACTUAL_COLUMN,
    probability_column: str = DEFAULT_PROBABILITY_COLUMN,
) -> None:
    """
    Validate data required to calculate model performance metrics.

    Parameters
    ----------
    data:
        Dataset containing actual outcomes and predicted probabilities.
    actual_column:
        Column containing binary actual outcomes, where 1 represents default.
    probability_column:
        Column containing predicted probabilities of default.

    Raises
    ------
    TypeError
        If data is not a pandas DataFrame.
    ValueError
        If required columns or valid values are missing.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Performance data must be a pandas DataFrame.")

    if data.empty:
        raise ValueError("Performance data is empty.")

    required_columns = {actual_column, probability_column}
    missing_columns = sorted(required_columns.difference(data.columns))

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if data[actual_column].isna().any():
        raise ValueError(f"{actual_column} contains missing values.")

    if data[probability_column].isna().any():
        raise ValueError(f"{probability_column} contains missing values.")

    actual_values = set(data[actual_column].unique())

    if not actual_values.issubset({0, 1}):
        raise ValueError(f"{actual_column} must contain only 0 and 1.")

    if data[actual_column].nunique() < 2:
        raise ValueError(
            "Both default and non-default outcomes are required "
            "to calculate AUC and KS."
        )

    if not data[probability_column].between(0.0, 1.0).all():
        raise ValueError(
            f"{probability_column} must contain values between 0 and 1."
        )


def validate_threshold(threshold: float) -> None:
    """
    Validate a binary classification probability threshold.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Classification threshold must be between 0 and 1.")


def calculate_auc(
    data: pd.DataFrame,
    actual_column: str = DEFAULT_ACTUAL_COLUMN,
    probability_column: str = DEFAULT_PROBABILITY_COLUMN,
) -> float:
    """
    Calculate the area under the receiver operating characteristic curve.
    """
    validate_performance_data(
        data=data,
        actual_column=actual_column,
        probability_column=probability_column,
    )

    return float(
        roc_auc_score(
            data[actual_column],
            data[probability_column],
        )
    )


def calculate_ks(
    data: pd.DataFrame,
    actual_column: str = DEFAULT_ACTUAL_COLUMN,
    probability_column: str = DEFAULT_PROBABILITY_COLUMN,
) -> float:
    """
    Calculate the Kolmogorov-Smirnov statistic.

    The KS statistic is the maximum difference between the cumulative
    distributions of defaults and non-defaults across model thresholds.
    """
    validate_performance_data(
        data=data,
        actual_column=actual_column,
        probability_column=probability_column,
    )

    false_positive_rate, true_positive_rate, _ = roc_curve(
        data[actual_column],
        data[probability_column],
    )

    return float(np.max(true_positive_rate - false_positive_rate))


def calculate_gini(auc: float) -> float:
    """
    Calculate the Gini coefficient from AUC.
    """
    if not 0.0 <= auc <= 1.0:
        raise ValueError("AUC must be between 0 and 1.")

    return float((2.0 * auc) - 1.0)


def calculate_bad_rate(
    data: pd.DataFrame,
    actual_column: str = DEFAULT_ACTUAL_COLUMN,
) -> float:
    """
    Calculate the observed default rate.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Performance data must be a pandas DataFrame.")

    if data.empty:
        raise ValueError("Performance data is empty.")

    if actual_column not in data.columns:
        raise ValueError(f"Missing required column: {actual_column}")

    if data[actual_column].isna().any():
        raise ValueError(f"{actual_column} contains missing values.")

    actual_values = set(data[actual_column].unique())

    if not actual_values.issubset({0, 1}):
        raise ValueError(f"{actual_column} must contain only 0 and 1.")

    return float(data[actual_column].mean())


def calculate_average_predicted_pd(
    data: pd.DataFrame,
    probability_column: str = DEFAULT_PROBABILITY_COLUMN,
) -> float:
    """
    Calculate the average predicted probability of default.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Performance data must be a pandas DataFrame.")

    if data.empty:
        raise ValueError("Performance data is empty.")

    if probability_column not in data.columns:
        raise ValueError(f"Missing required column: {probability_column}")

    if data[probability_column].isna().any():
        raise ValueError(f"{probability_column} contains missing values.")

    if not data[probability_column].between(0.0, 1.0).all():
        raise ValueError(
            f"{probability_column} must contain values between 0 and 1."
        )

    return float(data[probability_column].mean())


def calculate_precision(
    data: pd.DataFrame,
    threshold: float = DEFAULT_CLASSIFICATION_THRESHOLD,
    actual_column: str = DEFAULT_ACTUAL_COLUMN,
    probability_column: str = DEFAULT_PROBABILITY_COLUMN,
) -> float:
    """
    Calculate precision using a probability threshold.
    """
    validate_threshold(threshold)

    validate_performance_data(
        data=data,
        actual_column=actual_column,
        probability_column=probability_column,
    )

    predicted_class = (data[probability_column] >= threshold).astype(int)

    return float(
        precision_score(
            data[actual_column],
            predicted_class,
            zero_division=0,
        )
    )


def calculate_recall(
    data: pd.DataFrame,
    threshold: float = DEFAULT_CLASSIFICATION_THRESHOLD,
    actual_column: str = DEFAULT_ACTUAL_COLUMN,
    probability_column: str = DEFAULT_PROBABILITY_COLUMN,
) -> float:
    """
    Calculate recall using a probability threshold.
    """
    validate_threshold(threshold)

    validate_performance_data(
        data=data,
        actual_column=actual_column,
        probability_column=probability_column,
    )

    predicted_class = (data[probability_column] >= threshold).astype(int)

    return float(
        recall_score(
            data[actual_column],
            predicted_class,
            zero_division=0,
        )
    )


def calculate_calibration_error(
    data: pd.DataFrame,
    actual_column: str = DEFAULT_ACTUAL_COLUMN,
    probability_column: str = DEFAULT_PROBABILITY_COLUMN,
) -> float:
    """
    Calculate absolute calibration error.

    This metric measures the absolute difference between the average
    predicted probability of default and the observed default rate.
    """
    validate_performance_data(
        data=data,
        actual_column=actual_column,
        probability_column=probability_column,
    )

    average_predicted_pd = calculate_average_predicted_pd(
        data=data,
        probability_column=probability_column,
    )

    observed_bad_rate = calculate_bad_rate(
        data=data,
        actual_column=actual_column,
    )

    return float(abs(average_predicted_pd - observed_bad_rate))


def calculate_performance_metrics(
    data: pd.DataFrame,
    threshold: float = DEFAULT_CLASSIFICATION_THRESHOLD,
    actual_column: str = DEFAULT_ACTUAL_COLUMN,
    probability_column: str = DEFAULT_PROBABILITY_COLUMN,
) -> pd.DataFrame:
    """
    Calculate all supported credit risk model performance metrics.

    Returns
    -------
    pandas.DataFrame
        One row containing the calculated model performance metrics.
    """
    validate_threshold(threshold)

    validate_performance_data(
        data=data,
        actual_column=actual_column,
        probability_column=probability_column,
    )

    auc = calculate_auc(
        data=data,
        actual_column=actual_column,
        probability_column=probability_column,
    )

    metrics = {
        "record_count": int(len(data)),
        "default_count": int(data[actual_column].sum()),
        "non_default_count": int((data[actual_column] == 0).sum()),
        "auc": auc,
        "ks": calculate_ks(
            data=data,
            actual_column=actual_column,
            probability_column=probability_column,
        ),
        "gini": calculate_gini(auc),
        "bad_rate": calculate_bad_rate(
            data=data,
            actual_column=actual_column,
        ),
        "average_predicted_pd": calculate_average_predicted_pd(
            data=data,
            probability_column=probability_column,
        ),
        "precision": calculate_precision(
            data=data,
            threshold=threshold,
            actual_column=actual_column,
            probability_column=probability_column,
        ),
        "recall": calculate_recall(
            data=data,
            threshold=threshold,
            actual_column=actual_column,
            probability_column=probability_column,
        ),
        "calibration_error": calculate_calibration_error(
            data=data,
            actual_column=actual_column,
            probability_column=probability_column,
        ),
        "classification_threshold": float(threshold),
    }

    results = pd.DataFrame([metrics])

    numeric_metric_columns = [
        "auc",
        "ks",
        "gini",
        "bad_rate",
        "average_predicted_pd",
        "precision",
        "recall",
        "calibration_error",
        "classification_threshold",
    ]

    results[numeric_metric_columns] = results[numeric_metric_columns].round(4)

    return results
