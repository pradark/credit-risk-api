"""Model performance metric calculations for credit risk monitoring."""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


DEFAULT_ACTUAL_COLUMN: Final[str] = "actual_default"
DEFAULT_PROBABILITY_COLUMN: Final[str] = "predicted_probability"
DEFAULT_CLASSIFICATION_THRESHOLD: Final[float] = 0.50


def validate_performance_data(
    data: pd.DataFrame,
    actual_column: str = DEFAULT_ACTUAL_COLUMN,
    probability_column: str = DEFAULT_PROBABILITY_COLUMN,
) -> None:
    """Validate data required to calculate model performance metrics."""

    if not isinstance(
        data,
        pd.DataFrame,
    ):
        raise TypeError(
            "Performance data must be a pandas DataFrame."
        )

    if data.empty:
        raise ValueError(
            "Performance data is empty."
        )

    required_columns = {
        actual_column,
        probability_column,
    }

    missing_columns = sorted(
        required_columns.difference(
            data.columns
        )
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if data[
        actual_column
    ].isna().any():
        raise ValueError(
            f"{actual_column} contains missing values."
        )

    if data[
        probability_column
    ].isna().any():
        raise ValueError(
            f"{probability_column} contains missing values."
        )

    actual_values = set(
        data[
            actual_column
        ].unique()
    )

    if not actual_values.issubset(
        {
            0,
            1,
        }
    ):
        raise ValueError(
            f"{actual_column} must contain only 0 and 1."
        )

    if data[
        actual_column
    ].nunique() < 2:
        raise ValueError(
            "Both default and non-default outcomes are required "
            "to calculate AUC and KS."
        )

    if not data[
        probability_column
    ].between(
        0.0,
        1.0,
    ).all():
        raise ValueError(
            f"{probability_column} must contain values "
            "between 0 and 1."
        )


def validate_threshold(
    threshold: float,
) -> None:
    """Validate a binary classification probability threshold."""

    if not isinstance(
        threshold,
        (
            int,
            float,
        ),
    ):
        raise TypeError(
            "Classification threshold must be numeric."
        )

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "Classification threshold must be between 0 and 1."
        )


def build_predicted_class(
    data: pd.DataFrame,
    threshold: float = DEFAULT_CLASSIFICATION_THRESHOLD,
    probability_column: str = DEFAULT_PROBABILITY_COLUMN,
) -> pd.Series:
    """Convert predicted probabilities into binary predictions."""

    validate_threshold(
        threshold
    )

    if not isinstance(
        data,
        pd.DataFrame,
    ):
        raise TypeError(
            "Performance data must be a pandas DataFrame."
        )

    if probability_column not in data.columns:
        raise ValueError(
            f"Missing required column: {probability_column}"
        )

    if data[
        probability_column
    ].isna().any():
        raise ValueError(
            f"{probability_column} contains missing values."
        )

    if not data[
        probability_column
    ].between(
        0.0,
        1.0,
    ).all():
        raise ValueError(
            f"{probability_column} must contain values "
            "between 0 and 1."
        )

    return (
        data[
            probability_column
        ]
        >= threshold
    ).astype(
        int
    )


def calculate_auc(
    data: pd.DataFrame,
    actual_column: str = DEFAULT_ACTUAL_COLUMN,
    probability_column: str = DEFAULT_PROBABILITY_COLUMN,
) -> float:
    """Calculate area under the ROC curve."""

    validate_performance_data(
        data=data,
        actual_column=actual_column,
        probability_column=probability_column,
    )

    return float(
        roc_auc_score(
            data[
                actual_column
            ],
            data[
                probability_column
            ],
        )
    )


def calculate_ks(
    data: pd.DataFrame,
    actual_column: str = DEFAULT_ACTUAL_COLUMN,
    probability_column: str = DEFAULT_PROBABILITY_COLUMN,
) -> float:
    """Calculate the Kolmogorov-Smirnov statistic."""

    validate_performance_data(
        data=data,
        actual_column=actual_column,
        probability_column=probability_column,
    )

    false_positive_rate, true_positive_rate, _ = (
        roc_curve(
            data[
                actual_column
            ],
            data[
                probability_column
            ],
        )
    )

    return float(
        np.max(
            true_positive_rate
            - false_positive_rate
        )
    )


def calculate_gini(
    auc: float,
) -> float:
    """Calculate the Gini coefficient from AUC."""

    if not 0.0 <= auc <= 1.0:
        raise ValueError(
            "AUC must be between 0 and 1."
        )

    return float(
        (
            2.0
            * auc
        )
        - 1.0
    )


def calculate_bad_rate(
    data: pd.DataFrame,
    actual_column: str = DEFAULT_ACTUAL_COLUMN,
) -> float:
    """Calculate the observed default rate."""

    if not isinstance(
        data,
        pd.DataFrame,
    ):
        raise TypeError(
            "Performance data must be a pandas DataFrame."
        )

    if data.empty:
        raise ValueError(
            "Performance data is empty."
        )

    if actual_column not in data.columns:
        raise ValueError(
            f"Missing required column: {actual_column}"
        )

    if data[
        actual_column
    ].isna().any():
        raise ValueError(
            f"{actual_column} contains missing values."
        )

    actual_values = set(
        data[
            actual_column
        ].unique()
    )

    if not actual_values.issubset(
        {
            0,
            1,
        }
    ):
        raise ValueError(
            f"{actual_column} must contain only 0 and 1."
        )

    return float(
        data[
            actual_column
        ].mean()
    )


def calculate_average_predicted_pd(
    data: pd.DataFrame,
    probability_column: str = DEFAULT_PROBABILITY_COLUMN,
) -> float:
    """Calculate the average predicted probability of default."""

    if not isinstance(
        data,
        pd.DataFrame,
    ):
        raise TypeError(
            "Performance data must be a pandas DataFrame."
        )

    if data.empty:
        raise ValueError(
            "Performance data is empty."
        )

    if probability_column not in data.columns:
        raise ValueError(
            f"Missing required column: {probability_column}"
        )

    if data[
        probability_column
    ].isna().any():
        raise ValueError(
            f"{probability_column} contains missing values."
        )

    if not data[
        probability_column
    ].between(
        0.0,
        1.0,
    ).all():
        raise ValueError(
            f"{probability_column} must contain values "
            "between 0 and 1."
        )

    return float(
        data[
            probability_column
        ].mean()
    )


def calculate_prediction_standard_deviation(
    data: pd.DataFrame,
    probability_column: str = DEFAULT_PROBABILITY_COLUMN,
) -> float:
    """Calculate the population standard deviation of predicted PD."""

    validate_performance_data(
        data=data,
        probability_column=probability_column,
    )

    return float(
        data[
            probability_column
        ].std(
            ddof=0
        )
    )


def calculate_precision(
    data: pd.DataFrame,
    threshold: float = DEFAULT_CLASSIFICATION_THRESHOLD,
    actual_column: str = DEFAULT_ACTUAL_COLUMN,
    probability_column: str = DEFAULT_PROBABILITY_COLUMN,
) -> float:
    """Calculate precision using a probability threshold."""

    validate_performance_data(
        data=data,
        actual_column=actual_column,
        probability_column=probability_column,
    )

    predicted_class = build_predicted_class(
        data=data,
        threshold=threshold,
        probability_column=probability_column,
    )

    return float(
        precision_score(
            data[
                actual_column
            ],
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
    """Calculate recall using a probability threshold."""

    validate_performance_data(
        data=data,
        actual_column=actual_column,
        probability_column=probability_column,
    )

    predicted_class = build_predicted_class(
        data=data,
        threshold=threshold,
        probability_column=probability_column,
    )

    return float(
        recall_score(
            data[
                actual_column
            ],
            predicted_class,
            zero_division=0,
        )
    )


def calculate_accuracy(
    data: pd.DataFrame,
    threshold: float = DEFAULT_CLASSIFICATION_THRESHOLD,
    actual_column: str = DEFAULT_ACTUAL_COLUMN,
    probability_column: str = DEFAULT_PROBABILITY_COLUMN,
) -> float:
    """Calculate classification accuracy."""

    validate_performance_data(
        data=data,
        actual_column=actual_column,
        probability_column=probability_column,
    )

    predicted_class = build_predicted_class(
        data=data,
        threshold=threshold,
        probability_column=probability_column,
    )

    return float(
        accuracy_score(
            data[
                actual_column
            ],
            predicted_class,
        )
    )


def calculate_balanced_accuracy(
    data: pd.DataFrame,
    threshold: float = DEFAULT_CLASSIFICATION_THRESHOLD,
    actual_column: str = DEFAULT_ACTUAL_COLUMN,
    probability_column: str = DEFAULT_PROBABILITY_COLUMN,
) -> float:
    """Calculate balanced classification accuracy."""

    validate_performance_data(
        data=data,
        actual_column=actual_column,
        probability_column=probability_column,
    )

    predicted_class = build_predicted_class(
        data=data,
        threshold=threshold,
        probability_column=probability_column,
    )

    return float(
        balanced_accuracy_score(
            data[
                actual_column
            ],
            predicted_class,
        )
    )


def calculate_f1(
    data: pd.DataFrame,
    threshold: float = DEFAULT_CLASSIFICATION_THRESHOLD,
    actual_column: str = DEFAULT_ACTUAL_COLUMN,
    probability_column: str = DEFAULT_PROBABILITY_COLUMN,
) -> float:
    """Calculate the F1 score."""

    validate_performance_data(
        data=data,
        actual_column=actual_column,
        probability_column=probability_column,
    )

    predicted_class = build_predicted_class(
        data=data,
        threshold=threshold,
        probability_column=probability_column,
    )

    return float(
        f1_score(
            data[
                actual_column
            ],
            predicted_class,
            zero_division=0,
        )
    )


def calculate_confusion_matrix_counts(
    data: pd.DataFrame,
    threshold: float = DEFAULT_CLASSIFICATION_THRESHOLD,
    actual_column: str = DEFAULT_ACTUAL_COLUMN,
    probability_column: str = DEFAULT_PROBABILITY_COLUMN,
) -> dict[str, int]:
    """Return TN, FP, FN, and TP counts."""

    validate_performance_data(
        data=data,
        actual_column=actual_column,
        probability_column=probability_column,
    )

    predicted_class = build_predicted_class(
        data=data,
        threshold=threshold,
        probability_column=probability_column,
    )

    true_negative, false_positive, false_negative, true_positive = (
        confusion_matrix(
            data[
                actual_column
            ],
            predicted_class,
            labels=[
                0,
                1,
            ],
        ).ravel()
    )

    return {
        "true_positive": int(
            true_positive
        ),
        "false_positive": int(
            false_positive
        ),
        "true_negative": int(
            true_negative
        ),
        "false_negative": int(
            false_negative
        ),
    }


def calculate_specificity(
    data: pd.DataFrame,
    threshold: float = DEFAULT_CLASSIFICATION_THRESHOLD,
    actual_column: str = DEFAULT_ACTUAL_COLUMN,
    probability_column: str = DEFAULT_PROBABILITY_COLUMN,
) -> float:
    """Calculate the true-negative rate."""

    counts = calculate_confusion_matrix_counts(
        data=data,
        threshold=threshold,
        actual_column=actual_column,
        probability_column=probability_column,
    )

    denominator = (
        counts[
            "true_negative"
        ]
        + counts[
            "false_positive"
        ]
    )

    if denominator == 0:
        return 0.0

    return float(
        counts[
            "true_negative"
        ]
        / denominator
    )


def calculate_false_positive_rate(
    data: pd.DataFrame,
    threshold: float = DEFAULT_CLASSIFICATION_THRESHOLD,
    actual_column: str = DEFAULT_ACTUAL_COLUMN,
    probability_column: str = DEFAULT_PROBABILITY_COLUMN,
) -> float:
    """Calculate the false-positive rate."""

    return float(
        1.0
        - calculate_specificity(
            data=data,
            threshold=threshold,
            actual_column=actual_column,
            probability_column=probability_column,
        )
    )


def calculate_false_negative_rate(
    data: pd.DataFrame,
    threshold: float = DEFAULT_CLASSIFICATION_THRESHOLD,
    actual_column: str = DEFAULT_ACTUAL_COLUMN,
    probability_column: str = DEFAULT_PROBABILITY_COLUMN,
) -> float:
    """Calculate the false-negative rate."""

    recall = calculate_recall(
        data=data,
        threshold=threshold,
        actual_column=actual_column,
        probability_column=probability_column,
    )

    return float(
        1.0
        - recall
    )


def calculate_predicted_positive_rate(
    data: pd.DataFrame,
    threshold: float = DEFAULT_CLASSIFICATION_THRESHOLD,
    probability_column: str = DEFAULT_PROBABILITY_COLUMN,
) -> float:
    """Calculate the share of observations classified as positive."""

    predicted_class = build_predicted_class(
        data=data,
        threshold=threshold,
        probability_column=probability_column,
    )

    return float(
        predicted_class.mean()
    )


def calculate_brier_score(
    data: pd.DataFrame,
    actual_column: str = DEFAULT_ACTUAL_COLUMN,
    probability_column: str = DEFAULT_PROBABILITY_COLUMN,
) -> float:
    """Calculate Brier score for probability predictions."""

    validate_performance_data(
        data=data,
        actual_column=actual_column,
        probability_column=probability_column,
    )

    return float(
        brier_score_loss(
            data[
                actual_column
            ],
            data[
                probability_column
            ],
        )
    )


def calculate_log_loss(
    data: pd.DataFrame,
    actual_column: str = DEFAULT_ACTUAL_COLUMN,
    probability_column: str = DEFAULT_PROBABILITY_COLUMN,
) -> float:
    """Calculate binary log loss."""

    validate_performance_data(
        data=data,
        actual_column=actual_column,
        probability_column=probability_column,
    )

    return float(
        log_loss(
            data[
                actual_column
            ],
            data[
                probability_column
            ],
            labels=[
                0,
                1,
            ],
        )
    )


def calculate_calibration_error(
    data: pd.DataFrame,
    actual_column: str = DEFAULT_ACTUAL_COLUMN,
    probability_column: str = DEFAULT_PROBABILITY_COLUMN,
) -> float:
    """Calculate absolute portfolio-level calibration error."""

    validate_performance_data(
        data=data,
        actual_column=actual_column,
        probability_column=probability_column,
    )

    average_predicted_pd = (
        calculate_average_predicted_pd(
            data=data,
            probability_column=probability_column,
        )
    )

    observed_bad_rate = calculate_bad_rate(
        data=data,
        actual_column=actual_column,
    )

    return float(
        abs(
            average_predicted_pd
            - observed_bad_rate
        )
    )


def calculate_performance_metrics(
    data: pd.DataFrame,
    threshold: float = DEFAULT_CLASSIFICATION_THRESHOLD,
    actual_column: str = DEFAULT_ACTUAL_COLUMN,
    probability_column: str = DEFAULT_PROBABILITY_COLUMN,
) -> pd.DataFrame:
    """Calculate all supported model performance metrics."""

    validate_threshold(
        threshold
    )

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

    confusion_counts = (
        calculate_confusion_matrix_counts(
            data=data,
            threshold=threshold,
            actual_column=actual_column,
            probability_column=probability_column,
        )
    )

    metrics = {
        "record_count": int(
            len(
                data
            )
        ),
        "default_count": int(
            data[
                actual_column
            ].sum()
        ),
        "non_default_count": int(
            (
                data[
                    actual_column
                ]
                == 0
            ).sum()
        ),
        "auc": auc,
        "ks": calculate_ks(
            data=data,
            actual_column=actual_column,
            probability_column=probability_column,
        ),
        "gini": calculate_gini(
            auc
        ),
        "bad_rate": calculate_bad_rate(
            data=data,
            actual_column=actual_column,
        ),
        "average_predicted_pd": (
            calculate_average_predicted_pd(
                data=data,
                probability_column=probability_column,
            )
        ),
        "prediction_standard_deviation": (
            calculate_prediction_standard_deviation(
                data=data,
                probability_column=probability_column,
            )
        ),
        "minimum_predicted_pd": float(
            data[
                probability_column
            ].min()
        ),
        "maximum_predicted_pd": float(
            data[
                probability_column
            ].max()
        ),
        "accuracy": calculate_accuracy(
            data=data,
            threshold=threshold,
            actual_column=actual_column,
            probability_column=probability_column,
        ),
        "balanced_accuracy": (
            calculate_balanced_accuracy(
                data=data,
                threshold=threshold,
                actual_column=actual_column,
                probability_column=probability_column,
            )
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
        "specificity": calculate_specificity(
            data=data,
            threshold=threshold,
            actual_column=actual_column,
            probability_column=probability_column,
        ),
        "f1": calculate_f1(
            data=data,
            threshold=threshold,
            actual_column=actual_column,
            probability_column=probability_column,
        ),
        "false_positive_rate": (
            calculate_false_positive_rate(
                data=data,
                threshold=threshold,
                actual_column=actual_column,
                probability_column=probability_column,
            )
        ),
        "false_negative_rate": (
            calculate_false_negative_rate(
                data=data,
                threshold=threshold,
                actual_column=actual_column,
                probability_column=probability_column,
            )
        ),
        "predicted_positive_rate": (
            calculate_predicted_positive_rate(
                data=data,
                threshold=threshold,
                probability_column=probability_column,
            )
        ),
        "brier_score": calculate_brier_score(
            data=data,
            actual_column=actual_column,
            probability_column=probability_column,
        ),
        "log_loss": calculate_log_loss(
            data=data,
            actual_column=actual_column,
            probability_column=probability_column,
        ),
        "calibration_error": (
            calculate_calibration_error(
                data=data,
                actual_column=actual_column,
                probability_column=probability_column,
            )
        ),
        **confusion_counts,
        "classification_threshold": float(
            threshold
        ),
    }

    results = pd.DataFrame(
        [
            metrics
        ]
    )

    count_columns = [
        "record_count",
        "default_count",
        "non_default_count",
        "true_positive",
        "false_positive",
        "true_negative",
        "false_negative",
    ]

    results[
        count_columns
    ] = results[
        count_columns
    ].astype(
        int
    )

    numeric_metric_columns = [
        column
        for column in results.columns
        if column not in count_columns
    ]

    results[
        numeric_metric_columns
    ] = results[
        numeric_metric_columns
    ].round(
        4
    )

    return results