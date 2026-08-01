"""Utilities for model monitoring plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import pandas as pd
from sklearn.metrics import roc_curve

from app.monitoring.performance_metrics import (
    calculate_auc,
    calculate_ks,
    validate_performance_data,
)

from app.monitoring.calibration_metrics import (
    calculate_calibration_table,
)

def save_figure(
    figure: plt.Figure,
    output_path: str | Path,
) -> None:
    """Save a matplotlib figure."""

    Path(output_path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

def plot_roc_curve(
    data: pd.DataFrame,
    actual_column: str = "actual_default",
    probability_column: str = "predicted_probability",
    title: str = "ROC Curve",
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot the ROC curve and display AUC and KS."""

    validate_performance_data(
        data=data,
        actual_column=actual_column,
        probability_column=probability_column,
    )

    false_positive_rate, true_positive_rate, _ = roc_curve(
        data[actual_column],
        data[probability_column],
    )

    auc_value = calculate_auc(
        data=data,
        actual_column=actual_column,
        probability_column=probability_column,
    )

    ks_value = calculate_ks(
        data=data,
        actual_column=actual_column,
        probability_column=probability_column,
    )

    figure, axis = plt.subplots(
        figsize=(
            7,
            5,
        )
    )

    axis.plot(
        false_positive_rate,
        true_positive_rate,
        label=(
            f"Model: AUC={auc_value:.3f}, "
            f"KS={ks_value:.3f}"
        ),
    )

    axis.plot(
        [
            0,
            1,
        ],
        [
            0,
            1,
        ],
        linestyle="--",
        label="Random",
    )

    axis.set_xlim(
        0.0,
        1.0,
    )

    axis.set_ylim(
        0.0,
        1.05,
    )

    axis.set_xlabel(
        "False Positive Rate"
    )

    axis.set_ylabel(
        "True Positive Rate"
    )

    axis.set_title(
        title
    )

    axis.grid(
        alpha=0.3
    )

    axis.legend(
        loc="lower right"
    )

    figure.tight_layout()

    if output_path is not None:
        save_figure(
            figure=figure,
            output_path=output_path,
        )

    return figure

def plot_expected_vs_actual_default_rate(
    calibration_table: pd.DataFrame,
    title: str = (
        "Expected vs Actual Default Rate"
    ),
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot expected versus observed default rate."""

    required_columns = {
        "calibration_band",
        "average_predicted_pd",
        "actual_default_rate",
    }

    missing = sorted(
        required_columns.difference(
            calibration_table.columns
        )
    )

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    figure, axis = plt.subplots(
        figsize=(7, 5)
    )

    axis.plot(
        calibration_table[
            "calibration_band"
        ],
        calibration_table[
            "average_predicted_pd"
        ],
        marker="o",
        label="Expected PD",
    )

    axis.plot(
        calibration_table[
            "calibration_band"
        ],
        calibration_table[
            "actual_default_rate"
        ],
        marker="s",
        label="Actual Default Rate",
    )

    axis.set_xlabel(
        "Calibration Band"
    )

    axis.set_ylabel(
        "Default Rate"
    )

    axis.set_title(
        title
    )

    axis.grid(
        alpha=0.3
    )

    axis.legend()

    figure.tight_layout()

    if output_path is not None:
        save_figure(
            figure,
            output_path,
        )

    return figure