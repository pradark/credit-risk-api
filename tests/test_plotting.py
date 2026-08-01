"""Tests for model monitoring visualizations."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from app.monitoring.plotting import (
    plot_expected_vs_actual_default_rate,
    plot_roc_curve,
    save_figure,
)

from app.monitoring.calibration_metrics import (
    calculate_calibration_table,
)

def test_save_figure(
    tmp_path: Path,
) -> None:
    figure = plt.figure()

    output = (
        tmp_path
        / "test.png"
    )

    save_figure(
        figure,
        output,
    )

    assert output.exists()

    plt.close(
        figure
    )


def test_plot_roc_curve_returns_figure() -> None:
    data = pd.DataFrame(
        {
            "actual_default": [
                0,
                0,
                1,
                1,
            ],
            "predicted_probability": [
                0.10,
                0.30,
                0.70,
                0.90,
            ],
        }
    )

    figure = plot_roc_curve(
        data
    )

    assert isinstance(
        figure,
        plt.Figure,
    )

    assert len(
        figure.axes
    ) == 1

    plt.close(
        figure
    )


def test_plot_roc_curve_contains_two_lines() -> None:
    data = pd.DataFrame(
        {
            "actual_default": [
                0,
                0,
                1,
                1,
            ],
            "predicted_probability": [
                0.10,
                0.30,
                0.70,
                0.90,
            ],
        }
    )

    figure = plot_roc_curve(
        data
    )

    axis = figure.axes[
        0
    ]

    assert len(
        axis.lines
    ) == 2

    labels = [
        line.get_label()
        for line in axis.lines
    ]

    assert any(
        "AUC=" in label
        and "KS=" in label
        for label in labels
    )

    plt.close(
        figure
    )


def test_plot_roc_curve_saves_png(
    tmp_path: Path,
) -> None:
    data = pd.DataFrame(
        {
            "actual_default": [
                0,
                0,
                1,
                1,
            ],
            "predicted_probability": [
                0.10,
                0.30,
                0.70,
                0.90,
            ],
        }
    )

    output_path = (
        tmp_path
        / "roc_curve.png"
    )

    figure = plot_roc_curve(
        data=data,
        output_path=output_path,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0

    plt.close(
        figure
    )

def test_expected_vs_actual_returns_figure() -> None:
    data = pd.DataFrame(
        {
            "actual_default": [
                0,
                0,
                0,
                1,
                0,
                1,
                1,
                0,
                1,
                1,
            ],
            "predicted_probability": [
                0.05,
                0.10,
                0.20,
                0.30,
                0.40,
                0.50,
                0.60,
                0.70,
                0.80,
                0.90,
            ],
        }
    )

    table = calculate_calibration_table(
        data,
        number_of_bands=5,
    )

    figure = plot_expected_vs_actual_default_rate(
        table
    )

    assert isinstance(
        figure,
        plt.Figure,
    )

    plt.close(
        figure
    )


def test_expected_vs_actual_contains_two_lines() -> None:
    data = pd.DataFrame(
        {
            "actual_default": [
                0,
                0,
                0,
                1,
                0,
                1,
                1,
                0,
                1,
                1,
            ],
            "predicted_probability": [
                0.05,
                0.10,
                0.20,
                0.30,
                0.40,
                0.50,
                0.60,
                0.70,
                0.80,
                0.90,
            ],
        }
    )

    table = calculate_calibration_table(
        data,
        number_of_bands=5,
    )

    figure = plot_expected_vs_actual_default_rate(
        table
    )

    axis = figure.axes[0]

    assert len(
        axis.lines
    ) == 2

    plt.close(
        figure
    )


def test_expected_vs_actual_saves_png(
    tmp_path: Path,
) -> None:
    data = pd.DataFrame(
        {
            "actual_default": [
                0,
                0,
                0,
                1,
                0,
                1,
                1,
                0,
                1,
                1,
            ],
            "predicted_probability": [
                0.05,
                0.10,
                0.20,
                0.30,
                0.40,
                0.50,
                0.60,
                0.70,
                0.80,
                0.90,
            ],
        }
    )

    table = calculate_calibration_table(
        data,
        number_of_bands=5,
    )

    output = tmp_path / "calibration.png"

    figure = plot_expected_vs_actual_default_rate(
        table,
        output_path=output,
    )

    assert output.exists()

    plt.close(
        figure
    )