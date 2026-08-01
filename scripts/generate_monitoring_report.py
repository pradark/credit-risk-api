"""Generate model monitoring artifacts from prediction data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(
    __file__
).resolve().parent.parent

if str(
    PROJECT_ROOT
) not in sys.path:
    sys.path.insert(
        0,
        str(
            PROJECT_ROOT
        ),
    )


from app.monitoring.report import (  # noqa: E402
    generate_html_report,
    generate_monitoring_dashboard,
)


DEFAULT_INPUT_PATH = Path(
    "monitoring/performance_joined.parquet"
)

DEFAULT_OUTPUT_DIRECTORY = Path(
    "reports"
)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate credit risk model monitoring "
            "metrics, plots, and an HTML report."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=(
            "CSV or Parquet file containing "
            "predicted_probability and actual_default."
        ),
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help=(
            "Directory where monitoring artifacts "
            "will be generated."
        ),
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help=(
            "Probability threshold used for "
            "binary classification metrics."
        ),
    )

    parser.add_argument(
        "--number-of-bands",
        type=int,
        default=10,
        help=(
            "Number of quantile-based calibration bands."
        ),
    )

    return parser.parse_args()


def load_prediction_data(
    input_path: Path,
) -> pd.DataFrame:
    """Load prediction and outcome data from CSV or Parquet."""

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file does not exist: {input_path}"
        )

    suffix = input_path.suffix.lower()

    if suffix == ".parquet":
        return pd.read_parquet(
            input_path
        )

    if suffix == ".csv":
        return pd.read_csv(
            input_path
        )

    raise ValueError(
        "Input file must be a CSV or Parquet file."
    )


def main() -> None:
    """Generate monitoring artifacts and print their locations."""

    arguments = parse_arguments()

    prediction_data = load_prediction_data(
        arguments.input
    )

    dashboard = generate_monitoring_dashboard(
        prediction_data=prediction_data,
        output_directory=(
            arguments.output_directory
        ),
        threshold=arguments.threshold,
        number_of_bands=(
            arguments.number_of_bands
        ),
    )

    report_path = generate_html_report(
        dashboard
    )

    print(
        "Monitoring artifacts generated:"
    )

    print(
        "Performance metrics: "
        f"{dashboard['performance_metrics_path']}"
    )

    print(
        "Calibration table: "
        f"{dashboard['calibration_table_path']}"
    )

    print(
        "ROC curve: "
        f"{dashboard['roc_curve_path']}"
    )

    print(
        "Expected vs actual: "
        f"{dashboard['expected_vs_actual_path']}"
    )

    print(
        "Expected calibration error: "
        f"{dashboard['expected_calibration_error']:.4f}"
    )

    print(
        "Maximum calibration error: "
        f"{dashboard['maximum_calibration_error']:.4f}"
    )

    print(
        f"HTML report: {report_path}"
    )


if __name__ == "__main__":
    main()