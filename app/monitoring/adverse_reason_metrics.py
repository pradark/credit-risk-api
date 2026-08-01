"""Calculate adverse reason summary metrics for monitoring."""

from __future__ import annotations

import pandas as pd


REASON_CODE_COLUMN = "reason_code"
ADVERSE_REASON_COLUMN = "adverse_reason"
CONTRIBUTION_COLUMN = "contribution"
TOTAL_RECORDS_COLUMN = "total_records"


def validate_adverse_reason_data(
    data: pd.DataFrame,
) -> None:
    """Validate adverse reason data for summary calculation.

    Parameters
    ----------
    data:
        DataFrame containing one row per adverse reason per prediction.
        Expected columns: reason_code, adverse_reason, contribution.

    Raises
    ------
    TypeError
        If data is not a pandas DataFrame.
    ValueError
        If required columns are missing or the DataFrame is empty.
    """

    if not isinstance(
        data,
        pd.DataFrame,
    ):
        raise TypeError(
            "Adverse reason data must be a pandas DataFrame."
        )

    if data.empty:
        raise ValueError(
            "Adverse reason data is empty."
        )

    required = {
        REASON_CODE_COLUMN,
        ADVERSE_REASON_COLUMN,
        CONTRIBUTION_COLUMN,
    }

    missing = sorted(
        required.difference(
            data.columns
        )
    )

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )


def calculate_adverse_reason_summary(
    data: pd.DataFrame,
    model_version: str,
    report_date: str,
    environment: str,
    total_predictions: int | None = None,
) -> pd.DataFrame:
    """Calculate adverse reason frequency and contribution summary.

    For each unique reason code, computes selection count, selection
    rate (as share of total predictions), and average SHAP contribution.

    Parameters
    ----------
    data:
        Adverse reason DataFrame with reason_code, adverse_reason,
        and contribution columns.
    model_version:
        Model version tag.
    report_date:
        Report date string.
    environment:
        Deployment environment.
    total_predictions:
        Total number of predictions in the monitoring window. Used to
        calculate selection_rate. Defaults to the number of distinct
        (reason_code, contribution) combinations if not provided.

    Returns
    -------
    pd.DataFrame
        Summary with columns: reason_code, adverse_reason,
        selection_count, selection_rate, average_contribution,
        model_version, report_date, environment.
    """

    validate_adverse_reason_data(
        data
    )

    grouped = (
        data.groupby(
            [
                REASON_CODE_COLUMN,
                ADVERSE_REASON_COLUMN,
            ],
            observed=True,
            as_index=False,
        )
        .agg(
            selection_count=(
                CONTRIBUTION_COLUMN,
                "count",
            ),
            average_contribution=(
                CONTRIBUTION_COLUMN,
                "mean",
            ),
        )
    )

    denominator = (
        total_predictions
        if total_predictions is not None
        and total_predictions > 0
        else int(
            grouped[
                "selection_count"
            ].sum()
        )
    )

    grouped[
        "selection_rate"
    ] = (
        grouped[
            "selection_count"
        ]
        / denominator
    ).round(
        4
    )

    grouped[
        "average_contribution"
    ] = grouped[
        "average_contribution"
    ].round(
        4
    )

    grouped[
        "selection_count"
    ] = grouped[
        "selection_count"
    ].astype(
        int
    )

    grouped[
        "model_version"
    ] = model_version

    grouped[
        "report_date"
    ] = report_date

    grouped[
        "environment"
    ] = environment

    return grouped.sort_values(
        "selection_count",
        ascending=False,
    ).reset_index(
        drop=True
    )[
        [
            "reason_code",
            "adverse_reason",
            "selection_count",
            "selection_rate",
            "average_contribution",
            "model_version",
            "report_date",
            "environment",
        ]
    ]
