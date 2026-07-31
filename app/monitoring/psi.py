import numpy as np
import pandas as pd

from app.config import (
    PSI_WARNING_THRESHOLD,
    PSI_ALERT_THRESHOLD
)


def calculate_psi(
    reference,
    production,
    bins=10
):
    """
    Calculate Population Stability Index (PSI).

    PSI interpretation:
        < 0.10  : Normal
        0.10-0.25 : Warning
        > 0.25 : Alert

    Formula:

        PSI = Σ (actual_pct - expected_pct)
              * ln(actual_pct / expected_pct)

    Parameters:
        reference:
            Training/reference population values

        production:
            Production population values

        bins:
            Number of buckets for continuous variables

    Returns:
        float PSI value
    """

    reference = np.asarray(reference)
    production = np.asarray(production)


    # Create common bins using both populations
    breakpoints = np.linspace(
        min(
            reference.min(),
            production.min()
        ),
        max(
            reference.max(),
            production.max()
        ),
        bins + 1
    )


    reference_counts = np.histogram(
        reference,
        bins=breakpoints
    )[0]


    production_counts = np.histogram(
        production,
        bins=breakpoints
    )[0]


    reference_pct = (
        reference_counts /
        len(reference)
    )


    production_pct = (
        production_counts /
        len(production)
    )


    # Avoid zero division
    reference_pct = np.where(
        reference_pct == 0,
        0.0001,
        reference_pct
    )


    production_pct = np.where(
        production_pct == 0,
        0.0001,
        production_pct
    )


    psi_value = np.sum(
        (
            production_pct -
            reference_pct
        )
        *
        np.log(
            production_pct /
            reference_pct
        )
    )


    return float(psi_value)



def psi_status(
    psi_value
):
    """
    Convert PSI value into monitoring status.
    """

    if psi_value < PSI_WARNING_THRESHOLD:
        return "NORMAL"

    elif psi_value < PSI_ALERT_THRESHOLD:
        return "WARNING"

    else:
        return "ALERT"


def calculate_feature_psi(
    reference_df,
    production_df
):
    """
    Calculate PSI for all model features.

    Returns:
        pandas DataFrame:

        feature       psi       status
        income        0.012     NORMAL
        fico          0.062     NORMAL
    """

    results = []


    for column in reference_df.columns:

        psi_value = calculate_psi(
            reference_df[column],
            production_df[column]
        )


        results.append(
            {
                "feature": column,
                "psi": round(
                    psi_value,
                    4
                ),
                "status": psi_status(
                    psi_value
                )
            }
        )


    return pd.DataFrame(results)