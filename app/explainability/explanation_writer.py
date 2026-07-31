"""Build and persist single-row model explanation records.

Each output row represents one model decision and contains:

- prediction metadata
- probability of default
- selected adverse-action reasons
- complete feature contributions
- compliance-review information

Nested reason and contribution data are stored as Parquet list/struct
columns through PyArrow.
"""

from __future__ import annotations

import datetime
import uuid
from collections.abc import Sequence
from typing import Any, Literal

import pandas as pd

from app.config import (
    MODEL_VERSION,
    S3_BUCKET,
)
from app.monitoring.s3_utils import (
    write_parquet_to_s3,
)


TopN = int | Literal["all"]


REASON_OUTPUT_COLUMNS = [
    "reason_rank",
    "attribute_name",
    "display_name",
    "attribute_value",
    "average_attribute_value",
    "attribute_impact",
    "absolute_impact",
    "risk_vs_typical",
    "value_vs_typical",
    "reason_code",
    "adverse_reason",
    "compliance_review_required",
]


CONTRIBUTION_OUTPUT_COLUMNS = [
    "attribute_name",
    "attribute_value",
    "average_attribute_value",
    "attribute_impact",
    "absolute_impact",
    "ranked_impact_positive",
    "ranked_impact_negative",
    "ranked_impact_absolute",
    "risk_vs_typical",
    "value_vs_typical",
]


def validate_probability(
    default_probability: float,
) -> float:
    """Validate and normalize a probability of default."""

    try:
        probability = float(
            default_probability
        )
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise TypeError(
            "default_probability must be numeric"
        ) from exc

    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            "default_probability must be between 0 and 1"
        )

    return probability


def validate_decision_threshold(
    decision_threshold: float,
) -> float:
    """Validate and normalize a decision threshold."""

    try:
        threshold = float(
            decision_threshold
        )
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise TypeError(
            "decision_threshold must be numeric"
        ) from exc

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "decision_threshold must be between 0 and 1"
        )

    return threshold


def validate_top_n(
    requested_top_n: TopN,
) -> TopN:
    """Validate a requested reason count."""

    if requested_top_n == "all":
        return requested_top_n

    if not isinstance(
        requested_top_n,
        int,
    ):
        raise TypeError(
            "requested_top_n must be a positive integer or 'all'"
        )

    if requested_top_n < 1:
        raise ValueError(
            "requested_top_n must be at least 1"
        )

    return requested_top_n


def dataframe_records(
    data: pd.DataFrame,
    columns: Sequence[str],
) -> list[dict[str, Any]]:
    """Convert selected DataFrame columns into serializable records."""

    if not isinstance(
        data,
        pd.DataFrame,
    ):
        raise TypeError(
            "data must be a pandas DataFrame"
        )

    missing_columns = sorted(
        set(
            columns
        ).difference(
            data.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "DataFrame is missing required columns: "
            f"{missing_columns}"
        )

    records = data[
        list(
            columns
        )
    ].where(
        pd.notna(
            data[
                list(
                    columns
                )
            ]
        ),
        None,
    ).to_dict(
        orient="records"
    )

    normalized_records: list[
        dict[str, Any]
    ] = []

    for record in records:
        normalized_record: dict[
            str,
            Any
        ] = {}

        for key, value in record.items():
            if hasattr(
                value,
                "item",
            ):
                value = value.item()

            normalized_record[
                key
            ] = value

        normalized_records.append(
            normalized_record
        )

    return normalized_records


def build_compliance_flags(
    adverse_reasons: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Build compliance flags from selected explanation reasons."""

    if (
        "compliance_review_required"
        not in adverse_reasons.columns
    ):
        return []

    flagged = adverse_reasons[
        adverse_reasons[
            "compliance_review_required"
        ].astype(
            bool
        )
    ]

    flags: list[
        dict[str, Any]
    ] = []

    for _, row in flagged.iterrows():
        flags.append(
            {
                "attribute_name": row.get(
                    "attribute_name"
                ),
                "display_name": row.get(
                    "display_name"
                ),
                "reason_code": row.get(
                    "reason_code"
                ),
                "message": (
                    "Feature contribution requires "
                    "compliance review"
                ),
            }
        )

    return flags


def build_explanation_record(
    *,
    application_id: str,
    default_probability: float,
    decision_threshold: float,
    adverse_reasons: pd.DataFrame,
    all_feature_contributions: pd.DataFrame,
    requested_top_n: TopN = 4,
    prediction_id: str | None = None,
    prediction_timestamp: (
        datetime.datetime
        | None
    ) = None,
    model_version: str = MODEL_VERSION,
    decision: str | None = None,
    additional_metadata: (
        dict[str, Any]
        | None
    ) = None,
) -> pd.DataFrame:
    """Build one nested explanation row for one application.

    Args:
        application_id:
            Identifier for the application or model observation.
        default_probability:
            Predicted probability of the modeled event.
        decision_threshold:
            Threshold used to determine the decision.
        adverse_reasons:
            Selected Top N or all adverse reasons.
        all_feature_contributions:
            Complete feature-impact DataFrame.
        requested_top_n:
            Requested number of reasons or ``"all"``.
        prediction_id:
            Optional unique prediction identifier.
        prediction_timestamp:
            Optional UTC decision timestamp.
        model_version:
            Model version associated with the prediction.
        decision:
            Optional explicit decision. When omitted, a probability
            at or above the threshold produces ``"decline"``.
        additional_metadata:
            Optional model- or application-specific metadata.

    Returns:
        A one-row DataFrame suitable for Parquet storage.
    """

    if not application_id:
        raise ValueError(
            "application_id cannot be empty"
        )

    probability = validate_probability(
        default_probability
    )

    threshold = validate_decision_threshold(
        decision_threshold
    )

    normalized_top_n = validate_top_n(
        requested_top_n
    )

    if not isinstance(
        adverse_reasons,
        pd.DataFrame,
    ):
        raise TypeError(
            "adverse_reasons must be a pandas DataFrame"
        )

    if not isinstance(
        all_feature_contributions,
        pd.DataFrame,
    ):
        raise TypeError(
            "all_feature_contributions must be a pandas DataFrame"
        )

    active_prediction_id = (
        prediction_id
        or str(
            uuid.uuid4()
        )
    )

    active_timestamp = (
        prediction_timestamp
        or datetime.datetime.now(
            datetime.timezone.utc
        )
    )

    if active_timestamp.tzinfo is None:
        active_timestamp = (
            active_timestamp.replace(
                tzinfo=datetime.timezone.utc
            )
        )

    active_decision = (
        decision
        if decision is not None
        else (
            "decline"
            if probability >= threshold
            else "approve"
        )
    )

    adverse_reason_records = dataframe_records(
        data=adverse_reasons,
        columns=REASON_OUTPUT_COLUMNS,
    )

    contribution_records = dataframe_records(
        data=all_feature_contributions,
        columns=CONTRIBUTION_OUTPUT_COLUMNS,
    )

    compliance_flags = build_compliance_flags(
        adverse_reasons
    )

    record = {
        "prediction_id": active_prediction_id,
        "application_id": application_id,
        "prediction_timestamp": active_timestamp,
        "model_version": model_version,
        "default_probability": probability,
        "decision_threshold": threshold,
        "decision": active_decision,
        "requested_top_n": str(
            normalized_top_n
        ),
        "total_adverse_factors": len(
            adverse_reason_records
        ),
        "adverse_action_reasons": (
            adverse_reason_records
        ),
        "all_feature_contributions": (
            contribution_records
        ),
        "compliance_review_required": bool(
            compliance_flags
        ),
        "compliance_flags": compliance_flags,
        "additional_metadata": (
            additional_metadata
            or {}
        ),
    }

    return pd.DataFrame(
        [
            record
        ]
    )


def build_explanation_s3_key(
    *,
    application_id: str,
    prediction_id: str,
    prediction_timestamp: datetime.datetime,
    prefix: str = "explanations",
) -> str:
    """Build a partitioned S3 key for one explanation record."""

    if not application_id:
        raise ValueError(
            "application_id cannot be empty"
        )

    if not prediction_id:
        raise ValueError(
            "prediction_id cannot be empty"
        )

    run_date = prediction_timestamp.date().isoformat()

    return (
        f"{prefix}/"
        f"dt={run_date}/"
        f"application_id={application_id}/"
        f"explanation_{prediction_id}.parquet"
    )


def write_explanation_to_s3(
    *,
    application_id: str,
    default_probability: float,
    decision_threshold: float,
    adverse_reasons: pd.DataFrame,
    all_feature_contributions: pd.DataFrame,
    requested_top_n: TopN = 4,
    prediction_id: str | None = None,
    prediction_timestamp: (
        datetime.datetime
        | None
    ) = None,
    model_version: str = MODEL_VERSION,
    decision: str | None = None,
    additional_metadata: (
        dict[str, Any]
        | None
    ) = None,
    bucket: str = S3_BUCKET,
    prefix: str = "explanations",
) -> tuple[pd.DataFrame, str]:
    """Build and write one application explanation row to S3."""

    explanation = build_explanation_record(
        application_id=application_id,
        default_probability=default_probability,
        decision_threshold=decision_threshold,
        adverse_reasons=adverse_reasons,
        all_feature_contributions=(
            all_feature_contributions
        ),
        requested_top_n=requested_top_n,
        prediction_id=prediction_id,
        prediction_timestamp=prediction_timestamp,
        model_version=model_version,
        decision=decision,
        additional_metadata=additional_metadata,
    )

    active_prediction_id = str(
        explanation.loc[
            0,
            "prediction_id",
        ]
    )

    active_timestamp = explanation.loc[
        0,
        "prediction_timestamp",
    ]

    key = build_explanation_s3_key(
        application_id=application_id,
        prediction_id=active_prediction_id,
        prediction_timestamp=active_timestamp,
        prefix=prefix,
    )

    write_parquet_to_s3(
        explanation,
        bucket,
        key,
    )

    return explanation, key
