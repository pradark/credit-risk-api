"""End-to-end model explanation orchestration.

This module connects:

- model prediction
- TreeSHAP feature impacts
- configurable reason selection
- feature metadata enrichment
- single-row explanation construction
- optional S3 persistence

The service is model-agnostic. Models, predictors, feature names,
background data, registries, and storage functions are supplied by
the caller.
"""

from __future__ import annotations

import datetime
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import pandas as pd

from app.config import MODEL_VERSION
from app.explainability.explanation_writer import (
    build_explanation_record,
    write_explanation_to_s3,
)
from app.explainability.feature_impact import (
    generate_tree_feature_impacts,
)
from app.explainability.feature_registry import (
    FeatureRegistry,
)
from app.explainability.reason_selector import (
    RankingMethod,
    ReasonDirection,
    TopN,
    select_reasons,
)


PersistenceMode = Literal[
    "none",
    "s3",
]


def validate_feature_mapping(
    features: Mapping[str, Any],
    feature_names: Sequence[str],
) -> dict[str, Any]:
    """Validate and order one observation's model features."""

    if not isinstance(features, Mapping):
        raise TypeError(
            "features must be a mapping"
        )

    normalized_feature_names = list(
        feature_names
    )

    if not normalized_feature_names:
        raise ValueError(
            "feature_names cannot be empty"
        )

    if len(normalized_feature_names) != len(
        set(normalized_feature_names)
    ):
        raise ValueError(
            "feature_names must not contain duplicates"
        )

    missing_features = [
        feature_name
        for feature_name in normalized_feature_names
        if feature_name not in features
    ]

    if missing_features:
        raise ValueError(
            "Missing required features: "
            f"{missing_features}"
        )

    return {
        feature_name: features[feature_name]
        for feature_name in normalized_feature_names
    }


def create_observation_dataframe(
    *,
    application_id: str,
    features: Mapping[str, Any],
    feature_names: Sequence[str],
    application_id_column: str = "application_id",
    observation_timestamp: (
        datetime.datetime
        | None
    ) = None,
    timestamp_column: str | None = None,
) -> pd.DataFrame:
    """Create a one-row observation DataFrame."""

    if not application_id:
        raise ValueError(
            "application_id cannot be empty"
        )

    ordered_features = validate_feature_mapping(
        features=features,
        feature_names=feature_names,
    )

    record: dict[str, Any] = {
        application_id_column: application_id,
        **ordered_features,
    }

    if timestamp_column is not None:
        active_timestamp = (
            observation_timestamp
            or datetime.datetime.now(
                datetime.timezone.utc
            )
        )

        if active_timestamp.tzinfo is None:
            active_timestamp = active_timestamp.replace(
                tzinfo=datetime.timezone.utc
            )

        record[
            timestamp_column
        ] = active_timestamp

    return pd.DataFrame(
        [
            record
        ]
    )


def predict_default_probability(
    *,
    predictor: Callable[
        [
            dict[str, Any]
        ],
        float,
    ],
    features: dict[str, Any],
) -> float:
    """Call the supplied predictor and validate its probability."""

    if not callable(predictor):
        raise TypeError(
            "predictor must be callable"
        )

    probability = float(
        predictor(features)
    )

    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            "predictor returned a probability "
            "outside the range 0 to 1"
        )

    return probability


def determine_decision(
    *,
    default_probability: float,
    decision_threshold: float,
) -> str:
    """Return approve or decline using the configured threshold."""

    if not 0.0 <= decision_threshold <= 1.0:
        raise ValueError(
            "decision_threshold must be between 0 and 1"
        )

    return (
        "decline"
        if default_probability >= decision_threshold
        else "approve"
    )


def dataframe_to_records(
    dataframe: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Convert a DataFrame to JSON-compatible records."""

    records = dataframe.where(
        pd.notna(dataframe),
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


def explain_prediction(
    *,
    application_id: str,
    features: Mapping[str, Any],
    model: Any,
    predictor: Callable[
        [
            dict[str, Any]
        ],
        float,
    ],
    feature_names: Sequence[str],
    background_data: pd.DataFrame,
    decision_threshold: float,
    registry: FeatureRegistry | None = None,
    top_n: TopN = 4,
    direction: ReasonDirection = "adverse",
    ranking: RankingMethod = "impact",
    minimum_contribution: float = 0.0,
    model_version: str = MODEL_VERSION,
    prediction_id: str | None = None,
    prediction_timestamp: (
        datetime.datetime
        | None
    ) = None,
    persistence: PersistenceMode = "none",
    s3_bucket: str | None = None,
    s3_prefix: str = "explanations",
    application_id_column: str = "application_id",
    explainer: Any | None = None,
    additional_metadata: (
        dict[str, Any]
        | None
    ) = None,
) -> dict[str, Any]:
    """Explain one model prediction and optionally persist it.

    Args:
        application_id:
            Observation or application identifier.
        features:
            Model input values.
        model:
            Tree-based model used by SHAP.
        predictor:
            Callable returning the modeled event probability.
        feature_names:
            Ordered model feature names.
        background_data:
            Reference population used by TreeSHAP.
        decision_threshold:
            Probability threshold used for the decision.
        registry:
            Optional feature metadata registry.
        top_n:
            Number of selected reasons or ``"all"``.
        direction:
            Adverse, favorable, or all explanations.
        ranking:
            Impact, absolute impact, or business priority.
        minimum_contribution:
            Minimum contribution required for selection.
        persistence:
            ``"none"`` or ``"s3"``.
        s3_bucket:
            Optional S3 bucket override.
        s3_prefix:
            S3 explanation prefix.
        explainer:
            Optional prebuilt TreeSHAP explainer.

    Returns:
        API-ready explanation dictionary.
    """

    if persistence not in {
        "none",
        "s3",
    }:
        raise ValueError(
            "persistence must be 'none' or 's3'"
        )

    ordered_features = validate_feature_mapping(
        features=features,
        feature_names=feature_names,
    )

    probability = predict_default_probability(
        predictor=predictor,
        features=ordered_features,
    )

    decision = determine_decision(
        default_probability=probability,
        decision_threshold=decision_threshold,
    )

    active_timestamp = (
        prediction_timestamp
        or datetime.datetime.now(
            datetime.timezone.utc
        )
    )

    if active_timestamp.tzinfo is None:
        active_timestamp = active_timestamp.replace(
            tzinfo=datetime.timezone.utc
        )

    observations = create_observation_dataframe(
        application_id=application_id,
        features=ordered_features,
        feature_names=feature_names,
        application_id_column=application_id_column,
    )

    impacts = generate_tree_feature_impacts(
        model=model,
        observations=observations,
        feature_names=feature_names,
        background_data=background_data,
        entity_id_columns=[
            application_id_column
        ],
        explainer=explainer,
    )

    selected_reasons = select_reasons(
        impact_data=impacts,
        direction=direction,
        ranking=ranking,
        top_n=top_n,
        minimum_contribution=minimum_contribution,
        registry=registry,
        identifier_columns=[
            application_id_column
        ],
    )

    if persistence == "s3":
        write_arguments: dict[
            str,
            Any
        ] = {
            "application_id": application_id,
            "default_probability": probability,
            "decision_threshold": decision_threshold,
            "adverse_reasons": selected_reasons,
            "all_feature_contributions": impacts,
            "requested_top_n": top_n,
            "prediction_id": prediction_id,
            "prediction_timestamp": active_timestamp,
            "model_version": model_version,
            "decision": decision,
            "additional_metadata": additional_metadata,
            "prefix": s3_prefix,
        }

        if s3_bucket is not None:
            write_arguments[
                "bucket"
            ] = s3_bucket

        explanation_record, s3_key = (
            write_explanation_to_s3(
                **write_arguments
            )
        )
    else:
        explanation_record = build_explanation_record(
            application_id=application_id,
            default_probability=probability,
            decision_threshold=decision_threshold,
            adverse_reasons=selected_reasons,
            all_feature_contributions=impacts,
            requested_top_n=top_n,
            prediction_id=prediction_id,
            prediction_timestamp=active_timestamp,
            model_version=model_version,
            decision=decision,
            additional_metadata=additional_metadata,
        )

        s3_key = None

    record = explanation_record.iloc[
        0
    ]

    return {
        "prediction_id": record[
            "prediction_id"
        ],
        "application_id": record[
            "application_id"
        ],
        "prediction_timestamp": record[
            "prediction_timestamp"
        ],
        "model_version": record[
            "model_version"
        ],
        "default_probability": float(
            record[
                "default_probability"
            ]
        ),
        "decision_threshold": float(
            record[
                "decision_threshold"
            ]
        ),
        "decision": record[
            "decision"
        ],
        "requested_top_n": record[
            "requested_top_n"
        ],
        "total_selected_reasons": int(
            record[
                "total_adverse_factors"
            ]
        ),
        "reasons": record[
            "adverse_action_reasons"
        ],
        "all_feature_contributions": record[
            "all_feature_contributions"
        ],
        "compliance_review_required": bool(
            record[
                "compliance_review_required"
            ]
        ),
        "compliance_flags": record[
            "compliance_flags"
        ],
        "s3_key": s3_key,
    }
