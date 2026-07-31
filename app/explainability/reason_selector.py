"""Generic reason selection and ranking for model explanations.

This module selects ranked feature impacts for adverse, favorable,
or complete model explanations.

It is model-agnostic and works with any feature-impact DataFrame that
follows the schema produced by feature_impact.py.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import pandas as pd

from app.explainability.feature_registry import (
    FeatureDefinition,
    FeatureRegistry,
)


ReasonDirection = Literal[
    "adverse",
    "favorable",
    "all",
]

RankingMethod = Literal[
    "impact",
    "absolute_impact",
    "business_priority",
]

TopN = int | Literal["all"]


REQUIRED_IMPACT_COLUMNS = {
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
}


def validate_reason_parameters(
    *,
    direction: ReasonDirection,
    ranking: RankingMethod,
    top_n: TopN,
    minimum_contribution: float,
) -> None:
    """Validate reason selection parameters."""

    valid_directions = {
        "adverse",
        "favorable",
        "all",
    }

    if direction not in valid_directions:
        raise ValueError(
            "direction must be one of: "
            "'adverse', 'favorable', or 'all'"
        )

    valid_rankings = {
        "impact",
        "absolute_impact",
        "business_priority",
    }

    if ranking not in valid_rankings:
        raise ValueError(
            "ranking must be one of: "
            "'impact', 'absolute_impact', "
            "or 'business_priority'"
        )

    if top_n != "all":
        if not isinstance(top_n, int):
            raise TypeError(
                "top_n must be a positive integer or 'all'"
            )

        if top_n < 1:
            raise ValueError(
                "top_n must be at least 1"
            )

    if minimum_contribution < 0:
        raise ValueError(
            "minimum_contribution must be non-negative"
        )


def validate_impact_dataframe(
    impact_data: pd.DataFrame,
) -> None:
    """Validate the feature-impact DataFrame."""

    if not isinstance(
        impact_data,
        pd.DataFrame,
    ):
        raise TypeError(
            "impact_data must be a pandas DataFrame"
        )

    if impact_data.empty:
        raise ValueError(
            "impact_data cannot be empty"
        )

    missing_columns = sorted(
        REQUIRED_IMPACT_COLUMNS.difference(
            impact_data.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "impact_data is missing required columns: "
            f"{missing_columns}"
        )


def infer_identifier_columns(
    impact_data: pd.DataFrame,
) -> list[str]:
    """Infer observation identifier columns.

    Identifier columns are all columns that are not part of the
    standardized feature-impact schema.
    """

    standardized_columns = {
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
    }

    return [
        column
        for column in impact_data.columns
        if column not in standardized_columns
    ]


def _business_priority(
    definition: FeatureDefinition | None,
) -> int:
    """Return configured business priority.

    Lower numbers indicate higher priority.
    Missing priorities are ranked last.
    """

    if (
        definition is None
        or definition.metadata is None
    ):
        return 1_000_000

    raw_priority = definition.metadata.get(
        "business_priority"
    )

    if raw_priority is None:
        return 1_000_000

    try:
        return int(raw_priority)
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise ValueError(
            "business_priority must be an integer"
        ) from exc


def _enrich_with_registry(
    impact_data: pd.DataFrame,
    registry: FeatureRegistry | None,
) -> pd.DataFrame:
    """Add configured feature metadata to impact rows."""

    enriched = impact_data.copy()

    display_names: list[str] = []
    reason_codes: list[str | None] = []
    adverse_reasons: list[str | None] = []
    favorable_reasons: list[str | None] = []
    compliance_flags: list[bool] = []
    business_priorities: list[int] = []

    for feature_name in enriched[
        "attribute_name"
    ]:
        definition = (
            registry.get_optional(
                feature_name
            )
            if registry is not None
            else None
        )

        display_names.append(
            definition.display_name
            if definition is not None
            else feature_name
        )

        reason_codes.append(
            definition.reason_code
            if definition is not None
            else None
        )

        adverse_reasons.append(
            definition.adverse_reason
            if definition is not None
            else None
        )

        favorable_reasons.append(
            definition.favorable_reason
            if definition is not None
            else None
        )

        compliance_flags.append(
            definition.compliance_review_required
            if definition is not None
            else False
        )

        business_priorities.append(
            _business_priority(
                definition
            )
        )

    enriched[
        "display_name"
    ] = display_names

    enriched[
        "reason_code"
    ] = reason_codes

    enriched[
        "adverse_reason"
    ] = adverse_reasons

    enriched[
        "favorable_reason"
    ] = favorable_reasons

    enriched[
        "compliance_review_required"
    ] = compliance_flags

    enriched[
        "business_priority"
    ] = business_priorities

    return enriched


def _filter_by_direction(
    impact_data: pd.DataFrame,
    direction: ReasonDirection,
    minimum_contribution: float,
) -> pd.DataFrame:
    """Filter impact rows by explanation direction."""

    if direction == "adverse":
        return impact_data[
            impact_data[
                "attribute_impact"
            ]
            >= minimum_contribution
        ].copy()

    if direction == "favorable":
        return impact_data[
            impact_data[
                "attribute_impact"
            ]
            <= -minimum_contribution
        ].copy()

    if minimum_contribution == 0:
        return impact_data.copy()

    return impact_data[
        impact_data[
            "absolute_impact"
        ]
        >= minimum_contribution
    ].copy()


def _sort_reasons(
    impact_data: pd.DataFrame,
    *,
    direction: ReasonDirection,
    ranking: RankingMethod,
    identifier_columns: Sequence[str],
) -> pd.DataFrame:
    """Sort reasons within each observation."""

    if ranking == "business_priority":
        sort_columns = (
            list(identifier_columns)
            + [
                "business_priority",
                "absolute_impact",
            ]
        )

        ascending = (
            [True]
            * len(
                identifier_columns
            )
            + [
                True,
                False,
            ]
        )

        return impact_data.sort_values(
            sort_columns,
            ascending=ascending,
        )

    if ranking == "absolute_impact":
        sort_columns = (
            list(identifier_columns)
            + [
                "absolute_impact",
            ]
        )

        ascending = (
            [True]
            * len(
                identifier_columns
            )
            + [
                False,
            ]
        )

        return impact_data.sort_values(
            sort_columns,
            ascending=ascending,
        )

    if direction == "favorable":
        rank_column = (
            "ranked_impact_negative"
        )
    elif direction == "all":
        rank_column = (
            "ranked_impact_absolute"
        )
    else:
        rank_column = (
            "ranked_impact_positive"
        )

    sort_columns = (
        list(identifier_columns)
        + [
            rank_column,
        ]
    )

    ascending = (
        [True]
        * len(
            identifier_columns
        )
        + [
            True,
        ]
    )

    return impact_data.sort_values(
        sort_columns,
        ascending=ascending,
    )


def _apply_top_n(
    impact_data: pd.DataFrame,
    identifier_columns: Sequence[str],
    top_n: TopN,
) -> pd.DataFrame:
    """Return Top N or all rows per observation."""

    if top_n == "all":
        return impact_data.copy()

    return (
        impact_data.groupby(
            list(
                identifier_columns
            ),
            group_keys=False,
            sort=False,
            dropna=False,
        )
        .head(
            top_n
        )
        .copy()
    )


def _add_reason_rank(
    impact_data: pd.DataFrame,
    identifier_columns: Sequence[str],
) -> pd.DataFrame:
    """Add a one-based rank within each observation."""

    ranked = impact_data.copy()

    ranked[
        "reason_rank"
    ] = (
        ranked.groupby(
            list(
                identifier_columns
            ),
            sort=False,
            dropna=False,
        )
        .cumcount()
        + 1
    )

    return ranked


def select_reasons(
    impact_data: pd.DataFrame,
    *,
    direction: ReasonDirection = "adverse",
    ranking: RankingMethod = "impact",
    top_n: TopN = 4,
    minimum_contribution: float = 0.0,
    registry: FeatureRegistry | None = None,
    identifier_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Select ranked explanation reasons.

    Args:
        impact_data:
            Feature-impact DataFrame.
        direction:
            Which feature direction to return:
            adverse, favorable, or all.
        ranking:
            Ranking method:
            impact, absolute_impact, or business_priority.
        top_n:
            Positive integer or "all".
        minimum_contribution:
            Minimum absolute contribution required.
        registry:
            Optional model-specific feature registry.
        identifier_columns:
            Optional explicit observation identifiers.
            When omitted, identifiers are inferred.

    Returns:
        Ranked long-form reason DataFrame.
    """

    validate_reason_parameters(
        direction=direction,
        ranking=ranking,
        top_n=top_n,
        minimum_contribution=minimum_contribution,
    )

    validate_impact_dataframe(
        impact_data
    )

    active_identifier_columns = list(
        identifier_columns
        or infer_identifier_columns(
            impact_data
        )
    )

    if not active_identifier_columns:
        raise ValueError(
            "At least one identifier column is required"
        )

    missing_identifier_columns = sorted(
        set(
            active_identifier_columns
        ).difference(
            impact_data.columns
        )
    )

    if missing_identifier_columns:
        raise ValueError(
            "impact_data is missing identifier columns: "
            f"{missing_identifier_columns}"
        )

    enriched = _enrich_with_registry(
        impact_data=impact_data,
        registry=registry,
    )

    filtered = _filter_by_direction(
        impact_data=enriched,
        direction=direction,
        minimum_contribution=minimum_contribution,
    )

    if filtered.empty:
        return filtered.assign(
            reason_rank=pd.Series(
                dtype="int64"
            )
        )

    sorted_reasons = _sort_reasons(
        impact_data=filtered,
        direction=direction,
        ranking=ranking,
        identifier_columns=active_identifier_columns,
    )

    selected = _apply_top_n(
        impact_data=sorted_reasons,
        identifier_columns=active_identifier_columns,
        top_n=top_n,
    )

    ranked = _add_reason_rank(
        impact_data=selected,
        identifier_columns=active_identifier_columns,
    )

    return ranked.reset_index(
        drop=True
    )
