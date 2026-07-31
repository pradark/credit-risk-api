"""Tests for the generic feature metadata registry."""

import pytest

from app.explainability.feature_registry import (
    FeatureDefinition,
    FeatureRegistry,
)


def test_register_and_get_feature_definition() -> None:
    registry = FeatureRegistry()

    definition = FeatureDefinition(
        feature_name="feature_a",
        display_name="Feature A",
        reason_code="FEATURE_A_REASON",
        adverse_reason="Feature A increased modeled risk",
    )

    registry.register(definition)

    assert registry.get("feature_a") == definition


def test_registry_accepts_initial_definitions() -> None:
    registry = FeatureRegistry(
        [
            FeatureDefinition(
                feature_name="feature_a",
                display_name="Feature A",
            ),
            FeatureDefinition(
                feature_name="feature_b",
                display_name="Feature B",
            ),
        ]
    )

    assert registry.feature_names() == [
        "feature_a",
        "feature_b",
    ]


def test_register_rejects_duplicate_feature() -> None:
    registry = FeatureRegistry(
        [
            FeatureDefinition(
                feature_name="feature_a",
                display_name="Feature A",
            )
        ]
    )

    with pytest.raises(
        ValueError,
        match="Feature is already registered: feature_a",
    ):
        registry.register(
            FeatureDefinition(
                feature_name="feature_a",
                display_name="Updated Feature A",
            )
        )


def test_register_can_overwrite_feature() -> None:
    registry = FeatureRegistry(
        [
            FeatureDefinition(
                feature_name="feature_a",
                display_name="Feature A",
            )
        ]
    )

    registry.register(
        FeatureDefinition(
            feature_name="feature_a",
            display_name="Updated Feature A",
        ),
        overwrite=True,
    )

    assert (
        registry.get("feature_a").display_name
        == "Updated Feature A"
    )


def test_register_rejects_empty_feature_name() -> None:
    registry = FeatureRegistry()

    with pytest.raises(
        ValueError,
        match="feature_name cannot be empty",
    ):
        registry.register(
            FeatureDefinition(
                feature_name="",
                display_name="Invalid Feature",
            )
        )


def test_get_unknown_feature_raises_clear_error() -> None:
    registry = FeatureRegistry()

    with pytest.raises(
        KeyError,
        match="Feature is not registered: unknown_feature",
    ):
        registry.get("unknown_feature")


def test_get_optional_returns_none_for_unknown_feature() -> None:
    registry = FeatureRegistry()

    assert registry.get_optional("unknown_feature") is None


def test_contains_reports_registration_status() -> None:
    registry = FeatureRegistry(
        [
            FeatureDefinition(
                feature_name="feature_a",
                display_name="Feature A",
            )
        ]
    )

    assert registry.contains("feature_a") is True
    assert registry.contains("feature_b") is False


def test_to_dict_returns_serializable_registry() -> None:
    registry = FeatureRegistry(
        [
            FeatureDefinition(
                feature_name="feature_a",
                display_name="Feature A",
                reason_code="FEATURE_A_REASON",
                compliance_review_required=True,
                metadata={
                    "category": "example",
                },
            )
        ]
    )

    result = registry.to_dict()

    assert result["feature_a"]["feature_name"] == "feature_a"
    assert result["feature_a"]["display_name"] == "Feature A"
    assert result["feature_a"]["reason_code"] == "FEATURE_A_REASON"
    assert (
        result["feature_a"][
            "compliance_review_required"
        ]
        is True
    )


def test_from_dict_creates_generic_registry() -> None:
    registry = FeatureRegistry.from_dict(
        {
            "custom_feature_one": {
                "display_name": "Custom Feature One",
                "reason_code": "CUSTOM_REASON_ONE",
                "adverse_reason": (
                    "Custom feature one increased modeled risk"
                ),
            },
            "custom_feature_two": {
                "display_name": "Custom Feature Two",
                "compliance_review_required": True,
            },
        }
    )

    assert registry.feature_names() == [
        "custom_feature_one",
        "custom_feature_two",
    ]

    assert (
        registry.get("custom_feature_one").reason_code
        == "CUSTOM_REASON_ONE"
    )

    assert (
        registry.get(
            "custom_feature_two"
        ).compliance_review_required
        is True
    )


def test_from_dict_uses_feature_name_as_default_display_name() -> None:
    registry = FeatureRegistry.from_dict(
        {
            "custom_feature": {},
        }
    )

    assert (
        registry.get("custom_feature").display_name
        == "custom_feature"
    )
