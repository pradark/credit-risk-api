"""Generic feature metadata registry for model explainability."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class FeatureDefinition:
    """Metadata describing one model feature."""

    feature_name: str
    display_name: str
    reason_code: str | None = None
    adverse_reason: str | None = None
    favorable_reason: str | None = None
    compliance_review_required: bool = False
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the definition as a serializable dictionary."""

        return asdict(self)


class FeatureRegistry:
    """Store feature metadata independently of any specific model."""

    def __init__(
        self,
        definitions: list[FeatureDefinition] | None = None,
    ) -> None:
        self._definitions: dict[str, FeatureDefinition] = {}

        for definition in definitions or []:
            self.register(definition)

    def register(
        self,
        definition: FeatureDefinition,
        *,
        overwrite: bool = False,
    ) -> None:
        """Register one feature definition."""

        if not isinstance(definition, FeatureDefinition):
            raise TypeError(
                "definition must be a FeatureDefinition"
            )

        feature_name = definition.feature_name.strip()

        if not feature_name:
            raise ValueError(
                "feature_name cannot be empty"
            )

        if (
            feature_name in self._definitions
            and not overwrite
        ):
            raise ValueError(
                f"Feature is already registered: {feature_name}"
            )

        self._definitions[feature_name] = definition

    def get(
        self,
        feature_name: str,
    ) -> FeatureDefinition:
        """Return metadata for one registered feature."""

        try:
            return self._definitions[feature_name]
        except KeyError as exc:
            raise KeyError(
                f"Feature is not registered: {feature_name}"
            ) from exc

    def get_optional(
        self,
        feature_name: str,
    ) -> FeatureDefinition | None:
        """Return feature metadata when available."""

        return self._definitions.get(feature_name)

    def contains(
        self,
        feature_name: str,
    ) -> bool:
        """Return whether the registry contains a feature."""

        return feature_name in self._definitions

    def feature_names(self) -> list[str]:
        """Return feature names in registration order."""

        return list(self._definitions)

    def definitions(self) -> list[FeatureDefinition]:
        """Return all registered definitions."""

        return list(self._definitions.values())

    def to_dict(self) -> dict[str, dict[str, Any]]:
        """Return the complete registry as a dictionary."""

        return {
            feature_name: definition.to_dict()
            for feature_name, definition
            in self._definitions.items()
        }

    @classmethod
    def from_dict(
        cls,
        configuration: dict[str, dict[str, Any]],
    ) -> "FeatureRegistry":
        """Build a registry from model-specific configuration."""

        definitions: list[FeatureDefinition] = []

        for feature_name, values in configuration.items():
            definitions.append(
                FeatureDefinition(
                    feature_name=feature_name,
                    display_name=values.get(
                        "display_name",
                        feature_name,
                    ),
                    reason_code=values.get(
                        "reason_code"
                    ),
                    adverse_reason=values.get(
                        "adverse_reason"
                    ),
                    favorable_reason=values.get(
                        "favorable_reason"
                    ),
                    compliance_review_required=values.get(
                        "compliance_review_required",
                        False,
                    ),
                    metadata=values.get(
                        "metadata"
                    ),
                )
            )

        return cls(definitions)
