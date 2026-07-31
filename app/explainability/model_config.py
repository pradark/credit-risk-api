"""Model-specific configuration for the credit risk API.

The explainability framework itself is model-agnostic. This module
contains the configuration for the model currently served by this API:

- ordered feature names
- feature display names
- candidate reason codes and descriptions
- compliance-review settings
- decision threshold
- SHAP reference-data location

A different model can use the same explainability framework by supplying
a different configuration module or FeatureRegistry.
"""

from __future__ import annotations

import os
from pathlib import Path

from app.config import (
    BASE_DIR,
    MODEL_CLASSIFICATION_THRESHOLD,
)
from app.explainability.feature_registry import (
    FeatureRegistry,
)


MODEL_FEATURE_NAMES = [
    "income",
    "fico",
    "loan_amount",
    "age",
    "debt",
    "employment",
    "balance",
    "utilization",
    "accounts",
    "history",
]


EXPLANATION_DECISION_THRESHOLD = float(
    os.getenv(
        "EXPLANATION_DECISION_THRESHOLD",
        str(
            MODEL_CLASSIFICATION_THRESHOLD
        ),
    )
)


EXPLANATION_BACKGROUND_PATH = Path(
    os.getenv(
        "EXPLANATION_BACKGROUND_PATH",
        (
            BASE_DIR
            / "monitoring"
            / "reference"
            / "reference_data.parquet"
        ),
    )
)


EXPLANATION_BACKGROUND_SAMPLE_SIZE = int(
    os.getenv(
        "EXPLANATION_BACKGROUND_SAMPLE_SIZE",
        "500",
    )
)


EXPLANATION_S3_PREFIX = os.getenv(
    "EXPLANATION_S3_PREFIX",
    "explanations",
)


FEATURE_REGISTRY = FeatureRegistry.from_dict(
    {
        "income": {
            "display_name": "Income",
            "reason_code": "INCOME_LEVEL",
            "adverse_reason": (
                "Income level increased the modeled risk"
            ),
            "favorable_reason": (
                "Income level reduced the modeled risk"
            ),
        },
        "fico": {
            "display_name": "Credit score",
            "reason_code": "CREDIT_SCORE",
            "adverse_reason": (
                "Credit score increased the modeled risk"
            ),
            "favorable_reason": (
                "Credit score reduced the modeled risk"
            ),
        },
        "loan_amount": {
            "display_name": "Requested loan amount",
            "reason_code": "REQUESTED_LOAN_AMOUNT",
            "adverse_reason": (
                "Requested loan amount increased the modeled risk"
            ),
            "favorable_reason": (
                "Requested loan amount reduced the modeled risk"
            ),
        },
        "age": {
            "display_name": "Age",
            "reason_code": "AGE_REVIEW",
            "adverse_reason": (
                "Age-related information increased the modeled risk"
            ),
            "favorable_reason": (
                "Age-related information reduced the modeled risk"
            ),
            "compliance_review_required": True,
        },
        "debt": {
            "display_name": "Existing debt",
            "reason_code": "EXISTING_DEBT",
            "adverse_reason": (
                "Existing debt increased the modeled risk"
            ),
            "favorable_reason": (
                "Existing debt reduced the modeled risk"
            ),
        },
        "employment": {
            "display_name": "Employment history",
            "reason_code": "EMPLOYMENT_HISTORY",
            "adverse_reason": (
                "Employment history increased the modeled risk"
            ),
            "favorable_reason": (
                "Employment history reduced the modeled risk"
            ),
        },
        "balance": {
            "display_name": "Account balance",
            "reason_code": "ACCOUNT_BALANCE",
            "adverse_reason": (
                "Account balance increased the modeled risk"
            ),
            "favorable_reason": (
                "Account balance reduced the modeled risk"
            ),
        },
        "utilization": {
            "display_name": "Credit utilization",
            "reason_code": "CREDIT_UTILIZATION",
            "adverse_reason": (
                "Credit utilization increased the modeled risk"
            ),
            "favorable_reason": (
                "Credit utilization reduced the modeled risk"
            ),
        },
        "accounts": {
            "display_name": "Number of credit accounts",
            "reason_code": "CREDIT_ACCOUNT_COUNT",
            "adverse_reason": (
                "Credit account count increased the modeled risk"
            ),
            "favorable_reason": (
                "Credit account count reduced the modeled risk"
            ),
        },
        "history": {
            "display_name": "Credit history length",
            "reason_code": "CREDIT_HISTORY_LENGTH",
            "adverse_reason": (
                "Credit history length increased the modeled risk"
            ),
            "favorable_reason": (
                "Credit history length reduced the modeled risk"
            ),
        },
    }
)
