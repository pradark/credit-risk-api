"""Pydantic request schemas for the credit risk API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class CreditRequest(BaseModel):
    """Model features used for credit-risk prediction."""

    income: float = Field(
        gt=0,
        description="Annual income",
    )

    fico: float = Field(
        ge=300,
        le=850,
        description="FICO credit score",
    )

    loan_amount: float = Field(
        gt=0,
        description="Requested loan amount",
    )

    age: float = Field(
        ge=18,
        le=100,
        description="Borrower age",
    )

    debt: float = Field(
        ge=0,
        description="Existing debt",
    )

    employment: float = Field(
        ge=0,
        description="Years employed",
    )

    balance: float = Field(
        ge=0,
        description="Bank account balance",
    )

    utilization: float = Field(
        ge=0,
        le=1,
        description="Credit utilization ratio",
    )

    accounts: float = Field(
        ge=0,
        description="Number of credit accounts",
    )

    history: float = Field(
        ge=0,
        description="Credit history length",
    )


class ExplainRequest(CreditRequest):
    """Request for a prediction and local model explanation."""

    application_id: str = Field(
        min_length=1,
        description=(
            "Unique identifier for the application "
            "or model observation"
        ),
    )

    top_n: int | Literal["all"] = Field(
        default=4,
        description=(
            "Number of ranked reasons to return, "
            "or 'all'"
        ),
    )

    persist: bool = Field(
        default=True,
        description=(
            "Whether to persist the one-row "
            "explanation record to S3"
        ),
    )

    minimum_contribution: float = Field(
        default=0.0,
        ge=0,
        description=(
            "Minimum SHAP contribution required "
            "for a reason to be returned"
        ),
    )
    
