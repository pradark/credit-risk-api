from pydantic import BaseModel, Field


class CreditRequest(BaseModel):

    income: float = Field(
        gt=0,
        description="Annual income"
    )

    fico: float = Field(
        ge=300,
        le=850,
        description="FICO credit score"
    )

    loan_amount: float = Field(
        gt=0,
        description="Requested loan amount"
    )

    age: float = Field(
        ge=18,
        le=100,
        description="Borrower age"
    )

    debt: float = Field(
        ge=0,
        description="Existing debt"
    )

    employment: float = Field(
        ge=0,
        description="Years employed"
    )

    balance: float = Field(
        ge=0,
        description="Bank account balance"
    )

    utilization: float = Field(
        ge=0,
        le=1,
        description="Credit utilization ratio"
    )

    accounts: float = Field(
        ge=0,
        description="Number of credit accounts"
    )

    history: float = Field(
        ge=0,
        description="Credit history length"
    )