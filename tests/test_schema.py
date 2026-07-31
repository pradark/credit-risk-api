"""Tests for API request validation."""

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


VALID_FEATURES = {
    "income": 90000,
    "fico": 750,
    "loan_amount": 10000,
    "age": 45,
    "debt": 5000,
    "employment": 10,
    "balance": 15000,
    "utilization": 0.15,
    "accounts": 8,
    "history": 15,
}


def test_predict_missing_fields():
    request = {
        "income": 90000,
    }

    response = client.post(
        "/predict",
        json=request,
    )

    assert response.status_code == 422


def test_explain_requires_application_id():
    response = client.post(
        "/explain",
        json={
            **VALID_FEATURES,
        },
    )

    assert response.status_code == 422


def test_explain_rejects_empty_application_id():
    response = client.post(
        "/explain",
        json={
            "application_id": "",
            **VALID_FEATURES,
        },
    )

    assert response.status_code == 422


def test_explain_rejects_negative_minimum_contribution():
    response = client.post(
        "/explain",
        json={
            "application_id": "application-1",
            "minimum_contribution": -0.01,
            **VALID_FEATURES,
        },
    )

    assert response.status_code == 422