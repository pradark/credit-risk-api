from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_predict():

    request = {
        "income": 90000,
        "fico": 750,
        "loan_amount": 10000,
        "age": 45,
        "debt": 5000,
        "employment": 10,
        "balance": 15000,
        "utilization": 0.15,
        "accounts": 8,
        "history": 15
    }

    response = client.post(
        "/predict",
        json=request
    )

    assert response.status_code == 200

    result = response.json()

    assert "default_probability" in result

    probability = result["default_probability"]

    assert 0 <= probability <= 1


def test_invalid_fico():

    request = {
        "income": 90000,
        "fico": 950,
        "loan_amount": 10000,
        "age": 45,
        "debt": 5000,
        "employment": 10,
        "balance": 15000,
        "utilization": 0.15,
        "accounts": 8,
        "history": 15
    }

    response = client.post(
        "/predict",
        json=request
    )

    assert response.status_code == 422


def test_invalid_income():

    request = {
        "income": -50000,
        "fico": 750,
        "loan_amount": 10000,
        "age": 45,
        "debt": 5000,
        "employment": 10,
        "balance": 15000,
        "utilization": 0.15,
        "accounts": 8,
        "history": 15
    }

    response = client.post(
        "/predict",
        json=request
    )

    assert response.status_code == 422


def test_invalid_utilization():

    request = {
        "income": 90000,
        "fico": 750,
        "loan_amount": 10000,
        "age": 45,
        "debt": 5000,
        "employment": 10,
        "balance": 15000,
        "utilization": 1.5,
        "accounts": 8,
        "history": 15
    }

    response = client.post(
        "/predict",
        json=request
    )

    assert response.status_code == 422