from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

def test_predict_missing_fields():
    request = {
        "income": 90000
    }

    response = client.post(
        "/predict",
        json=request
    )

    assert response.status_code == 422