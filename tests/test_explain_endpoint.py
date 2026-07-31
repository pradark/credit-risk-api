"""Tests for the FastAPI /explain endpoint."""

from fastapi.testclient import TestClient

from app import main


client = TestClient(
    main.app
)


VALID_REQUEST = {
    "application_id": "application-1001",
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
    "top_n": 4,
    "persist": False,
    "minimum_contribution": 0.0,
}


def test_explain_returns_prediction_and_reasons(
    monkeypatch,
) -> None:
    """A valid request should return an explanation response."""

    captured = {}

    def fake_explain_prediction(
        **kwargs,
    ):
        captured.update(
            kwargs
        )

        return {
            "prediction_id": "prediction-1001",
            "application_id": "application-1001",
            "prediction_timestamp": (
                "2026-07-31T17:00:00+00:00"
            ),
            "model_version": "credit-risk-model-v1",
            "default_probability": 0.81,
            "decision_threshold": 0.50,
            "decision": "decline",
            "requested_top_n": "4",
            "total_selected_reasons": 2,
            "reasons": [
                {
                    "reason_rank": 1,
                    "attribute_name": "utilization",
                    "display_name": "Credit utilization",
                    "attribute_value": 0.15,
                    "average_attribute_value": 0.10,
                    "attribute_impact": 0.40,
                    "absolute_impact": 0.40,
                    "risk_vs_typical": "more",
                    "value_vs_typical": "more",
                    "reason_code": "CREDIT_UTILIZATION",
                    "adverse_reason": (
                        "Credit utilization increased "
                        "the modeled risk"
                    ),
                    "compliance_review_required": False,
                },
                {
                    "reason_rank": 2,
                    "attribute_name": "loan_amount",
                    "display_name": "Requested loan amount",
                    "attribute_value": 10000,
                    "average_attribute_value": 8000,
                    "attribute_impact": 0.25,
                    "absolute_impact": 0.25,
                    "risk_vs_typical": "more",
                    "value_vs_typical": "more",
                    "reason_code": "REQUESTED_LOAN_AMOUNT",
                    "adverse_reason": (
                        "Requested loan amount increased "
                        "the modeled risk"
                    ),
                    "compliance_review_required": False,
                },
            ],
            "all_feature_contributions": [],
            "compliance_review_required": False,
            "compliance_flags": [],
            "s3_key": None,
        }

    monkeypatch.setattr(
        main,
        "explain_prediction",
        fake_explain_prediction,
    )

    monkeypatch.setattr(
        main,
        "get_explanation_background",
        lambda: object(),
    )

    monkeypatch.setattr(
        main,
        "put_metric",
        lambda *args, **kwargs: None,
    )

    response = client.post(
        "/explain",
        json=VALID_REQUEST,
    )

    assert response.status_code == 200

    result = response.json()

    assert result[
        "application_id"
    ] == "application-1001"

    assert result[
        "default_probability"
    ] == 0.81

    assert result[
        "decision"
    ] == "decline"

    assert result[
        "total_selected_reasons"
    ] == 2

    assert len(
        result[
            "reasons"
        ]
    ) == 2

    assert captured[
        "top_n"
    ] == 4

    assert captured[
        "persistence"
    ] == "none"

    assert captured[
        "minimum_contribution"
    ] == 0.0


def test_explain_supports_all_reasons(
    monkeypatch,
) -> None:
    """The endpoint should pass top_n='all' to the service."""

    captured = {}

    def fake_explain_prediction(
        **kwargs,
    ):
        captured.update(
            kwargs
        )

        return {
            "prediction_id": "prediction-1002",
            "application_id": "application-1001",
            "prediction_timestamp": (
                "2026-07-31T17:00:00+00:00"
            ),
            "model_version": "credit-risk-model-v1",
            "default_probability": 0.40,
            "decision_threshold": 0.50,
            "decision": "approve",
            "requested_top_n": "all",
            "total_selected_reasons": 1,
            "reasons": [],
            "all_feature_contributions": [],
            "compliance_review_required": False,
            "compliance_flags": [],
            "s3_key": None,
        }

    monkeypatch.setattr(
        main,
        "explain_prediction",
        fake_explain_prediction,
    )

    monkeypatch.setattr(
        main,
        "get_explanation_background",
        lambda: object(),
    )

    monkeypatch.setattr(
        main,
        "put_metric",
        lambda *args, **kwargs: None,
    )

    request = {
        **VALID_REQUEST,
        "top_n": "all",
    }

    response = client.post(
        "/explain",
        json=request,
    )

    assert response.status_code == 200
    assert captured["top_n"] == "all"


def test_explain_enables_s3_persistence(
    monkeypatch,
) -> None:
    """persist=true should request S3 persistence."""

    captured = {}

    def fake_explain_prediction(
        **kwargs,
    ):
        captured.update(
            kwargs
        )

        return {
            "prediction_id": "prediction-1003",
            "application_id": "application-1001",
            "prediction_timestamp": (
                "2026-07-31T17:00:00+00:00"
            ),
            "model_version": "credit-risk-model-v1",
            "default_probability": 0.81,
            "decision_threshold": 0.50,
            "decision": "decline",
            "requested_top_n": "4",
            "total_selected_reasons": 0,
            "reasons": [],
            "all_feature_contributions": [],
            "compliance_review_required": False,
            "compliance_flags": [],
            "s3_key": (
                "explanations/"
                "dt=2026-07-31/"
                "application_id=application-1001/"
                "explanation_prediction-1003.parquet"
            ),
        }

    monkeypatch.setattr(
        main,
        "explain_prediction",
        fake_explain_prediction,
    )

    monkeypatch.setattr(
        main,
        "get_explanation_background",
        lambda: object(),
    )

    monkeypatch.setattr(
        main,
        "put_metric",
        lambda *args, **kwargs: None,
    )

    request = {
        **VALID_REQUEST,
        "persist": True,
    }

    response = client.post(
        "/explain",
        json=request,
    )

    assert response.status_code == 200
    assert captured["persistence"] == "s3"

    assert response.json()[
        "s3_key"
    ] is not None


def test_explain_returns_500_when_service_fails(
    monkeypatch,
) -> None:
    """Unexpected explanation failures should return HTTP 500."""

    def fake_explain_prediction(
        **kwargs,
    ):
        raise RuntimeError(
            "SHAP failed"
        )

    monkeypatch.setattr(
        main,
        "explain_prediction",
        fake_explain_prediction,
    )

    monkeypatch.setattr(
        main,
        "get_explanation_background",
        lambda: object(),
    )

    monkeypatch.setattr(
        main,
        "put_metric",
        lambda *args, **kwargs: None,
    )

    response = client.post(
        "/explain",
        json=VALID_REQUEST,
    )

    assert response.status_code == 500

    assert response.json() == {
        "detail": "Explanation failed",
    }
