import pandas as pd

from monitoring.metrics import calculate_metrics



def test_monitoring_metrics():

    features = pd.DataFrame(
        {
            "fico": [
                750,
                700
            ],
            "income": [
                90000,
                60000
            ],
            "loan_amount": [
                10000,
                20000
            ],
            "utilization": [
                0.15,
                0.40
            ]
        }
    )


    predictions = pd.DataFrame(
        {
            "default_probability": [
                0.02,
                0.25
            ]
        }
    )


    metrics = calculate_metrics(
        features,
        predictions
    )


    assert metrics["prediction_count"] == 2


    assert metrics["pd_low_count"] == 1


    assert metrics["pd_very_high_count"] == 1


    assert metrics["avg_fico"] == 725.0


    assert metrics["avg_income"] == 75000.0
