import joblib
import pandas as pd

from app.config import MODEL_PATH


model = joblib.load(
    MODEL_PATH
)

def predict_probability(features):

    df = pd.DataFrame(
        [features]
    )

    probability = model.predict_proba(df)[0][1]

    return float(probability)