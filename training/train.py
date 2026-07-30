import pandas as pd
import lightgbm as lgb
import joblib

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


# Create sample credit-risk-like data
X, y = make_classification(
    n_samples=5000,
    n_features=10,
    random_state=42
)

X = pd.DataFrame(
    X,
    columns=[
        "income",
        "fico",
        "loan_amount",
        "age",
        "debt",
        "employment",
        "balance",
        "utilization",
        "accounts",
        "history"
    ]
)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


model = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=4
)


model.fit(
    X_train,
    y_train
)


joblib.dump(
    model,
    "models/model.pkl"
)


print("Model saved")