from pathlib import Path
import os


# -----------------------------
# Application paths
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent


MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        BASE_DIR / "models" / "model.pkl"
    )
)


# -----------------------------
# AWS Configuration
# -----------------------------

AWS_REGION = os.getenv(
    "AWS_REGION",
    "us-east-1"
)


S3_BUCKET = os.getenv(
    "S3_BUCKET",
    "credit-risk-monitoring-pradark"
)


# -----------------------------
# S3 Data Locations
# -----------------------------

S3_FEATURE_PREFIX = os.getenv(
    "S3_FEATURE_PREFIX",
    "features"
)


S3_PREDICTION_PREFIX = os.getenv(
    "S3_PREDICTION_PREFIX",
    "predictions"
)


S3_MONITORING_PREFIX = os.getenv(
    "S3_MONITORING_PREFIX",
    "monitoring"
)


# -----------------------------
# Monitoring Configuration
# -----------------------------

MONITORING_WINDOW_DAYS = int(
    os.getenv(
        "MONITORING_WINDOW_DAYS",
        "1"
    )
)


# Metrics thresholds

PSI_WARNING_THRESHOLD = float(
    os.getenv(
        "PSI_WARNING_THRESHOLD",
        "0.10"
    )
)


PSI_ALERT_THRESHOLD = float(
    os.getenv(
        "PSI_ALERT_THRESHOLD",
        "0.25"
    )
)


PREDICTION_DRIFT_THRESHOLD = float(
    os.getenv(
        "PREDICTION_DRIFT_THRESHOLD",
        "0.10"
    )
)

MIN_PRODUCTION_SAMPLES = int(
    os.getenv(
        "MIN_PRODUCTION_SAMPLES",
        "100"
    )
)