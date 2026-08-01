"""Application configuration loaded from environment variables."""

from pathlib import Path
import os


# -----------------------------
# Application paths
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent


MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        BASE_DIR / "models" / "model.pkl",
    )
)


# -----------------------------
# AWS configuration
# -----------------------------

AWS_REGION = os.getenv(
    "AWS_REGION",
    "us-east-1",
)

AWS_ACCOUNT_ID = os.getenv(
    "AWS_ACCOUNT_ID",
    "",
)

ENVIRONMENT = os.getenv(
    "ENVIRONMENT",
    "development",
)


# -----------------------------
# S3 data locations
# -----------------------------

S3_BUCKET = os.getenv(
    "S3_BUCKET",
    "credit-risk-monitoring-pradark",
)

S3_FEATURE_PREFIX = os.getenv(
    "S3_FEATURE_PREFIX",
    "features",
)

S3_PREDICTION_PREFIX = os.getenv(
    "S3_PREDICTION_PREFIX",
    "predictions",
)

S3_OUTCOME_PREFIX = os.getenv(
    "S3_OUTCOME_PREFIX",
    "outcomes",
)

S3_EXPLANATION_PREFIX = os.getenv(
    "S3_EXPLANATION_PREFIX",
    "explanations",
)

S3_MONITORING_PREFIX = os.getenv(
    "S3_MONITORING_PREFIX",
    "monitoring",
)

S3_ANALYTICS_PREFIX = os.getenv(
    "S3_ANALYTICS_PREFIX",
    "analytics",
)


# -----------------------------
# Model metadata
# -----------------------------

MODEL_VERSION = os.getenv(
    "MODEL_VERSION",
    "credit-risk-model-v1",
)

MODEL_CLASSIFICATION_THRESHOLD = float(
    os.getenv(
        "MODEL_CLASSIFICATION_THRESHOLD",
        "0.50",
    )
)


# -----------------------------
# Monitoring configuration
# -----------------------------

MONITORING_WINDOW_DAYS = int(
    os.getenv(
        "MONITORING_WINDOW_DAYS",
        "1",
    )
)

MIN_PRODUCTION_SAMPLES = int(
    os.getenv(
        "MIN_PRODUCTION_SAMPLES",
        "100",
    )
)

MIN_PERFORMANCE_SAMPLES = int(
    os.getenv(
        "MIN_PERFORMANCE_SAMPLES",
        "100",
    )
)


# -----------------------------
# Drift thresholds
# -----------------------------

PSI_WARNING_THRESHOLD = float(
    os.getenv(
        "PSI_WARNING_THRESHOLD",
        "0.10",
    )
)

PSI_ALERT_THRESHOLD = float(
    os.getenv(
        "PSI_ALERT_THRESHOLD",
        "0.25",
    )
)

PREDICTION_DRIFT_THRESHOLD = float(
    os.getenv(
        "PREDICTION_DRIFT_THRESHOLD",
        "0.10",
    )
)


# -----------------------------
# Model performance thresholds
# -----------------------------

AUC_WARNING_THRESHOLD = float(
    os.getenv(
        "AUC_WARNING_THRESHOLD",
        "0.70",
    )
)

AUC_ALERT_THRESHOLD = float(
    os.getenv(
        "AUC_ALERT_THRESHOLD",
        "0.60",
    )
)

KS_WARNING_THRESHOLD = float(
    os.getenv(
        "KS_WARNING_THRESHOLD",
        "0.30",
    )
)

KS_ALERT_THRESHOLD = float(
    os.getenv(
        "KS_ALERT_THRESHOLD",
        "0.20",
    )
)

CALIBRATION_WARNING_THRESHOLD = float(
    os.getenv(
        "CALIBRATION_WARNING_THRESHOLD",
        "0.05",
    )
)

CALIBRATION_ALERT_THRESHOLD = float(
    os.getenv(
        "CALIBRATION_ALERT_THRESHOLD",
        "0.10",
    )
)

BAD_RATE_WARNING_THRESHOLD = float(
    os.getenv(
        "BAD_RATE_WARNING_THRESHOLD",
        "0.15",
    )
)

BAD_RATE_ALERT_THRESHOLD = float(
    os.getenv(
        "BAD_RATE_ALERT_THRESHOLD",
        "0.25",
    )
)


# -----------------------------
# CloudWatch namespaces
# -----------------------------

CLOUDWATCH_RUNTIME_NAMESPACE = os.getenv(
    "CLOUDWATCH_RUNTIME_NAMESPACE",
    "CreditRiskAPI",
)

CLOUDWATCH_MONITORING_NAMESPACE = os.getenv(
    "CLOUDWATCH_MONITORING_NAMESPACE",
    "CreditRiskModelMonitoring",
)


# -----------------------------
# AWS Glue configuration
# -----------------------------

GLUE_DATABASE_NAME = os.getenv(
    "GLUE_DATABASE_NAME",
    "credit_risk_monitoring",
)


# -----------------------------
# Amazon Athena configuration
# -----------------------------

ATHENA_WORKGROUP = os.getenv(
    "ATHENA_WORKGROUP",
    "primary",
)

ATHENA_OUTPUT_LOCATION = os.getenv(
    "ATHENA_OUTPUT_LOCATION",
    f"s3://{os.getenv('S3_BUCKET', 'credit-risk-monitoring-pradark')}/athena-results/",
)


# -----------------------------
# Amazon QuickSight configuration
# -----------------------------

QUICKSIGHT_DATA_SOURCE_ID = os.getenv(
    "QUICKSIGHT_DATA_SOURCE_ID",
    "credit-risk-athena-datasource",
)

QUICKSIGHT_DATA_SET_PREFIX = os.getenv(
    "QUICKSIGHT_DATA_SET_PREFIX",
    "credit-risk",
)
