# Credit Risk Prediction API

A production-style machine learning credit risk prediction service built
with **Python 3.11**, **FastAPI**, **LightGBM**, **Docker**, and
**AWS-native monitoring** (CloudWatch, S3, Glue, Athena, QuickSight).

---

## Architecture

```
FastAPI
  -> /predict    -> S3 (prediction Parquet)
  -> /explain    -> S3 (explanation Parquet)
  -> /health
  -> /model-info
  -> CloudWatch runtime metrics

Scheduled monitoring pipeline (run_monitoring_pipeline.py)
  -> Load predictions + outcomes from S3
  -> Join on application_id
  -> Performance metrics (AUC, KS, Gini, calibration, ...)
  -> Calibration monitoring (ECE, MCE)
  -> PSI drift monitoring
  -> Segment metrics
  -> Adverse reason summary
  -> Write partitioned Parquet to S3 analytics/
  -> Publish summary KPIs to CloudWatch
  -> Register partitions in AWS Glue Data Catalog
  -> Query via Amazon Athena
  -> Visualise in Amazon QuickSight
```

---

## Technology Stack

| Category        | Technology                                      |
|-----------------|-------------------------------------------------|
| Language        | Python 3.11                                     |
| Machine learning | LightGBM, scikit-learn, SHAP, pandas, NumPy    |
| API             | FastAPI, Uvicorn, Pydantic                      |
| Storage         | Amazon S3 (Parquet)                             |
| Metrics         | Amazon CloudWatch                               |
| Catalog         | AWS Glue Data Catalog                           |
| Query           | Amazon Athena                                   |
| Dashboards      | Amazon QuickSight                               |
| Explainability  | SHAP                                            |
| Testing         | pytest, unittest.mock                           |
| Packaging       | uv                                              |
| Containerisation| Docker                                          |
| CI/CD           | GitHub Actions                                  |

---

## Setup

### Prerequisites

- Python 3.11
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

```bash
git clone https://github.com/pradark/credit-risk-api.git
cd credit-risk-api
uv sync --dev
```

---

## Training

Trains a LightGBM classifier on synthetic credit risk data and saves the
model and reference dataset.

```bash
uv run python training/train.py
```

Outputs:
- `models/model.pkl`
- `models/metadata.json`
- `monitoring/reference/reference_data.parquet`

---

## FastAPI

### Start the server

```bash
uvicorn app.main:app --reload
```

Docs: http://localhost:8000/docs

### Endpoints

#### GET /health

```json
{"status": "healthy"}
```

#### POST /predict

```json
{
  "income": 90000, "fico": 750, "loan_amount": 10000,
  "age": 45, "debt": 5000, "employment": 10,
  "balance": 15000, "utilization": 0.15,
  "accounts": 8, "history": 15
}
```

Response:

```json
{"default_probability": 0.12}
```

#### POST /explain

Returns prediction, ranked adverse reasons, SHAP contributions, and
compliance flags. Persists the explanation record to S3 when
`persist: true`.

#### GET /model-info

Returns model metadata and governance placeholders.

---

## Docker

```bash
docker build -t credit-risk-api:latest .
docker run --rm -p 8000:8000 \
  -e AWS_REGION=us-east-1 \
  -e S3_BUCKET=my-bucket \
  credit-risk-api:latest
```

Do not bake AWS credentials into the image. Use IAM roles (ECS task
role) or environment variable injection at runtime.

---

## S3 Data Layout

All production data is written as Parquet using Hive-style partitioning.

```
s3://<bucket>/
  predictions/
    dt=YYYY-MM-DD/
      model_version=<version>/
        prediction_<id>.parquet

  outcomes/
    dt=YYYY-MM-DD/
      outcome_<id>.parquet

  explanations/
    dt=YYYY-MM-DD/
      application_id=<id>/
        explanation_<prediction-id>.parquet

  analytics/
    performance_metrics/
      model_version=<version>/
        report_date=YYYY-MM-DD/
          part-<uuid>.parquet

    calibration_metrics/   (same partition scheme)
    psi_metrics/
    segment_metrics/
    adverse_reason_summary/
    pipeline_runs/
```

---

## CloudWatch Metrics

### Runtime namespace: `CreditRiskAPI`

| Metric                  | Description                            |
|-------------------------|----------------------------------------|
| PredictionCount         | Total prediction requests              |
| DefaultProbability      | Predicted PD value                     |
| HighRiskPredictionCount | Predictions above 0.75 threshold       |
| PredictionErrorCount    | Failed prediction requests             |
| ExplanationCount        | Successful explanation requests        |
| ExplanationErrorCount   | Failed explanation requests            |

### Monitoring namespace: `CreditRiskModelMonitoring`

| Metric                    | Description                          |
|---------------------------|--------------------------------------|
| AUC                       | Area under ROC curve                 |
| KS                        | Kolmogorov-Smirnov statistic         |
| Gini                      | Gini coefficient (2×AUC − 1)         |
| BadRate                   | Observed default rate                |
| AveragePredictedPD        | Mean predicted probability           |
| ExpectedCalibrationError  | Population-weighted ECE              |
| MaximumCalibrationError   | Worst-band absolute calibration gap  |
| MaximumPSI                | Largest feature PSI                  |
| WarningFeatureCount       | Features with PSI 0.10–0.25          |
| AlertFeatureCount         | Features with PSI > 0.25             |
| PipelineFailure           | Pipeline failure count               |

---

## Performance Monitoring

```bash
uv run python scripts/run_monitoring_pipeline.py \
  --input monitoring/performance_joined.parquet \
  --reference-input monitoring/reference/reference_data.parquet \
  --output-bucket credit-risk-monitoring-pradark \
  --analytics-prefix analytics \
  --model-version credit-risk-model-v1 \
  --environment development \
  --publish-to-cloudwatch \
  --write-to-s3
```

The pipeline calculates:
- AUC, KS, Gini, bad rate, average predicted PD
- Calibration metrics (ECE, MCE) by score band
- PSI for each feature
- Segment metrics by any configured segment column
- Adverse reason frequency and contribution

---

## Calibration Monitoring

Calibration monitoring compares average predicted PD against observed
default rate by score band and calculates Expected Calibration Error
(ECE) and Maximum Calibration Error (MCE).

**Calibration monitoring does not automatically recalibrate
probabilities.** All model changes require investigation, validation,
governance approval, versioning, controlled deployment, and
post-deployment monitoring.

---

## PSI (Population Stability Index)

PSI status thresholds:

| Range        | Status  |
|--------------|---------|
| < 0.10       | stable  |
| 0.10 – 0.25  | warning |
| ≥ 0.25       | alert   |

---

## AWS Glue Data Catalog

The `create_all_tables` function in `app/monitoring/glue_catalog.py`
creates or updates Glue external tables for all six monitoring datasets.
Tables are partitioned by `model_version` and `report_date`.

```python
from app.monitoring.glue_catalog import create_all_tables

create_all_tables(
    database_name="credit_risk_monitoring",
    bucket="credit-risk-monitoring-pradark",
    analytics_prefix="analytics",
)
```

---

## Amazon Athena

The `app/monitoring/athena_validator.py` module provides functions to
run queries, poll status, and validate that monitoring tables are
populated.

Example queries are available in `athena_validator.EXAMPLE_QUERIES`.

---

## CloudWatch Dashboards and Alarms

```python
from app.monitoring.cloudwatch_dashboard import put_dashboard, put_alarms

put_dashboard(model_version="credit-risk-model-v1", environment="production")

put_alarms(
    model_version="credit-risk-model-v1",
    environment="production",
    sns_topic_arn="arn:aws:sns:us-east-1:123456789012:credit-risk-alerts",
)
```

The dashboard covers: AUC/KS/Gini trend, calibration ECE/MCE, PSI,
prediction volume, default count, error rates, and pipeline failures.

Alarms are configured for: AUC (warning/alert), KS (warning/alert),
ECE (warning/alert), MCE, PSI (warning/alert), prediction errors, and
pipeline failures.

**Alarms must never trigger automatic model changes, retraining,
recalibration, or threshold updates.**

---

## Amazon QuickSight

Six recommended dashboards are defined in
`app/monitoring/quicksight_assets.py`:

1. Executive Model Health — AUC, KS, Gini, bad rate, ECE, MCE, PSI KPIs
2. Calibration — expected vs actual by band, ECE/MCE trend
3. Drift — PSI by feature, stable/warning/alert counts
4. Segment Performance — bad rate and calibration by segment
5. Adverse Reasons — reason frequency, selection rate, SHAP contribution
6. Governance — model version, pipeline run history, threshold breaches

QuickSight resources are parameterized because account IDs, namespaces,
and permissions vary by AWS account. Use `create_athena_data_source` and
`build_all_dataset_inputs` to provision data sources and datasets.

```python
from app.monitoring.quicksight_assets import (
    create_athena_data_source,
    get_dashboard_specifications,
)
```

---

## Governance Controls

| Field                  | Description                                   |
|------------------------|-----------------------------------------------|
| model_version          | Version identifier                            |
| governance_status      | development / validation / pending_approval / approved / deployed / retired |
| approval_reference     | Governance approval ticket reference          |
| approval_date          | Date of formal governance approval            |
| deployment_date        | Production deployment date                    |
| classification_threshold | Probability threshold used for decisions    |

The monitoring pipeline does not change governance status automatically.
Model version changes, threshold updates, and status transitions require
explicit approval and a controlled deployment process.

---

## Testing

```bash
uv run pytest
```

Test coverage includes:
- API endpoints (health, predict, explain, model-info)
- Pydantic schema validation
- Explainability (SHAP, reason selector, feature registry)
- S3 persistence (predictions, explanations)
- Performance metrics (all functions)
- Calibration monitoring (ECE, MCE)
- PSI drift monitoring
- Segment metrics
- Adverse reason summary
- CloudWatch publishing
- Parquet serialisation and partitioned S3 keys
- BI dataset writer (all validation and write paths)
- Monitoring pipeline (end-to-end, S3, CloudWatch, PSI, segments)
- Glue table definitions (create, update, partitions)
- Athena query polling (SUCCEEDED, FAILED, CANCELLED, timeout)
- CloudWatch dashboard JSON
- CloudWatch alarm definitions
- QuickSight asset definitions and dashboard specs
- No HTML in production workflow (AST and runtime checks)
- No automatic recalibration

---

## CI/CD

GitHub Actions runs on every pull request and push to `main`:

- Triggers: `pull_request` to `main`, `push` to `main`
- Installs `uv` and Python 3.11
- Runs `uv sync --dev`
- Runs `uv run pytest`

CI does not require real AWS credentials. All AWS calls are mocked using
`unittest.mock.Mock` or `botocore.stub.Stubber`.

---

## IAM Permissions

The ECS task role (or executing principal) requires:

```json
{
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:PutObject", "s3:GetObject", "s3:ListBucket",
        "cloudwatch:PutMetricData",
        "cloudwatch:PutDashboard", "cloudwatch:GetDashboard",
        "cloudwatch:PutMetricAlarm", "cloudwatch:DescribeAlarms"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "glue:CreateDatabase", "glue:GetDatabase",
        "glue:CreateTable", "glue:UpdateTable", "glue:GetTable",
        "glue:BatchCreatePartition", "glue:GetPartitions"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "athena:StartQueryExecution",
        "athena:GetQueryExecution",
        "athena:GetQueryResults"
      ],
      "Resource": "*"
    }
  ]
}
```

Do not use broad `AdministratorAccess` in production.

---

## Environment Variables

| Variable                      | Default                          |
|-------------------------------|----------------------------------|
| MODEL_PATH                    | models/model.pkl                 |
| AWS_REGION                    | us-east-1                        |
| AWS_ACCOUNT_ID                | (required for QuickSight)        |
| S3_BUCKET                     | credit-risk-monitoring-pradark   |
| S3_ANALYTICS_PREFIX           | analytics                        |
| MODEL_VERSION                 | credit-risk-model-v1             |
| MODEL_CLASSIFICATION_THRESHOLD| 0.50                             |
| PSI_WARNING_THRESHOLD         | 0.10                             |
| PSI_ALERT_THRESHOLD           | 0.25                             |
| AUC_WARNING_THRESHOLD         | 0.70                             |
| AUC_ALERT_THRESHOLD           | 0.60                             |
| CALIBRATION_WARNING_THRESHOLD | 0.05                             |
| CALIBRATION_ALERT_THRESHOLD   | 0.10                             |
| CLOUDWATCH_RUNTIME_NAMESPACE  | CreditRiskAPI                    |
| CLOUDWATCH_MONITORING_NAMESPACE| CreditRiskModelMonitoring       |
| GLUE_DATABASE_NAME            | credit_risk_monitoring           |
| ATHENA_WORKGROUP              | primary                          |
| ENVIRONMENT                   | development                      |

---

## Limitations and Future Roadmap

**Current limitations:**
- Calibration monitoring does not automatically recalibrate. Any
  calibration change requires governance approval.
- Model changes require governance approval — the pipeline never
  retrains or redeploys automatically.
- Adverse action reason wording is demonstration content requiring
  legal and compliance approval before production use.
- AWS deployment requires appropriate IAM permissions and AWS account
  setup.
- QuickSight dashboard creation is parameterized — account-specific
  configuration (namespace, principal ARN, edition) must be supplied.

**Future roadmap:**
- MLflow model registry integration
- Automated Glue crawler for partition discovery
- QuickSight embedding in internal tooling
- Batch scoring support
- Drift alerting with SNS → PagerDuty integration
- Multi-model version comparison in Athena

---

## Author

**Pradeep Arkachar**

Credit Risk Analytics · Machine Learning · FinTech
