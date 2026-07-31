# Credit Risk Prediction API

A production-style machine learning credit risk prediction service built
with **Python**, **FastAPI**, **LightGBM**, **Docker**, **Apache
Airflow**, and **AWS**.

## Overview

This project demonstrates an end-to-end machine learning workflow for
consumer credit risk, from model training through deployment and
monitoring.

### Features

-   Credit risk prediction using a LightGBM model
-   REST API built with FastAPI
-   Pydantic request validation
-   Docker containerization
-   Apache Airflow orchestration
-   Model performance monitoring
-   Population Stability Index (PSI) and performance metrics
-   Unit testing with Pytest
-   AWS deployment artifacts

## Technology Stack

  Category           Technologies
  ------------------ ---------------------------------------
  Language           Python 3.11
  Machine Learning   LightGBM, scikit-learn, pandas, NumPy
  API                FastAPI, Uvicorn
  Workflow           Apache Airflow
  Testing            Pytest
  Containerization   Docker
  Cloud              AWS ECS, CloudWatch, S3

## Repository Structure

``` text
credit-risk-api/
├── app/
├── models/
├── training/
├── tests/
├── dags/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## API Endpoints

### Health Check

``` http
GET /health
```

Response

``` json
{
  "status": "healthy"
}
```

### Predict Default Probability

``` http
POST /predict
```

Example request

``` json
{
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
```

Example response

``` json
{
  "default_probability": 0.12
}
```

## Running Locally

``` bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open:

-   http://localhost:8000/docs
-   http://localhost:8000/redoc

## Run Tests

``` bash
pytest
```

## Docker

``` bash
docker build -t credit-risk-api .
docker run -p 8000:8000 credit-risk-api
```

## MLOps Features

-   Automated model monitoring
-   Performance metric calculations
-   Feature drift readiness
-   Prediction logging
-   CloudWatch integration
-   CI/CD ready

## Future Enhancements

-   GitHub Actions CI/CD
-   MLflow model registry
-   SHAP explainability
-   Batch scoring
-   Amazon ECR deployment
-   Amazon ECS automated deployment

## Author

**Pradeep Arkachar**

Credit Risk Analytics • Machine Learning • FinTech
