from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

from app.monitoring.run_monitoring import run_monitoring


default_args = {
    "owner": "risk-analytics",
    "retries": 2,
}


with DAG(
    dag_id="credit_risk_monitoring",
    default_args=default_args,
    description="Daily credit risk model monitoring",
    schedule="0 2 * * *",
    start_date=datetime(2026, 7, 1),
    catchup=False,
    tags=[
        "credit-risk",
        "ml-monitoring",
    ],
) as dag:

    calculate_feature_psi = PythonOperator(
        task_id="calculate_feature_psi",
        python_callable=run_monitoring,
    )


    calculate_feature_psi