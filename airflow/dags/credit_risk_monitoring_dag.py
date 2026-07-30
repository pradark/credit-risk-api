from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

from monitoring.run_monitoring import main


default_args = {
    "owner": "risk-data-science",
    "retries": 2
    "depends_on_past": False
}


with DAG(
    dag_id="credit_risk_monitoring_daily",
    default_args=default_args,
    description="Daily credit risk model monitoring",
    schedule="0 2 * * *",
    start_date=datetime(2026, 7, 30),
    catchup=False,
    tags=[
        "credit-risk",
        "ml-monitoring"
    ]
) as dag:


    run_monitoring = PythonOperator(
        task_id="run_monitoring",
        python_callable=main
    )


    run_monitoring
