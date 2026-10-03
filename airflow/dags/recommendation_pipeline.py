"""Small Airflow DAG that invokes the same explicit local CLI stages."""

import os
from datetime import datetime

try:
    from airflow.sdk import DAG
except ImportError:  # Airflow 2
    from airflow import DAG
try:
    from airflow.providers.standard.operators.bash import BashOperator
except ImportError:  # Airflow 2 provider path
    from airflow.operators.bash import BashOperator

with DAG(
    dag_id="fashion_recommendation_training",
    description="Prepare data, train/evaluate, and refresh retrieval artifacts",
    start_date=datetime(2025, 1, 1),
    schedule="@weekly",
    catchup=False,
    tags=["recommendations", "student-project"],
) as dag:
    project_root = os.getenv("PROJECT_ROOT", os.getcwd())
    def script_task(task_id, script):
        return BashOperator(task_id=task_id, bash_command=f"python scripts/{script}.py", cwd=project_root)

    prepare = script_task("prepare_data", "prepare_data")
    features = script_task("build_features", "build_features")
    train = script_task("train_model", "train")
    export = script_task("export_embeddings", "export_embeddings")
    index = script_task("build_ann_index", "build_index")
    evaluate = script_task("evaluate", "evaluate")
    redis = script_task("refresh_redis", "sync_redis")
    prepare >> features >> train >> export >> index >> evaluate >> redis