# Local setup

## Python environment and data

Use Python 3.11 for the dependency set. TensorFlow and FAISS wheels vary by operating system; install a platform-compatible TensorFlow build if the pinned range does not support your machine. The Pandas preprocessing, metrics and mocked API tests do not need TensorFlow.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Put Kaggle source CSVs in `data/raw/`, then run:

```powershell
python scripts/prepare_data.py
python scripts/build_features.py
python scripts/train.py
python scripts/export_embeddings.py
python scripts/build_index.py
python scripts/evaluate.py
```

For a software-only smoke dataset, first run `python -m src.data.sample_data --output data/processed/sample`. Point `processed_dir` in a temporary config at that output before building features; synthetic metrics are not model evidence.

## Redis and API

With Docker Desktop installed:

```powershell
docker compose up -d redis
uvicorn api.main:app --reload
```

Check `http://localhost:8000/health` and `http://localhost:8000/docs`. For a complete model response, artifacts including the ANN index and user embeddings must exist. Redis is optional: API cache reads/writes degrade to misses if Redis cannot be reached.

Alternatively `docker compose up --build` starts Redis and the API container. Train/export artifacts on the host first; they are mounted read-only into the container.

## Airflow

Airflow is intentionally not in the default requirements because it has its own constrained dependency set. On Windows, use WSL2. In a separate WSL environment, the following uses the Airflow 3.3.2 constraints for Python 3.11; consult the [official Airflow install guide](https://airflow.apache.org/docs/apache-airflow/stable/installation.html) if that release or Python version has changed:

```bash
python3.11 -m venv .airflow-venv
source .airflow-venv/bin/activate
python -m pip install --upgrade pip
export AIRFLOW_VERSION=3.3.2
export PYTHON_VERSION=3.11
export CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"
pip install "apache-airflow==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}"
export AIRFLOW_HOME="$PWD/airflow"
export PROJECT_ROOT="$PWD"
airflow standalone
```

Install the project dependencies into that same environment (TensorFlow/Pandas versions may need platform-specific selection). The DAG expects the repository, dataset and Python dependencies to be available to its worker; its Bash tasks run the project scripts from the repository root. Visit `http://localhost:8080` and enable `fashion_recommendation_training`. Standalone mode is for local learning, not production.

## Tests and optional Spark

```powershell
pytest
```

Spark is only useful when data and aggregations outgrow local memory. The default project path uses Pandas; `src/data/spark_preprocessing.py` demonstrates distributed transaction counts and requires a Java-compatible Spark environment.