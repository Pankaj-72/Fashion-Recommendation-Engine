# Scratch-from-zero development order

This order follows the runtime dependency chain. Each phase should leave a runnable, understandable increment before the next one is added. Git history should record the state that actually existed at that time; never backdate commits.

## Phase 1 — repository contract and architecture

1. `README.md` — states the problem, intended scope, honest data/model status, and developer entry points. Created first so contributors know what they are building. Depends on no code; later docs and commands must agree with it. Read it before implementation; explain the project as a retrieval system for implicit purchase events.
2. `.gitignore` — keeps private data, credentials, environments, logs, and generated artifacts out of Git. Needed before creating those files. All later local workflows depend on it. Explain why raw data and trained artifacts are reproducible outputs, not source.
3. `requirements.txt` — declares Python dependencies and optional heavier integrations. Created before environment setup; source modules depend on these packages. Explain why Spark and TensorFlow are not required for every local smoke operation.
4. `.env.example` — documents names of local settings without secrets. The API and Redis client consume these settings. Explain that `.env` is local-only.
5. `config/config.yaml` — centralizes paths and model/training knobs. Pipelines consume it; it depends only on documented defaults. Explain how config avoids machine-specific paths.
6. `data/README.md` and `data/{raw,processed}/.gitkeep` — define dataset acquisition and storage boundaries before ingestion code exists. Preprocessing depends on this contract.
7. `docs/ARCHITECTURE.md`, `docs/DEVELOPMENT_ORDER.md` — establish the data flow and explain file dependencies. They are updated as implementation details settle.

## Phase 2 — data contract and ingestion

8. `src/__init__.py`, `src/data/__init__.py` — make application packages importable. All Python modules depend on normal package imports.
9. `src/data/schema.py` — names required source columns and validates inputs. Loaders and tests depend on it; explain required versus optional metadata.
10. `src/data/loader.py` — reads CSVs and normalizes IDs/dates/types. Preprocessing depends on it; explain the three source tables.
11. `src/data/preprocessing.py` — removes invalid/duplicate rows, fills optional values, applies a time-aware split, and persists clean tables. Feature jobs depend on the outputs. Explain leakage-safe chronological validation.
12. `src/data/sample_data.py` — produces deterministic, labeled synthetic data for local plumbing checks. Tests and demos use it; it never substitutes for the Kaggle dataset.
13. `scripts/prepare_data.py` — small CLI wrapper around configured ingestion/preprocessing. Students invoke this rather than editing library internals.

## Phase 3 — feature engineering

14. `src/features/user_features.py` — aggregates purchase counts, recency, frequency and available demographics from training history. Model/API feature preparation depends on it; explain that validation events must not leak into user aggregates.
15. `src/features/item_features.py` — derives popularity and maps article category/product/color metadata. The item tower and response metadata depend on it.
16. `src/features/build_features.py` — coordinates and saves user/item feature tables. Training and serving artifact generation depend on consistent feature definitions.
17. `src/data/spark_preprocessing.py` — optional equivalent distributed aggregation for datasets exceeding local memory. It is added only after Pandas semantics are established; explain when Spark's overhead is justified.

## Phase 4 — model and training

18. `src/models/two_tower.py` — defines user/item encoders and dot-product retrieval task with TFRS. Training depends on its input schema; explain independently computed embeddings and candidate scoring.
19. `src/training/dataset.py` — creates positive examples and sampled non-interacted negatives from training history only. It supplies batches to the model; explain implicit feedback and sampling bias.
20. `src/training/train.py` — loads features, trains reproducibly, and saves model/config metadata. Embedding generation depends on its artifacts.
21. `scripts/train.py` — config-driven command entry point, separate from reusable training functions.

## Phase 5 — evaluation

22. `src/evaluation/metrics.py` — computes Recall@K and NDCG@K for ranked held-out items. The evaluator and report depend on it; explain hit coverage versus rank-sensitive gain.
23. `src/evaluation/evaluate.py` — scores validation users against eligible candidates and writes a report. It follows training and uses only time-held-out positives; explain candidate universe and limitations.
24. `scripts/evaluate.py` — user-facing evaluation command.

## Phase 6 — embedding export and ANN

25. `src/retrieval/embeddings.py` — exports normalized item vectors and aligned article IDs, plus user encodings. Index creation depends on exact ID/vector alignment.
26. `src/retrieval/ann.py` — builds/searches a FAISS inner-product index, with a clear exact-search fallback when FAISS is absent. API service depends on it; explain ANN speed/recall tradeoff and ID mapping.
27. `scripts/build_index.py` — rebuilds index from the current model snapshot.

## Phase 7 — cache/feature store

28. `src/cache/redis_store.py` — encapsulates serialization, namespaced keys, timeouts and missing-key behavior. Serving consumes this abstraction; tests can inject a fake client. Explain Redis is an optimization and cache miss must be handled.

## Phase 8 — API serving

29. `api/schemas/recommendations.py` — defines request/response shape and validation. Route and client depend on it.
30. `api/services/recommender.py` — coordinates user lookup, embeddings, retrieval, metadata and cache. Routes call this single business boundary.
31. `api/routes/recommendations.py` — thin HTTP handler with bounded `k` and explicit unknown-user behavior.
32. `api/main.py` — assembles FastAPI, health check and dependency wiring. Explain how `/docs` derives from schemas.

## Phase 9 — orchestration and local services

33. `airflow/dags/recommendation_pipeline.py` — schedules explicit subprocess stages in dependency order; it calls the same CLI scripts students use. Explain idempotency and artifact versioning.
34. `docker-compose.yml` — provides local Redis and optional API service. Airflow can be installed separately to keep the default setup lightweight.
35. `docs/SETUP.md`, `docs/API.md`, `docs/MODEL.md` — document exact commands, serving contract, model assumptions and metric meanings after those contracts exist.

## Phase 10 — tests and interview guide

36. `tests/test_preprocessing.py`, `tests/test_features.py`, `tests/test_metrics.py`, `tests/test_api.py`, `tests/test_redis_store.py` — protect important contracts with small fixtures and mocked external services. Tests depend on the modules they verify; explain tests prove behavior, not model quality.
37. `docs/INTERVIEW_GUIDE.md` — maps implemented behavior to concise explanations and candid limitations. It should be written from the finished implementation, not claims of scale or quality.
38. Optional frontend — only after API behavior is stable; a small client can then be added without affecting core ML.

## Cross-cutting files created alongside their owning phase

- `pyproject.toml` — Phase 10 test/lint tool settings and repository import path. It is consumed by pytest and Ruff; explain tool configuration is separate from runtime model configuration.
- `Dockerfile`, `docker-compose.yml` — Phase 9 local API/Redis runtime. The Dockerfile packages the API, while Compose wires Redis and a read-only artifact mount. Explain containerization does not train the model.
- `requirements-api.txt`, `.dockerignore` — Phase 9 keep the API image lean and keep local datasets/secrets/virtual environments out of the Docker build context; only serving dependencies are installed in the API image.
- `scripts/build_features.py`, `scripts/train.py`, `scripts/evaluate.py`, `scripts/export_embeddings.py`, `scripts/build_index.py`, `scripts/sync_redis.py` — small phase-specific command wrappers used manually and by the DAG. They depend on corresponding `src/` functions and should not contain model logic.
- `tests/` — Phase 10 behavior checks; expected failures/limitations are visible in the docs. Explain deterministic unit fixtures do not establish recommendation quality.
- `docs/{API,MODEL,SETUP,INTERVIEW_GUIDE}.md` — final documentation phase after actual contracts exist; they depend on the API/model/setup behavior and should be revised when those interfaces change.
- `notebooks/` is intentionally empty: no notebook is required for the runnable pipeline, and exploration notebooks are an optional follow-up rather than required scaffolding.

## Dependency graph

```text
config + data contract
        ↓
Kaggle ingestion → validation / chronological split
        ↓
user + item feature engineering
        ↓
negative sampling → two-tower training
        ↓
Recall@K / NDCG@K evaluation
        ↓
user/item embedding export → FAISS index
        ↓
Redis cache / feature lookup
        ↓
FastAPI recommendation service
        ↓
Airflow schedules the same reproducible pipeline stages
```

Airflow is intentionally last in the runtime chain: orchestration should call working pipeline commands rather than hide or replace them.