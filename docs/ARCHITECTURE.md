# Architecture plan

## Phase 0: boundaries and decisions

- **Input:** Kaggle `transactions_train.csv`, `customers.csv`, and `articles.csv`.
- **Learning signal:** purchases are positive implicit-feedback interactions. Validation is the latest configured number of days, preventing future interactions from entering training aggregates.
- **Model:** two independently encoded towers produce same-dimensional user and article vectors. A dot product ranks candidates; sampled non-purchased articles provide negatives.
- **Scale:** Pandas is the default local path. Spark is an opt-in path for distributed aggregation when the data volume warrants its startup and operational cost.
- **Serving:** export article vectors and an ID-aligned FAISS index. The API gets user features/embedding, retrieves candidates, attaches item metadata, and may cache results/features in Redis.
- **Orchestration:** Airflow invokes the same explicit CLI stages used manually. It is optional for local development.
- **Honesty:** no real metrics are reported until trained against the actual dataset. Synthetic records are only plumbing fixtures.

## Request data flow

```text
GET /recommendations/{customer_id}?k=10
 → validate request
 → look up cached user vector/features in Redis (or load local artifacts)
 → encode user if needed
 → query item ANN index
 → filter already purchased items where history is available
 → attach article metadata and cache response
 → return ranked article IDs, scores, and useful attributes
```

## Training data flow

```text
CSV source → schema/type validation → deduplication and chronological split
 → training-only user/item aggregates → positive/negative examples
 → TFRS two-tower optimization → model and vector export → rebuilt ANN index
 → held-out Recall@K and NDCG@K → Redis refresh
```

Artifacts are generated locally and excluded from source control. Production deployment would need atomic/versioned artifact promotion, monitoring, privacy review, and load testing; this student implementation does not claim those capabilities.