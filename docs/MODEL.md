# Model, training and evaluation

## Two towers

The model maps a customer and an article separately into the same `embedding_dim` vector space. Each tower uses a categorical ID lookup/embedding plus learned side-category embeddings. The user tower also receives a small numeric vector (age when present, purchase count, purchase frequency and days since last purchase), plus categorical last-purchase weekday/month as lightweight time context. The article tower consumes product type/group, color and index group when present. A dense projection produces L2-normalized vectors; their dot product is a cosine-like relevance score.

This architecture lets article vectors be computed before a request and searched repeatedly. It is a retrieval model, so a production ranking stage could later add richer cross-features.

## Interaction labels and negative sampling

Purchases are implicit positive feedback; absence of a purchase is not proof of dislike. The training builder deduplicates positive customer/article pairs and samples up to a configured number of articles that customer did not purchase from the **training item universe**. The binary cross-entropy on the two-tower dot product distinguishes observed pairs from sampled pairs. This is simple and explainable, but sampled negatives may include items the customer never saw and the objective can be sensitive to the sampling distribution.

Validation interactions are not used for negative construction, user aggregates, or model fitting. The split is chronological by latest N calendar days. Very short datasets may yield an empty training split; increase data or adjust `validation_days` rather than silently mixing time periods.

## Training and artifacts

Run `python scripts/prepare_data.py`, `python scripts/build_features.py`, and `python scripts/train.py`. Weights and metadata are written under the configured artifact directory, ignored by Git. Then `python scripts/export_embeddings.py` generates article and customer vectors; `python scripts/build_index.py` builds the candidate index. Finally `python scripts/evaluate.py` uses ANN vectors when exported vectors exist, otherwise reports a popularity baseline. The synthetic fixture validates plumbing only and must not be presented as a real evaluation.

## Evaluation metrics

- **Recall@K**: fraction of a user's held-out relevant articles that appear in the first K recommendations. It measures relevant-item coverage.
- **NDCG@K**: discounted gain of relevant articles, normalized by the best possible ordering. Relevant items near the top count more.

Both are macro-averaged over validation users with at least one held-out positive. The evaluator reports the train-popularity baseline before model vectors exist, then evaluates two-tower ANN candidates after vector export, filtering articles already purchased in training. Candidate universe and filtering policy should be stated with every reported metric. No benchmark result is supplied by this repository.

## ANN retrieval

Brute-force scoring costs approximately O(number of candidate articles × embedding dimension) per user. FAISS inner-product indexing speeds up retrieval as the catalog grows. The current `IndexFlatIP` is exact (the interface permits swapping to approximate FAISS index types later); the NumPy fallback is exact too. Both provide a useful local baseline, not proof of ANN latency gains. Index row numbers map through the separately saved ordered `item_ids.json` list to article IDs. Vector and ID files must be rebuilt together.