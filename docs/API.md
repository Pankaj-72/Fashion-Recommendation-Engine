# API

Start with `uvicorn api.main:app --reload`. OpenAPI documentation is available at `/docs`.

## `GET /health`

Returns `status`, whether model/index artifacts are ready, and indexed item count. Health is process/readiness information, not a quality signal.

## `GET /recommendations/{user_id}?k=10`

`user_id` is the Kaggle customer ID string. `k` is an integer from 1 through 100.

```json
{
  "customer_id": "customer_001",
  "recommendations": [
    {"article_id": "0000001001", "score": 0.72, "metadata": {"colour_group_name": "Black"}}
  ],
  "source": "ann"
}
```

The score is the embedding similarity, not a calibrated purchase probability. `404` indicates an unknown user without usable cold-start metadata; `422` indicates invalid `k`; `503` indicates missing model/index/user-vector artifacts. Known users without history can receive popularity-based cold-start suggestions when item features are available. Seen recent articles are filtered where history is available. Exact response content depends on trained local artifacts and is not fabricated here.
