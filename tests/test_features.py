import pandas as pd

from src.features.item_features import build_item_features
from src.features.user_features import build_user_features


def test_user_features_include_cold_start_customer():
    tx = pd.DataFrame({"customer_id": ["u1", "u1"], "article_id": ["a", "b"], "t_dat": pd.to_datetime(["2025-01-01", "2025-01-03"])})
    users = pd.DataFrame({"customer_id": ["u1", "u2"], "age": [20, None]})
    result = build_user_features(tx, users)
    cold = result.set_index("customer_id").loc["u2"]
    assert cold.purchase_count == 0
    assert cold.recency_days == 90


def test_item_popularity_is_train_derived():
    tx = pd.DataFrame({"article_id": ["a", "a"]})
    articles = pd.DataFrame({"article_id": ["a", "b"], "colour_group_name": [None, "Blue"]})
    result = build_item_features(tx, articles).set_index("article_id")
    assert result.loc["a", "purchase_count"] == 2
    assert result.loc["b", "colour_group_name"] == "Blue"