import pandas as pd

from src.data.preprocessing import chronological_split, clean_tables


def test_cleaning_normalizes_ids_removes_duplicates_and_bad_dates():
    tx = pd.DataFrame({"customer_id": [" c1 ", " c1 ", None], "article_id": ["12", "12", "13"], "t_dat": ["2025-01-01", "2025-01-01", "bad"]})
    customers = pd.DataFrame({"customer_id": ["c1", "c1"]})
    articles = pd.DataFrame({"article_id": ["12"]})
    clean, users, items = clean_tables(tx, customers, articles)
    assert len(clean) == 1
    assert clean.iloc[0].article_id == "0000000012"
    assert len(users) == len(items) == 1


def test_split_is_chronological():
    tx = pd.DataFrame({"customer_id": ["u"] * 4, "article_id": list("abcd"), "t_dat": pd.date_range("2025-01-01", periods=4)})
    train, valid = chronological_split(tx, validation_days=2)
    assert train.t_dat.max() < valid.t_dat.min()
    assert len(valid) == 2