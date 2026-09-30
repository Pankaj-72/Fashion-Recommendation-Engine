"""Input contracts for the H&M competition tables."""

from __future__ import annotations

from collections.abc import Iterable

TRANSACTION_COLUMNS = {"customer_id", "article_id", "t_dat"}
CUSTOMER_COLUMNS = {"customer_id"}
ARTICLE_COLUMNS = {"article_id"}


def require_columns(frame, required: Iterable[str], table_name: str) -> None:
    """Raise a useful error when a source table is missing required columns."""
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {', '.join(missing)}")