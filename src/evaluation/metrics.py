"""Ranking metrics for one or more relevant items per user."""

from __future__ import annotations

import math


def recall_at_k(recommended, relevant, k: int) -> float:
    if k < 1:
        raise ValueError("k must be positive")
    relevant = set(relevant)
    if not relevant:
        return 0.0
    return len(set(recommended[:k]) & relevant) / len(relevant)


def ndcg_at_k(recommended, relevant, k: int) -> float:
    if k < 1:
        raise ValueError("k must be positive")
    relevant = set(relevant)
    if not relevant:
        return 0.0
    dcg = sum(1.0 / math.log2(rank + 2) for rank, item in enumerate(recommended[:k]) if item in relevant)
    ideal = sum(1.0 / math.log2(rank + 2) for rank in range(min(k, len(relevant))))
    return dcg / ideal if ideal else 0.0


def evaluate_rankings(rankings: dict, relevant_by_user: dict, ks=(5, 10, 20)) -> dict:
    users = [user for user, relevant in relevant_by_user.items() if relevant]
    if not users:
        return {f"recall@{k}": 0.0 for k in ks} | {f"ndcg@{k}": 0.0 for k in ks}
    report = {}
    for k in ks:
        report[f"recall@{k}"] = sum(recall_at_k(rankings.get(user, []), relevant_by_user[user], k) for user in users) / len(users)
        report[f"ndcg@{k}"] = sum(ndcg_at_k(rankings.get(user, []), relevant_by_user[user], k) for user in users) / len(users)
    return report