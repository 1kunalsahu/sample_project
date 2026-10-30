from __future__ import annotations

def evidence_recall(retrieved: list[str], relevant: list[str], k: int) -> float | None:
    if not relevant: return None
    return len(set(retrieved[:k]) & set(relevant))/len(set(relevant))

def query_hit_rate(retrieved: list[str], relevant: list[str], k: int) -> float | None:
    if not relevant: return None
    return float(bool(set(retrieved[:k]) & set(relevant)))
