from __future__ import annotations
from retrieval.rrf import fuse

def hybrid_search(question: str, bm25, dense, documents: list[dict], k: int=5, candidate_k: int=50, rrf_constant: int=60, reranker=None) -> list[dict]:
    lexical=bm25.search(question,candidate_k)
    semantic=dense.search(question,candidate_k)
    fused=fuse([lexical,semantic],rrf_constant)[:candidate_k]
    by_id={d["chunk_id"]:d for d in documents}
    candidates=[{**by_id[x["chunk_id"]],"retrieval_score":x["score"],"source_ranks":x["source_ranks"]} for x in fused if x["chunk_id"] in by_id]
    if reranker is not None:
        candidates=reranker.rerank(question,candidates)
    return candidates[:k]
