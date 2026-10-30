from __future__ import annotations

def fuse(rankings: list[list[dict]], k: int=60) -> list[dict]:
    scores={}; ranks={}
    for name, ranking in enumerate(rankings):
        for rank,item in enumerate(ranking,1):
            cid=item["chunk_id"]; scores[cid]=scores.get(cid,0)+1/(k+rank); ranks.setdefault(cid,{})[str(name)]=rank
    return [{"chunk_id":cid,"score":score,"source_ranks":ranks[cid]} for cid,score in sorted(scores.items(),key=lambda x:(-x[1],x[0]))]
