from __future__ import annotations

def validate_citations(claims: list[dict], evidence: list[dict]) -> list[dict]:
    ids={c["chunk_id"] for c in evidence}; out=[]
    for claim in claims:
        cites=claim.get("citations",[])
        out.append({"claim_id":claim["claim_id"],"citation_ids":cites,"invalid_citation_ids":[x for x in cites if x not in ids],"has_citation":bool(cites)})
    return out
