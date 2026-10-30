from __future__ import annotations

def summarize_claims(verdicts: list[dict]) -> dict:
    counts={x:0 for x in ("SUPPORTED","UNSUPPORTED","CONTRADICTED")}
    for v in verdicts:
        if v.get("verdict") not in counts: raise ValueError("Unknown verdict")
        counts[v["verdict"]]+=1
    n=sum(counts.values()); bad=counts["UNSUPPORTED"]+counts["CONTRADICTED"]
    return {"claim_count":n,"supported":counts["SUPPORTED"],"unsupported":counts["UNSUPPORTED"],"contradicted":counts["CONTRADICTED"],"hallucination_rate":bad/n if n else None,"faithfulness":counts["SUPPORTED"]/n if n else None,"any_unsupported":bool(bad) if n else None}
