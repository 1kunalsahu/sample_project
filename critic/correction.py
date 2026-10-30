from __future__ import annotations

def filter_claims(claims: list[dict], verdicts: list[dict], regenerate: bool=False) -> dict:
    """No unsupported statement is rewritten from model invention; correction suggestions are returned separately."""
    lookup={v["claim_id"]:v for v in verdicts}; kept=[]; removed=[]; suggestions=[]
    for claim in claims:
        v=lookup[claim["claim_id"]]
        if v["verdict"] == "SUPPORTED": kept.append(claim)
        else:
            removed.append(claim)
            if v.get("correction"): suggestions.append({"claim_id":claim["claim_id"],"suggestion":v["correction"]})
    answer=" ".join(c["text"]+((" "+" ".join(f"[{x}]" for x in c.get("citations",[]))) if c.get("citations") else "") for c in kept)
    return {"claims":kept,"removed_claims":removed,"correction_suggestions":suggestions,"regeneration_requested":regenerate,"answer":answer}
