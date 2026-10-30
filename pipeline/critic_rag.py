from __future__ import annotations
import json, time
from generation.generator import answer_prompt
from generation.claim_extractor import extract_claims
from critic.verifier import critic_prompt, parse_verdicts
from critic.correction import filter_claims

def run_critic_rag(question: str, evidence: list[dict], generator, critic, regenerate=False) -> dict:
    started=time.perf_counter()
    draft=generator.complete(answer_prompt(question,evidence),temperature=0)
    claims=extract_claims(draft["text"])
    if not claims: return {"draft_answer":draft["text"],"answer":draft["text"],"claims":[],"verdicts":[],"status":"no_claims","total_latency_seconds":time.perf_counter()-started}
    review=critic.complete(critic_prompt(question,evidence,draft["text"],claims),temperature=0)
    verdicts=parse_verdicts(review["text"],claims,{c["chunk_id"] for c in evidence})
    correction=filter_claims(claims,verdicts,regenerate)
    return {"draft_answer":draft["text"],"answer":correction["answer"],"claims":claims,"verdicts":verdicts,"correction":correction,"status":"completed","total_latency_seconds":time.perf_counter()-started,"generation_usage":draft.get("usage",{}),"critic_usage":review.get("usage",{})}
