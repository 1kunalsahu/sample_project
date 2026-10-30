from __future__ import annotations
import time
from generation.generator import answer_prompt

def run_baseline(question: str, evidence: list[dict], generator) -> dict:
    started=time.perf_counter(); result=generator.complete(answer_prompt(question,evidence),temperature=0)
    return {"draft_answer":result["text"],"answer":result["text"],"model":result["model"],"usage":result["usage"],"generation_latency_seconds":result["latency_seconds"],"total_latency_seconds":time.perf_counter()-started}
