
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import yaml

from ingestion.fiqa import load_records
from retrieval.bm25 import BM25
from retrieval.dense_search import DenseSearch
from retrieval.hybrid import hybrid_search
from retrieval.reranker import CrossEncoderReranker
from generation.generator import create_chat_model, answer_prompt
from generation.claim_extractor import extract_claims
from critic.verifier import review_claims
from critic.correction import filter_claims


def main():
    ap = argparse.ArgumentParser(
        description="Run FiQA RAG with a critic pass and save auditable per-question/claim records"
    )
    ap.add_argument(
        "--condition",
        choices=["baseline", "critic"],
        default="critic",
    )
    ap.add_argument("--data", default="data/raw/fiqa/fiqa")
    ap.add_argument("--split", default="test")
    ap.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Pilot limit; use 0 to process the full split",
    )
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument(
        "--output",
        help="Optional JSONL output path (must not already exist)",
    )
    ap.add_argument("--models-config", default="configs/models.yaml")
    ap.add_argument("--retrieval-config", default="configs/retrieval.yaml")
    ap.add_argument("--candidate-k", type=int)
    ap.add_argument("--dense-model")
    ap.add_argument(
        "--reranker-model",
        help="Set to 'none' to disable the configured reranker",
    )

    args = ap.parse_args()

    if args.k < 1 or args.limit < 0:
        ap.error("k must be positive and limit cannot be negative")

    model_config = (
        yaml.safe_load(
            Path(args.models_config).read_text(encoding="utf-8")
        )
        or {}
    )
    retrieval_config = (
        yaml.safe_load(
            Path(args.retrieval_config).read_text(encoding="utf-8")
        )
        or {}
    )

    candidate_k = args.candidate_k or int(
        retrieval_config.get("candidate_k", 50)
    )
    dense_model = args.dense_model or retrieval_config.get(
        "dense_model",
        "sentence-transformers/all-MiniLM-L6-v2",
    )

    reranker_name = (
        args.reranker_model
        if args.reranker_model is not None
        else retrieval_config.get("reranker_model")
    )

    if reranker_name and str(reranker_name).lower() == "none":
        reranker_name = None

    if candidate_k < args.k:
        ap.error("candidate-k must be >= k")

    backend = (
        os.getenv("CRITICRAG_BACKEND")
        or model_config.get("backend", "ollama")
    )
    base_url = os.getenv("OLLAMA_HOST") or model_config.get("base_url")

    generator = create_chat_model(
        backend,
        os.getenv("CRITICRAG_MODEL")
        or model_config.get("generator"),
        base_url,
    )
    critic = create_chat_model(
        backend,
        os.getenv("CRITICRAG_CRITIC_MODEL")
        or model_config.get("critic")
        or generator.model,
        base_url,
    )
    evaluator = create_chat_model(
        backend,
        os.getenv("CRITICRAG_EVALUATOR_MODEL")
        or model_config.get("evaluator")
        or critic.model,
        base_url,
    )

    root = Path(args.data)
    documents = []

    with (root / "corpus.jsonl").open(encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            documents.append(
                {
                    "chunk_id": item["_id"],
                    "text": (
                        item.get("title", "")
                        + " "
                        + item.get("text", "")
                    ).strip(),
                }
            )

    records = load_records(root, args.split)
    if args.limit:
        records = records[:args.limit]

    retriever = BM25(documents)
    dense = DenseSearch(
        documents,
        dense_model,
        retrieval_config.get("embedding_cache", ".cache/criticrag"),
    )
    reranker = (
        CrossEncoderReranker(reranker_name)
        if reranker_name
        else None
    )

    retrieval_metadata = {
        "method": "BM25 + dense + RRF"
        + (" + cross-encoder reranker" if reranker else ""),
        "dense_model": dense_model,
        "reranker_model": reranker_name,
        "candidate_k": candidate_k,
        "k": args.k,
        "rrf_constant": int(
            retrieval_config.get("rrf_constant", 60)
        ),
    }

    output = (
        Path(args.output)
        if args.output
        else Path("results/raw")
        / (
            f"{args.condition}_fiqa_{args.split}_"
            f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.jsonl"
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("x", encoding="utf-8") as sink:
        for index, record in enumerate(records, 1):
            ranked = hybrid_search(
                record["question"],
                retriever,
                dense,
                documents,
                args.k,
                candidate_k,
                int(retrieval_config.get("rrf_constant", 60)),
                reranker,
            )
            evidence = ranked

            try:
                generated = generator.complete(
                    answer_prompt(record["question"], evidence),
                    temperature=0,
                )
                final_answer = generated["text"]

                claims = extract_claims(generated["text"])
                critic_response = None
                critic_latency = None

                if args.condition == "critic" and claims:
                    reviewed = review_claims(
                        critic,
                        record["question"],
                        evidence,
                        generated["text"],
                        claims,
                    )
                    critic_response = reviewed["raw_responses"]
                    critic_latency = reviewed["latency_seconds"]
                    action_verdicts = reviewed["verdicts"]

                    final_answer = filter_claims(
                        claims,
                        action_verdicts,
                        regenerate=False,
                    )["answer"]

                # Evaluate the final answer independently of the intervention critic.
                evaluated_claims = extract_claims(final_answer)
                evaluation_response = None
                evaluation_latency = None
                claim_rows = []

                if evaluated_claims:
                    evaluated = review_claims(
                        evaluator,
                        record["question"],
                        evidence,
                        final_answer,
                        evaluated_claims,
                    )
                    evaluation_response = evaluated["raw_responses"]
                    evaluation_latency = evaluated["latency_seconds"]
                    evaluation_verdicts = evaluated["verdicts"]

                    # Index only verdicts that contain a claim ID.
                    verdict_by_id = {
                        verdict["claim_id"]: verdict
                        for verdict in evaluation_verdicts
                        if verdict.get("claim_id")
                    }

                    # Preserve every extracted claim, even if the evaluator
                    # did not return a matching verdict.
                    for claim in evaluated_claims:
                        verdict = verdict_by_id.get(claim["claim_id"])

                        if verdict is None:
                            claim_rows.append(
                                {
                                    **claim,
                                    "verdict": "INVALID",
                                    "supporting_evidence_ids": [],
                                    "rationale": (
                                        "Evaluator did not return a verdict "
                                        "for this claim."
                                    ),
                                }
                            )
                            continue

                        claim_rows.append(
                            {
                                **claim,
                                "verdict": verdict.get(
                                    "verdict", "INVALID"
                                ),
                                "supporting_evidence_ids": (
                                    verdict.get("evidence_chunk_ids", [])
                                ),
                                "rationale": verdict.get("rationale", ""),
                            }
                        )

                row = {
                    "id": record["id"],
                    "question": record["question"],
                    "condition": args.condition,
                    "retrieval": retrieval_metadata,
                    "retrieved_ids": [
                        item["chunk_id"] for item in ranked
                    ],
                    "retrieval_details": [
                        {
                            "chunk_id": item["chunk_id"],
                            "score": item["retrieval_score"],
                            "source_ranks": item["source_ranks"],
                            "rrf_score": item.get("rrf_score"),
                        }
                        for item in ranked
                    ],
                    "gold_ids": record["ground_truth_chunks"],
                    "draft_answer": generated["text"],
                    "answer": final_answer,
                    "claims": claim_rows,
                    "generation_latency_seconds": generated[
                        "latency_seconds"
                    ],
                    "critic_latency_seconds": critic_latency,
                    "evaluation_latency_seconds": evaluation_latency,
                    "model": generated["model"],
                    "critic_model": (
                        critic.model if args.condition == "critic" else None
                    ),
                    "evaluation_model": evaluator.model,
                    "raw_critic_response": critic_response,
                    "raw_evaluation_response": evaluation_response,
                    "status": "completed",
                }

            except Exception as exc:
                row = {
                    "id": record["id"],
                    "question": record["question"],
                    "condition": args.condition,
                    "retrieval": retrieval_metadata,
                    "retrieved_ids": [
                        item["chunk_id"] for item in ranked
                    ],
                    "retrieval_details": [
                        {
                            "chunk_id": item["chunk_id"],
                            "score": item["retrieval_score"],
                            "source_ranks": item["source_ranks"],
                            "rrf_score": item.get("rrf_score"),
                        }
                        for item in ranked
                    ],
                    "gold_ids": record["ground_truth_chunks"],
                    "claims": [],
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }

            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            sink.flush()

            print(
                f"[{index}/{len(records)}] "
                f"{row['id']}: {row['status']}"
            )

    print(f"Saved records to {output}")


if __name__ == "__main__":
    main()
