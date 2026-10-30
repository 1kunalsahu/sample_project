from __future__ import annotations
import argparse, json
from pathlib import Path
from ingestion.fiqa import download, load_records
from retrieval.bm25 import BM25
from retrieval.dense_search import DenseSearch
from retrieval.hybrid import hybrid_search
from retrieval.reranker import CrossEncoderReranker
from evaluation.retrieval_metrics import evidence_recall, query_hit_rate
import yaml

def main():
    ap=argparse.ArgumentParser(description="Run FiQA hybrid BM25 + dense retrieval evaluation")
    ap.add_argument("--download",action="store_true",help="Download public BEIR FiQA corpus and qrels")
    ap.add_argument("--data",default="data/raw/fiqa/fiqa"); ap.add_argument("--split",default="test"); ap.add_argument("--limit",type=int)
    ap.add_argument("--k",type=int,default=5); ap.add_argument("--output",default="results/metrics/fiqa_retrieval.json")
    ap.add_argument("--retrieval-config",default="configs/retrieval.yaml")
    ap.add_argument("--candidate-k",type=int); ap.add_argument("--dense-model")
    ap.add_argument("--reranker-model",help="Set to 'none' to disable the configured reranker")
    args=ap.parse_args()
    if args.download: download(Path(args.data).parent)
    retrieval_config=yaml.safe_load(Path(args.retrieval_config).read_text(encoding="utf-8")) or {}
    candidate_k=args.candidate_k or int(retrieval_config.get("candidate_k",50))
    dense_model=args.dense_model or retrieval_config.get("dense_model","sentence-transformers/all-MiniLM-L6-v2")
    reranker_name=args.reranker_model if args.reranker_model is not None else retrieval_config.get("reranker_model")
    if reranker_name and str(reranker_name).lower()=="none": reranker_name=None
    if args.k<1 or candidate_k<args.k: ap.error("k must be positive and candidate-k must be >= k")
    root=Path(args.data); corpus=[]
    with (root/"corpus.jsonl").open(encoding="utf-8") as f:
        for line in f:
            x=json.loads(line); corpus.append({"chunk_id":x["_id"],"document_id":x["_id"],"text":(x.get("title","")+" "+x.get("text","")).strip(),"source":"FiQA"})
    records=load_records(root,args.split)
    if args.limit: records=records[:args.limit]
    index=BM25(corpus)
    dense=DenseSearch(corpus,dense_model,retrieval_config.get("embedding_cache", ".cache/criticrag"))
    reranker=CrossEncoderReranker(reranker_name) if reranker_name else None
    rows=[]
    for q in records:
        ranked=hybrid_search(q["question"],index,dense,corpus,args.k,candidate_k,int(retrieval_config.get("rrf_constant",60)),reranker)
        ids=[x["chunk_id"] for x in ranked]; gold=q["ground_truth_chunks"]
        rows.append({"id":q["id"],"question":q["question"],"retrieved_ids":ids,"retrieval_details":[{"chunk_id":x["chunk_id"],"score":x["retrieval_score"],"source_ranks":x["source_ranks"],"rrf_score":x.get("rrf_score")} for x in ranked],"gold_ids":gold,"recall_at_k":evidence_recall(ids,gold,args.k),"hit_at_k":query_hit_rate(ids,gold,args.k)})
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    summary={"dataset":"FiQA-2018 BEIR","split":args.split,"retrieval":{"method":"BM25 + dense + RRF"+(" + cross-encoder reranker" if reranker else ""),"dense_model":dense_model,"reranker_model":reranker_name,"candidate_k":candidate_k,"k":args.k,"rrf_constant":int(retrieval_config.get("rrf_constant",60))},"processed":len(rows),"question_level_evidence_hit_rate":sum(r["hit_at_k"] or 0 for r in rows)/len(rows) if rows else None,"evidence_set_recall":sum(r["recall_at_k"] or 0 for r in rows)/len(rows) if rows else None,"per_example":rows,"note":"Retrieval metrics only; FiQA qrels are relevance labels, not human claim annotations or answer references."}
    out.write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps({k:v for k,v in summary.items() if k!="per_example"},indent=2))

if __name__=="__main__": main()
