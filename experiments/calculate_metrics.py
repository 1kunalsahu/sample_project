from __future__ import annotations
import argparse, json
from pathlib import Path
from evaluation.aggregate import aggregate_records

def main():
    parser=argparse.ArgumentParser(description="Calculate retrieval recall, claim hallucination, and citation faithfulness from saved JSON/JSONL records")
    parser.add_argument("--input",required=True,help="JSONL per-example records or a JSON report with per_example rows")
    parser.add_argument("--k",nargs="+",type=int,default=[3,5,10]); parser.add_argument("--output",default="results/metrics/aggregate_metrics.json")
    args=parser.parse_args()
    if any(k<1 for k in args.k): parser.error("k must be positive")
    src=Path(args.input)
    if src.suffix.lower()==".jsonl":
        records=[json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        payload=json.loads(src.read_text(encoding="utf-8")); records=payload if isinstance(payload,list) else payload.get("per_example",[])
    summary=aggregate_records(records,args.k)
    summary["input"]=str(src); summary["limitations"]=[]
    if not summary["claims"]["evaluable"]: summary["limitations"].append("Hallucination rate is undefined because no claim verdicts were supplied.")
    if not summary["cited_claims"]: summary["limitations"].append("Citation faithfulness is undefined because no claims with citations were supplied.")
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(summary,indent=2,ensure_ascii=False))

if __name__=="__main__": main()
