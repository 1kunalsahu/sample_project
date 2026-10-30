"""Download and normalize the public BEIR FiQA benchmark."""
from __future__ import annotations
import hashlib, json, shutil, ssl, urllib.request, zipfile
from pathlib import Path

URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip"
EXPECTED_MD5 = "17918ed23cd04fb15047f73e6c3bd9d9"

def download(destination: str | Path = "data/raw/fiqa") -> Path:
    root = Path(destination); root.mkdir(parents=True, exist_ok=True)
    archive = root.parent / "fiqa.zip"
    if not archive.exists():
        try:
            import certifi
            context=ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            context=ssl.create_default_context()
        with urllib.request.urlopen(URL,context=context,timeout=60) as response, archive.open("wb") as target:
            shutil.copyfileobj(response,target)
    digest=hashlib.md5(archive.read_bytes()).hexdigest()
    if digest != EXPECTED_MD5: raise ValueError(f"FiQA archive checksum mismatch: expected {EXPECTED_MD5}, got {digest}")
    with zipfile.ZipFile(archive) as zf:
        base=root.resolve()
        for member in zf.infolist():
            target=(root/member.filename).resolve()
            if target != base and base not in target.parents: raise ValueError(f"Unsafe path in FiQA archive: {member.filename}")
        zf.extractall(root)
    return root / "fiqa"

def load_records(root: str | Path = "data/raw/fiqa/fiqa", split: str = "test") -> list[dict]:
    root = Path(root)
    corpus = {r["_id"]: r for r in _jsonl(root / "corpus.jsonl")}
    queries = {r["_id"]: r["text"] for r in _jsonl(root / "queries.jsonl")}
    qrel = root / "qrels" / f"{split}.tsv"
    if not qrel.exists(): raise FileNotFoundError(f"FiQA split not found: {qrel}")
    grouped={}
    for line in qrel.read_text(encoding="utf-8").splitlines()[1:]:
        qid, docid, score = line.split("\t")[:3]
        if qid not in queries or docid not in corpus: continue
        grouped.setdefault(qid,[]).append((docid,int(score)))
    return [{"id":f"fiqa_{split}_{qid}","query_id":qid,"document_id":"",
        "question":queries[qid],"ground_truth_answer":"","ground_truth_chunks":[d for d,s in rel if s>0],
        "gold_spans":[],"question_type":"unannotated","difficulty":"unannotated",
        "number_of_required_chunks":len([1 for _,s in rel if s>0]),"answerable":bool(any(s>0 for _,s in rel)),"relevance":{d:s for d,s in rel}}
        for qid,rel in grouped.items()]

def _jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip(): yield json.loads(line)
