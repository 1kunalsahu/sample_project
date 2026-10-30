from __future__ import annotations
import hashlib
import json
from pathlib import Path

class DenseSearch:
    """Sentence-transformers index with a reusable on-disk corpus embedding cache."""
    def __init__(self, documents, model_name="sentence-transformers/all-MiniLM-L6-v2", cache_dir=".cache/criticrag"):
        try:
            from sentence_transformers import SentenceTransformer
            from huggingface_hub import snapshot_download
        except ImportError as e: raise RuntimeError("Install criticrag[retrieval] to enable dense search") from e
        try:
            local_model=snapshot_download(repo_id=model_name,local_files_only=True)
            local_only=True
        except Exception:
            local_model=model_name
            local_only=False
        self.documents=documents; self.model=SentenceTransformer(local_model,local_files_only=local_only)
        cache_key=hashlib.sha256(json.dumps({"model":model_name,"documents":[(d["chunk_id"],d["text"]) for d in documents]},ensure_ascii=False,separators=(",",":")).encode()).hexdigest()
        cache_path=Path(cache_dir)/f"dense-{cache_key}.npy" if cache_dir else None
        if cache_path and cache_path.exists():
            import numpy as np
            self.vectors=np.load(cache_path,mmap_mode="r")
            if len(self.vectors)!=len(documents): raise ValueError(f"Dense cache row count mismatch: {cache_path}")
        else:
            self.vectors=self.model.encode([d["text"] for d in documents],normalize_embeddings=True,batch_size=256,show_progress_bar=True)
            if cache_path:
                import numpy as np
                cache_path.parent.mkdir(parents=True,exist_ok=True)
                np.save(cache_path,self.vectors)
        self.dimension=self.vectors.shape[-1]
    def search(self, query, k=5):
        q=self.model.encode([query],normalize_embeddings=True)
        if q.shape[-1] != self.dimension: raise ValueError("Embedding dimension mismatch")
        scores=self.vectors @ q[0]
        inds=sorted(range(len(scores)),key=lambda i:(-float(scores[i]),self.documents[i]["chunk_id"]))[:k]
        return [{"chunk_id":self.documents[i]["chunk_id"],"score":float(scores[i])} for i in inds]
