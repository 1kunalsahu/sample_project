from __future__ import annotations
import re

def chunk_document(document_id: str, text: str, *, source: str = "", max_chars: int = 1200, overlap: int = 150) -> list[dict]:
    """Paragraph-aware chunks with deterministic IDs and character offsets."""
    if max_chars < 1 or overlap < 0 or overlap >= max_chars: raise ValueError("Require max_chars > overlap >= 0")
    clean=re.sub(r"\s+", " ", text).strip()
    if not clean: return []
    spans=[]; start=0
    while start < len(clean):
        end=min(start+max_chars,len(clean))
        if end < len(clean):
            boundary=max(clean.rfind(". ", start, end), clean.rfind("; ", start, end))
            if boundary > start + max_chars//2: end=boundary+1
        spans.append((start,end,clean[start:end].strip()))
        if end == len(clean): break
        start=max(start+1,end-overlap)
    return [{"chunk_id":f"{document_id}_{i:04d}","document_id":document_id,"source":source,"page":None,"section":None,"text":t,"start_offset":a,"end_offset":b} for i,(a,b,t) in enumerate(spans)]
