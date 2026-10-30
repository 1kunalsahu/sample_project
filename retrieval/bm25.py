from __future__ import annotations
import math, re
from collections import Counter

def tokenize(text: str) -> list[str]: return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower())

class BM25:
    """Small dependency-free Okapi BM25 implementation for reproducible local runs."""
    def __init__(self, documents: list[dict], k1: float=1.5, b: float=.75):
        self.documents=documents; self.k1=k1; self.b=b
        self.tokens=[tokenize(d["text"]) for d in documents]
        self.lengths=[len(t) for t in self.tokens]; self.avg=sum(self.lengths)/max(1,len(self.lengths))
        self.tf=[Counter(t) for t in self.tokens]; df=Counter()
        for c in self.tf: df.update(c.keys())
        self.idf={w:math.log(1+(len(documents)-n+.5)/(n+.5)) for w,n in df.items()}
    def search(self, query: str, k: int=5) -> list[dict]:
        q=set(tokenize(query)); scored=[]
        for i,doc in enumerate(self.documents):
            s=0.0
            for term in q:
                f=self.tf[i].get(term,0)
                if f: s += self.idf.get(term,0)*f*(self.k1+1)/(f+self.k1*(1-self.b+self.b*self.lengths[i]/max(1,self.avg)))
            scored.append({"chunk_id":doc["chunk_id"],"score":s})
        return sorted(scored,key=lambda x:(-x["score"],x["chunk_id"]))[:k]
