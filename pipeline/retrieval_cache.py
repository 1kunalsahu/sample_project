from __future__ import annotations
import hashlib, json

def fingerprint(config: dict) -> str:
    return hashlib.sha256(json.dumps(config,sort_keys=True,separators=(",",":")).encode()).hexdigest()

class RetrievalCache:
    def __init__(self): self._items={}
    def put(self, question_id: str, config: dict, value: dict): self._items[(question_id,fingerprint(config))]=value
    def get(self, question_id: str, config: dict):
        try: return self._items[(question_id,fingerprint(config))]
        except KeyError as e: raise KeyError("Missing retrieval result or incompatible retrieval config") from e
