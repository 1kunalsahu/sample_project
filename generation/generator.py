from __future__ import annotations
import json, os, time, urllib.request

class OpenAICompatibleGenerator:
    """Minimal chat-completions adapter; explicitly configured endpoint, no hidden fallback."""
    def __init__(self, model: str | None=None, base_url: str | None=None, api_key: str | None=None):
        self.model=model or os.getenv("CRITICRAG_MODEL")
        self.base=(base_url or os.getenv("CRITICRAG_API_BASE", "https://api.openai.com/v1")).rstrip("/")
        self.key=api_key or os.getenv("CRITICRAG_API_KEY")
        if not self.model or not self.key: raise RuntimeError("Set CRITICRAG_MODEL and CRITICRAG_API_KEY before inference")
    def complete(self, messages, temperature=0, json_mode=False):
        payload={"model":self.model,"messages":messages,"temperature":temperature}
        if json_mode: payload["response_format"]={"type":"json_object"}
        body=json.dumps(payload).encode()
        req=urllib.request.Request(self.base+"/chat/completions",data=body,headers={"Authorization":"Bearer "+self.key,"Content-Type":"application/json"})
        start=time.perf_counter()
        with urllib.request.urlopen(req,timeout=120) as response: payload=json.load(response)
        return {"text":payload["choices"][0]["message"]["content"],"usage":payload.get("usage",{}),"latency_seconds":time.perf_counter()-start,"model":payload.get("model",self.model)}

class OllamaGenerator:
    """Local chat adapter for Ollama's native /api/chat endpoint."""
    def __init__(self, model: str, base_url: str | None=None):
        self.model=model
        self.base=(base_url or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        if not self.model: raise RuntimeError("Set a local Ollama model in configs/models.yaml")
    def complete(self, messages, temperature=0, json_mode=False):
        payload={"model":self.model,"messages":messages,"stream":False,"options":{"temperature":temperature}}
        if json_mode: payload["format"]="json"
        body=json.dumps(payload).encode()
        req=urllib.request.Request(self.base+"/api/chat",data=body,headers={"Content-Type":"application/json"})
        start=time.perf_counter()
        try:
            with urllib.request.urlopen(req,timeout=600) as response: payload=json.load(response)
        except Exception as exc:
            raise RuntimeError(f"Ollama request failed at {self.base}. Ensure Ollama is running and model '{self.model}' is installed: {exc}") from exc
        usage={"prompt_tokens":payload.get("prompt_eval_count"),"completion_tokens":payload.get("eval_count")}
        return {"text":payload["message"]["content"],"usage":usage,"latency_seconds":time.perf_counter()-start,"model":payload.get("model",self.model)}

def create_chat_model(backend: str, model: str | None, base_url: str | None=None):
    """Construct a provider client from experiment config."""
    if backend == "ollama": return OllamaGenerator(model=model or "",base_url=base_url)
    if backend == "openai_compatible": return OpenAICompatibleGenerator(model=model,base_url=base_url)
    raise ValueError(f"Unsupported inference backend: {backend}")

def answer_prompt(question: str, chunks: list[dict]) -> list[dict]:
    evidence="\n\n".join(f"[{c['chunk_id']}] {c['text']}" for c in chunks)
    return [{"role":"system","content":"Answer only from supplied evidence. Cite every factual claim with its exact chunk ID. Say when evidence is insufficient."},{"role":"user","content":f"Question: {question}\n\nEvidence:\n{evidence}"}]
