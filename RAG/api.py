from __future__ import annotations
from typing import Dict, Optional

from fastapi import FastAPI

try:
    from RAG.rag import Pipeline  # reuse the Pipeline; if desired, can be split too
except ModuleNotFoundError:
    from rag import Pipeline


app = FastAPI(title="RAG (Voyage embeddings + Keyword BM25, optional xAI chat)")
_pipe: Optional[Pipeline] = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def api_ask(payload: Dict[str, object]):
    query = str(payload.get('query', ''))
    k = int(payload.get('k', 10))
    store = str(payload.get('store', '.rag_store'))
    mode = str(payload.get('mode', 'embed'))
    global _pipe
    if _pipe is None:
        _pipe = Pipeline(store)
    return _pipe.ask(query, k, mode)


