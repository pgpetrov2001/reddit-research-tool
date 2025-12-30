from __future__ import annotations
import os
import pickle
import numpy as np
from typing import List

from rank_bm25 import BM25Okapi

try:
    from RAG.models import Candidate
    from RAG.vector_store import VectorStore
    from RAG.ai import embed_query, maybe_ai_keywords
except ModuleNotFoundError:
    from models import Candidate
    from vector_store import VectorStore
    from ai import embed_query, async_embed_query, maybe_ai_keywords
import re

_word_re = re.compile(r"\w+")


class VectorRetriever:
    def __init__(self, store_dir: str):
        self.store = VectorStore(store_dir)
        self.store.load()

    def _retrieve(self, qv: np.ndarray, topk: int) -> List[Candidate]:
        raw_hits = self.store.search(qv, topk)
        out: List[Candidate] = []
        for cid, score in raw_hits:
            out.append(Candidate(chunk=self.store.meta[cid], score=score))
        return out

    def retrieve(self, query: str, topk: int) -> List[Candidate]:
        qv = embed_query(query)
        return self._retrieve(qv, topk)

    async def async_retrieve(self, query: str, topk: int) -> List[Candidate]:
        qv = async_embed_query(query)
        return self._retrieve(qv, topk)


class KeywordRetriever:
    def __init__(self, store_dir: str):
        self.store_dir = store_dir
        self.store = VectorStore(store_dir)
        self.store.load()
        self.chunk_ids: List[str]
        self.tokenized: List[List[str]]
        bm25_path = os.path.join(store_dir, 'bm25.pkl')
        if os.path.exists(bm25_path):
            with open(bm25_path, 'rb') as f:
                data = pickle.load(f)
            self.tokenized = data['tokenized']
            self.chunk_ids = data['ids']
        else:
            items = [self.store.meta[cid] for cid in self.store.meta]
            items.sort(key=lambda c: c.id)
            self.chunk_ids = [c.id for c in items]
            self.tokenized = [[t.lower() for t in _word_re.findall(c.text)] for c in items]
        self.bm25 = BM25Okapi(self.tokenized)

    def retrieve(self, query: str, topk: int) -> List[Candidate]:
        tokens = [t.lower() for t in _word_re.findall(query)]
        scores = self.bm25.get_scores(tokens)
        order = np.argsort(scores)[::-1][:topk]
        out: List[Candidate] = []
        for idx in order:
            cid = self.chunk_ids[int(idx)]
            out.append(Candidate(chunk=self.store.meta[cid], score=float(scores[int(idx)])))
        return out

    def retrieve_with_expanded_keywords(self, query: str, topk: int):
        expanded = maybe_ai_keywords(query) or query
        phrases = [p.strip() for p in expanded.split(',') if p.strip()]
        tokens = []
        for p in phrases:
            tokens.extend([t.lower() for t in _word_re.findall(p)])
        if not tokens:
            tokens = [t.lower() for t in _word_re.findall(query)]
        scores = self.bm25.get_scores(tokens)
        order = np.argsort(scores)[::-1][:topk]
        out: List[Candidate] = []
        for idx in order:
            cid = self.chunk_ids[int(idx)]
            out.append(Candidate(chunk=self.store.meta[cid], score=float(scores[int(idx)])))
        return out, expanded


