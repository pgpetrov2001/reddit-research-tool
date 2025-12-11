from __future__ import annotations
import json
import os
from typing import Dict, List, Tuple

import faiss

try:
    from RAG.models import Chunk
except ModuleNotFoundError:
    from models import Chunk


class VectorStore:
    def __init__(self, store_dir: str):
        self.store_dir = store_dir
        self.index_path = os.path.join(store_dir, 'faiss.index')
        self.meta_path = os.path.join(store_dir, 'meta.jsonl')
        self.idmap_path = os.path.join(store_dir, 'idmap.json')
        self.index = None
        self.idmap: Dict[str, int] = {}
        self.meta: Dict[str, Chunk] = {}

    def load(self):
        self.index = faiss.read_index(self.index_path)
        with open(self.idmap_path, 'r', encoding='utf-8') as f:
            self.idmap = json.load(f)
        self.meta = {}
        with open(self.meta_path, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                self.meta[obj['id']] = Chunk(**obj)

    def search(self, query_vec, topk: int) -> List[Tuple[str, float]]:
        q = query_vec.astype('float32')
        faiss.normalize_L2(q)
        D, I = self.index.search(q, topk)
        hits: List[Tuple[str,float]] = []
        inv_idmap = {v:k for k,v in self.idmap.items()}
        for idx, score in zip(I[0], D[0]):
            if idx < 0:
                continue
            cid = inv_idmap.get(int(idx))
            if cid:
                hits.append((cid, float(score)))
        return hits


