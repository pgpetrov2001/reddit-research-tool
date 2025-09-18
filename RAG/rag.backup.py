#!/usr/bin/env python3
"""
Single-file RAG for a folder of .txt/.md files **or** a Reddit posts JSONL file.

Features
- Ingest & chunk files (TXT/MD) or **Reddit posts JSONL**
- Hybrid retrieval: BM25 + Embeddings (FAISS) with RRF fusion
- Optional cross-encoder reranker (SentenceTransformers)
- Optional answer generation with OpenAI (otherwise extractive fallback)

Usage
------
# 1) Install deps
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install sentence-transformers faiss-cpu rank-bm25 numpy tqdm fastapi uvicorn openai

# 2a) Index a folder of TXT/MD
python rag.py index --docs ./docs --store .rag_store

# 2b) **Index Reddit posts JSONL** (one JSON object per line)
python rag.py index-jsonl --posts ./hair.posts.jsonl --store .rag_store

# 3) Ask questions (CLI)
python rag.py ask --query "What do people say about bad haircuts?" --k 8 --store .rag_store

# 4) (Optional) Run a tiny API
uvicorn rag:app --reload --port 8000

Env (optional)
OPENAI_API_KEY=...       # to enable LLM answers
OPENAI_MODEL=gpt-4o-mini  # or another chat-capable model

Skeptical defaults
- Chunk 800 words with 160 overlap
- BM25 topK=100, Vector topK=100, RRF_k=60, rerank@12 (if installed)
- “I don’t know” stance when context doesn’t cover the answer
"""

from __future__ import annotations
import os, re, json, pickle, argparse, hashlib, time, sys
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional

import numpy as np
from tqdm import tqdm

# --- Optional heavy deps are imported lazily where needed ---

# --------------------- Config ---------------------
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", 160))
BM25_TOPK = int(os.getenv("RAG_BM25_TOPK", 100))
VEC_TOPK = int(os.getenv("RAG_VEC_TOPK", 100))
RRF_K = int(os.getenv("RAG_RRF_K", 60))
RERANK_TOPK = int(os.getenv("RAG_RERANK_TOPK", 12))
USE_RERANKER = os.getenv("RAG_USE_RERANKER", "true").lower() == "true"
EMB_MODEL = os.getenv("RAG_EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
MAX_CONTEXT_WORDS = int(os.getenv("RAG_MAX_CTX_WORDS", 3500))

# --------------------- Data shapes ----------------
@dataclass
class Chunk:
    id: str
    doc_id: str
    title: str
    source: str
    section: Optional[str]
    text: str
    updated_at: Optional[str]

@dataclass
class Candidate:
    chunk: Chunk
    score: float
    scores: Dict[str, float]

# --------------------- Utils ---------------------
_word_re = re.compile(r"\w+")

def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _word_re.findall(text)]

def rrf(rankings: Dict[str, List[str]], k: int = 60) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for _, ranked_ids in rankings.items():
        for rank, cid in enumerate(ranked_ids, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return scores

def trim_by_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " …"

# --------------------- Chunking -------------------

def split_markdown_sections(md: str) -> List[Tuple[str, str]]:
    parts = re.split(r"(^#+\s+.*$)", md, flags=re.M)
    if not parts:
        return [("", md)]
    out: List[Tuple[str,str]] = []
    cur_head = "Intro"
    buf: List[str] = []
    for p in parts:
        if p.startswith("#"):
            if buf:
                out.append((cur_head.strip(), "".join(buf).strip()))
                buf = []
            cur_head = p.strip().lstrip('#').strip()
        else:
            buf.append(p)
    if buf:
        out.append((cur_head.strip(), "".join(buf).strip()))
    return out


def chunk_text(text: str, target_words: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    words = text.split()
    step = max(1, target_words - overlap)
    chunks: List[str] = []
    for start in range(0, len(words), step):
        window = words[start:start + target_words]
        if not window:
            break
        chunks.append(" ".join(window))
    return chunks

# --------------------- Ingest (TXT/MD) ---------------------
ALLOWED_EXT = {".txt", ".md"}

def file_to_chunks(path: str, doc_id: str, title: str) -> List[Chunk]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    sections = split_markdown_sections(raw)
    out: List[Chunk] = []
    i = 0
    for sec_title, sec_text in sections:
        for piece in chunk_text(sec_text):
            cid = hashlib.sha1(f"{doc_id}:{sec_title}:{i}:{hashlib.sha1(piece.encode()).hexdigest()}".encode()).hexdigest()[:16]
            out.append(Chunk(
                id=cid,
                doc_id=doc_id,
                title=title,
                source=path,
                section=sec_title,
                text=piece,
                updated_at=time.strftime('%Y-%m-%d')
            ))
            i += 1
    return out


def ingest_dir(docs_dir: str) -> List[Chunk]:
    chunks: List[Chunk] = []
    for root, _, files in os.walk(docs_dir):
        for fn in files:
            ext = os.path.splitext(fn)[1].lower()
            if ext not in ALLOWED_EXT:
                continue
            path = os.path.join(root, fn)
            doc_id = hashlib.sha1(path.encode()).hexdigest()[:12]
            title = os.path.splitext(os.path.basename(path))[0]
            chunks.extend(file_to_chunks(path, doc_id, title))
    return chunks

# --------------------- Reddit JSONL Ingest ----------------
def _try_parse_json_line(line: str) -> Optional[dict]:
    line = line.strip()
    if not line:
        return None
    # 1) Try as-is
    try:
        return json.loads(line)
    except Exception:
        pass
    # 2) Common cases: trailing comma or missing braces
    s = line.rstrip(", ")
    try:
        return json.loads(s)
    except Exception:
        pass
    # 3) Missing outer braces (e.g., '"author": "x", ...}')
    if not s.startswith("{") and s.endswith("}"):
        try:
            return json.loads("{" + s + "}")
        except Exception:
            pass
    # 4) Fully missing braces (no closing brace)
    if not s.startswith("{") and not s.endswith("}"):
        try:
            return json.loads("{" + s + "}")
        except Exception:
            pass
    return None


def ingest_posts_jsonl(posts_path: str) -> List[Chunk]:
    """Ingest a JSONL file where each line is a Reddit post JSON object.
    Expected fields: id, title, selftext, url, subreddit, author, created_utc, link_flair_text (optional).
    Builds chunks from (title + selftext). If selftext is empty, title is still indexed.
    """
    chunks: List[Chunk] = []
    doc_counter = 0
    with open(posts_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            obj = _try_parse_json_line(raw_line)
            if not obj:
                continue
            pid = (obj.get("id") or f"post_{doc_counter}")
            title = (obj.get("title") or "").strip() or f"reddit_post_{pid}"
            selftext = (obj.get("selftext") or "").strip()
            url = (obj.get("url") or "").strip()
            subreddit = (obj.get("subreddit") or "").strip()
            flair = (obj.get("link_flair_text") or "").strip()
            created = obj.get("created_utc")
            try:
                if isinstance(created, (int, float)):
                    updated_at = time.strftime("%Y-%m-%d", time.gmtime(int(created)))
                else:
                    updated_at = time.strftime("%Y-%m-%d", time.gmtime())
            except Exception:
                updated_at = time.strftime("%Y-%m-%d", time.gmtime())

            header_bits = []
            if subreddit: header_bits.append(f"r/{subreddit}")
            if flair: header_bits.append(f"[{flair}]")
            section = " ".join(header_bits).strip() or "post"
            body = (title + "\n\n" + selftext).strip() if selftext else title

            source = url if url else (f"reddit://{subreddit}/{pid}" if subreddit else f"reddit://{pid}")
            doc_id = pid

            pieces = chunk_text(body, target_words=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
            local_i = 0
            for piece in pieces:
                cid = hashlib.sha1(f"{doc_id}:{section}:{local_i}:{hashlib.sha1(piece.encode()).hexdigest()}".encode()).hexdigest()[:16]
                chunks.append(Chunk(
                    id=cid,
                    doc_id=doc_id,
                    title=title,
                    source=source,
                    section=section,
                    text=piece,
                    updated_at=updated_at
                ))
                local_i += 1
            doc_counter += 1
    return chunks

# --------------------- Index build ----------------

def build_indexes(chunks: List[Chunk], store_dir: str, emb_model: str = EMB_MODEL):
    os.makedirs(store_dir, exist_ok=True)
    meta_path = os.path.join(store_dir, 'meta.jsonl')
    with open(meta_path, 'w', encoding='utf-8') as f:
        for ch in chunks:
            f.write(json.dumps(asdict(ch), ensure_ascii=False) + "\n")

    # Embeddings
    from sentence_transformers import SentenceTransformer
    import faiss
    texts = [c.text for c in chunks]
    model = SentenceTransformer(emb_model)
    X = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    np.save(os.path.join(store_dir, 'embeddings.npy'), X)

    # FAISS index (cosine via inner product)
    d = X.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(X)
    faiss.write_index(index, os.path.join(store_dir, 'faiss.index'))

    # ID map
    idmap = {c.id: i for i, c in enumerate(chunks)}
    with open(os.path.join(store_dir, 'idmap.json'), 'w') as f:
        json.dump(idmap, f)

    # BM25
    from rank_bm25 import BM25Okapi
    tokenized = [tokenize(c.text) for c in chunks]
    ids = [c.id for c in chunks]
    with open(os.path.join(store_dir, 'bm25.pkl'), 'wb') as f:
        pickle.dump({'tokenized': tokenized, 'ids': ids}, f)

# --------------------- Retrieval ------------------
class VectorStore:
    def __init__(self, store_dir: str):
        self.store_dir = store_dir
        self.index_path = os.path.join(store_dir, 'faiss.index')
        self.emb_path = os.path.join(store_dir, 'embeddings.npy')
        self.meta_path = os.path.join(store_dir, 'meta.jsonl')
        self.idmap_path = os.path.join(store_dir, 'idmap.json')
        self.index = None
        self.embeddings = None
        self.idmap: Dict[str, int] = {}
        self.meta: Dict[str, Chunk] = {}

    def load(self):
        import faiss
        self.index = faiss.read_index(self.index_path)
        self.embeddings = np.load(self.emb_path)
        with open(self.idmap_path, 'r') as f:
            self.idmap = json.load(f)
        self.meta = {}
        with open(self.meta_path, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                self.meta[obj['id']] = Chunk(**obj)

    def search(self, query_vec: np.ndarray, topk: int) -> List[Tuple[str, float]]:
        import faiss
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

class VectorRetriever:
    def __init__(self, store_dir: str, model_name: str = EMB_MODEL):
        from sentence_transformers import SentenceTransformer
        self.store = VectorStore(store_dir)
        self.store.load()
        self.model = SentenceTransformer(model_name)

    def encode(self, text: str) -> np.ndarray:
        return self.model.encode([text], normalize_embeddings=True)

    def retrieve(self, query: str, topk: int) -> List[Tuple[str, float]]:
        qv = self.encode(query)
        return self.store.search(qv, topk)

class BM25Retriever:
    def __init__(self, store_dir: str):
        with open(os.path.join(store_dir, 'bm25.pkl'), 'rb') as f:
            data = pickle.load(f)
        from rank_bm25 import BM25Okapi
        self.bm25 = BM25Okapi(data['tokenized'])
        self.chunk_ids: List[str] = data['ids']

    def retrieve(self, query: str, topk: int) -> List[Tuple[str, float]]:
        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:topk]
        return [(self.chunk_ids[i], float(scores[i])) for i in top_idx]

class HybridRetriever:
    def __init__(self, store_dir: str, rrf_k: int = RRF_K):
        self.vec = VectorRetriever(store_dir)
        self.bm25 = BM25Retriever(store_dir)
        self.store = self.vec.store
        self.rrf_k = rrf_k

    def retrieve(self, query: str, bm25_topk: int, vec_topk: int, final_topk: int) -> List[Candidate]:
        bm25_hits = self.bm25.retrieve(query, bm25_topk)
        vec_hits = self.vec.retrieve(query, vec_topk)
        rankings = {'bm25': [cid for cid,_ in bm25_hits], 'vec': [cid for cid,_ in vec_hits]}
        fused = rrf(rankings, k=self.rrf_k)
        sorted_ids = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:max(final_topk*3, final_topk)]
        bm25_map = dict(bm25_hits)
        vec_map = dict(vec_hits)
        out: List[Candidate] = []
        for cid, fused_score in sorted_ids:
            chunk = self.store.meta[cid]
            out.append(Candidate(chunk=chunk, score=fused_score, scores={'bm25': bm25_map.get(cid, 0.0), 'vector': vec_map.get(cid, 0.0), 'rrf': fused_score}))
        return out

# --------------------- Reranker (optional) --------
class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, cands: List[Candidate], topk: int) -> List[Candidate]:
        pairs = [(query, c.chunk.text) for c in cands]
        scores = self.model.predict(pairs)
        for c, s in zip(cands, scores):
            c.scores['rerank'] = float(s)
            c.score = float(s)
        cands.sort(key=lambda c: c.score, reverse=True)
        return cands[:topk]

# --------------------- Answer generation ----------
SYSTEM_PROMPT = (
    "You are a careful assistant. Answer using ONLY the provided context. "
    "If the answer is not in the context, say you don't know. Cite sources as [title](path)."
)

def build_context(cands: List[Candidate], max_words: int = MAX_CONTEXT_WORDS) -> str:
    by_doc: Dict[Tuple[str,str], List[Candidate]] = {}
    for c in cands:
        key = (c.chunk.title, c.chunk.source)
        by_doc.setdefault(key, []).append(c)
    blocks = []
    for (title, src), items in by_doc.items():
        blocks.append(f"### {title} ({src})\n" + "\n\n".join([i.chunk.text for i in items]))
    ctx = "\n\n".join(blocks)
    return trim_by_words(ctx, max_words)


def maybe_openai_answer(query: str, context: str) -> Optional[str]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI()
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        messages = [
            {"role":"system","content": SYSTEM_PROMPT},
            {"role":"user","content": f"Context:\n\n{context}\n\nQuestion: {query}\nRespond with citations."}
        ]
        resp = client.chat.completions.create(model=model, messages=messages, temperature=0.0)
        return resp.choices[0].message.content
    except Exception as e:
        return f"(OpenAI error, falling back to extractive)\n{e}"


def extractive_answer(query: str, cands: List[Candidate]) -> str:
    # Simple extractive fallback: return the top chunk(s) with a disclaimer
    top = cands[:3]
    parts = [
        "I don't have a generative model configured. Here are relevant passages from your files:\n"
    ]
    for c in top:
        parts.append(f"---\n# {c.chunk.title} ({c.chunk.source})\n{c.chunk.text}\n")
    return "\n".join(parts)

# --------------------- Pipeline -------------------
class Pipeline:
    def __init__(self, store_dir: str):
        self.hybrid = HybridRetriever(store_dir)
        # Try to enable reranker; fall back gracefully if not installed
        self.reranker: Optional[CrossEncoderReranker]
        if USE_RERANKER:
            try:
                self.reranker = CrossEncoderReranker()
            except Exception:
                self.reranker = None
        else:
            self.reranker = None

    def ask(self, query: str, k: int = 10) -> Dict:
        cands = self.hybrid.retrieve(query, bm25_topk=BM25_TOPK, vec_topk=VEC_TOPK, final_topk=max(k*3, k))
        used_reranker = False
        if self.reranker is not None:
            cands = self.reranker.rerank(query, cands, topk=k)
            used_reranker = True
        else:
            cands = cands[:k]
        context = build_context(cands)
        llm = maybe_openai_answer(query, context)
        answer = llm if llm else extractive_answer(query, cands)
        citations = [
            {
                'title': c.chunk.title,
                'source': c.chunk.source,
                'section': c.chunk.section,
                'chunk_id': c.chunk.id
            } for c in cands
        ]
        return {
            'answer': answer,
            'citations': citations,
            'used_reranker': used_reranker,
            'meta': {'bm25_topk': BM25_TOPK, 'vec_topk': VEC_TOPK, 'k': k}
        }

# --------------------- CLI ------------------------

def cmd_index(args):
    docs_dir = args.docs
    store = args.store
    print(f"Ingesting from {docs_dir} …")
    chunks = ingest_dir(docs_dir)
    if not chunks:
        print("No .txt or .md files found.")
        sys.exit(1)
    print(f"Building indexes into {store} (chunks={len(chunks)}) …")
    build_indexes(chunks, store)
    print("Done. ✅")


def cmd_index_jsonl(args):
    posts_path = args.posts
    store = args.store
    print(f"Ingesting Reddit posts from {posts_path} …")
    chunks = ingest_posts_jsonl(posts_path)
    if not chunks:
        print("No posts parsed from JSONL. Check the file format.")
        sys.exit(1)
    print(f"Building indexes into {store} (chunks={len(chunks)}) …")
    build_indexes(chunks, store)
    print("Done. ✅")


def cmd_ask(args):
    pipe = Pipeline(args.store)
    out = pipe.ask(args.query, k=args.k)
    print("\n=== Answer ===\n")
    print(out['answer'])
    print("\n=== Citations ===")
    for c in out['citations']:
        sec = f" › {c['section']}" if c['section'] else ""
        print(f"- [{c['title']}]({c['source']}){sec}  (chunk={c['chunk_id']})")

# --------------------- FastAPI (optional) ---------
try:
    from fastapi import FastAPI
    app = FastAPI(title="RAG from Texts & Reddit JSONL")
    _pipe: Optional[Pipeline] = None

    @app.get("/health")
    def health():
        return {"status":"ok"}

    @app.post("/ask")
    def api_ask(payload: Dict[str, object]):
        query = str(payload.get('query', ''))
        k = int(payload.get('k', 10))
        store = str(payload.get('store', '.rag_store'))
        global _pipe
        if _pipe is None:
            _pipe = Pipeline(store)
        return _pipe.ask(query, k)
except Exception:
    app = None  # FastAPI not installed; API mode unavailable

# --------------------- Entry ----------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="RAG over a folder of text files or a Reddit posts JSONL file")
    sub = ap.add_subparsers(dest="cmd")

    ap_i = sub.add_parser("index", help="Ingest and build indexes from TXT/MD files")
    ap_i.add_argument("--docs", default="./docs")
    ap_i.add_argument("--store", default=".rag_store")
    ap_i.set_defaults(func=cmd_index)

    ap_j = sub.add_parser("index-jsonl", help="Ingest Reddit posts from a JSONL file and build indexes")
    ap_j.add_argument("--posts", required=True, help="Path to posts JSONL file (one JSON object per line)")
    ap_j.add_argument("--store", default=".rag_store")
    ap_j.set_defaults(func=cmd_index_jsonl)

    ap_q = sub.add_parser("ask", help="Ask a question against the index")
    ap_q.add_argument("--query", required=True)
    ap_q.add_argument("--k", type=int, default=10)
    ap_q.add_argument("--store", default=".rag_store")
    ap_q.set_defaults(func=cmd_ask)

    args = ap.parse_args()
    if not args.cmd:
        ap.print_help()
        sys.exit(0)
    args.func(args)

