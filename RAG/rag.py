#!/usr/bin/env python3
"""
Single-file RAG over Reddit JSONL using Voyage AI embeddings and optional xAI chat.

Changes from previous version
- Uses OpenAI Python client pointed at xAI (no local embedding models)
- Embedding-only retrieval (BM25 removed)
- One vector per submission (post or comment); no chunking
- Optional answer generation via xAI chat; otherwise extractive fallback

Quickstart
----------
# 1) Install deps
python -m pip install --upgrade pip
pip install numpy tqdm faiss-cpu fastapi uvicorn openai python-dotenv voyageai

# 2) Create .env
VOYAGE_AI_API_SECRET=...        # required (Voyage embeddings)
VOYAGE_MODEL=voyage-2           # optional; embedding model id (e.g., voyage-2, voyage-large-2)
XAI_API_KEY=...                 # optional (for chat answers)
XAI_BASE_URL=https://api.x.ai/v1  # optional; override if needed
XAI_CHAT_MODEL=grok-4           # optional; for generative answers

# 3) Index Reddit JSONL
python rag.py index-jsonl --posts ./out/tressless.posts.jsonl --store .rag_store
python rag.py index-jsonl --comments ./out/tressless.comments.jsonl --store .rag_store

# 4) Ask
python rag.py ask --query "What helps with minoxidil shedding?" --k 8 --store .rag_store --mode embed --action ask
# keyword retrieval uses LLM-expanded keywords:
python rag.py ask --query "What helps with minoxidil shedding?" --k 8 --store .rag_store --mode keyword --action retrieve

# 5) API (optional)
# Run: uvicorn RAG.api:app --reload --port 8000
"""

from __future__ import annotations
import os, re, json, argparse, sys, pickle, math
from dataclasses import asdict
from typing import List, Tuple, Dict, Optional

import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
import faiss


# Load environment variables early
load_dotenv()


from RAG.models import Chunk
from RAG.ai import embed_texts as voyage_embed_texts
from RAG.ingest import ingest_posts_jsonl, ingest_comments_jsonl
from RAG.pipeline import Pipeline


EMBED_BATCH = int(os.getenv("RAG_EMBED_BATCH", 64))


# --------------------- Utils ---------------------
_word_re = re.compile(r"\w+")


def embed_texts(texts: List[str]) -> np.ndarray:
    return voyage_embed_texts(texts, EMBED_BATCH)


# --------------------- Index build ----------------
def build_embedding_index(chunks: List[Chunk], store_dir: str, faiss_write_y: Optional[int] = None) -> Dict[str, int]:
    os.makedirs(store_dir, exist_ok=True)

    meta_path = os.path.join(store_dir, 'meta.jsonl')
    idmap_path = os.path.join(store_dir, 'idmap.json')
    faiss_path = os.path.join(store_dir, 'faiss.index')

    # Load existing ids from idmap to support incremental/resumable indexing
    existing_ids: set = set()
    idmap_existing: Dict[str, int] = {}
    if os.path.exists(idmap_path):
        try:
            with open(idmap_path, 'r', encoding='utf-8') as f:
                idmap_existing = json.load(f)
            existing_ids.update(idmap_existing.keys())
        except Exception:
            pass

    new_chunks = [c for c in chunks if c.id not in existing_ids]
    added_meta = 0
    added_embeddings = 0

    print(f"There are {len(new_chunks)} new chunks to embed.")

    # Load existing FAISS index if present
    index = faiss.read_index(faiss_path) if os.path.exists(faiss_path) else None

    if new_chunks:
        # Configure write schedule
        use_schedule = isinstance(faiss_write_y, int) and faiss_write_y is not None and faiss_write_y > 2
        batch_idx = 0
        next_write_at = 1  # Always write first batch

        pending_meta_lines: List[str] = []

        def flush_to_disk():
            nonlocal pending_meta_lines
            print(f"Flushing to disk at batch {batch_idx}.")
            # Persist FAISS index
            faiss.write_index(index, faiss_path)
            # Persist id map
            print(f"Persisting id map to {idmap_path}.")
            with open(idmap_path, 'w', encoding='utf-8') as f:
                json.dump(idmap_existing, f)
            # Append pending meta lines
            if pending_meta_lines:
                print(f"Appending {len(pending_meta_lines)} pending meta lines to {meta_path}.")
                with open(meta_path, 'a', encoding='utf-8') as mf:
                    for line in pending_meta_lines:
                        mf.write(line)
                pending_meta_lines = []

        for i in range(0, len(new_chunks), EMBED_BATCH):
            batch_idx += 1
            batch_chunks = new_chunks[i:i+EMBED_BATCH]
            texts_batch = [c.text for c in batch_chunks]
            try:
                X_batch = embed_texts(texts_batch)
            except Exception as e:
                print(f"(embed batch failed, skipping) {e}")
                continue
            if index is None:
                d = X_batch.shape[1]
                index = faiss.IndexFlatIP(d)
            start_idx = index.ntotal
            index.add(X_batch)

            # Update idmap (in-memory) for this batch
            for offset, ch in enumerate(batch_chunks):
                idmap_existing[ch.id] = start_idx + offset

            # Buffer metadata lines for this batch
            for ch in batch_chunks:
                pending_meta_lines.append(json.dumps(asdict(ch), ensure_ascii=False) + "\n")
                added_meta += 1
            added_embeddings += len(batch_chunks)

            # Decide whether to write to disk now
            if not use_schedule:
                # Legacy behavior: write each batch
                flush_to_disk()
            else:
                if batch_idx >= next_write_at:
                    flush_to_disk()
                    # After writing at j=batch_idx, next write at ceil(j * x) where x=y/(y-1)
                    y = faiss_write_y
                    next_write_at = math.ceil(batch_idx * y / (y - 1))

        # Final flush to ensure last batches are persisted
        if use_schedule and pending_meta_lines:
            flush_to_disk()

    return {
        'parsed': len(chunks),
        'new_meta': added_meta,
        'new_embeddings': added_embeddings,
    }


def build_bm25_index(chunks: List[Chunk], store_dir: str) -> Dict[str, int]:
    os.makedirs(store_dir, exist_ok=True)

    meta_path = os.path.join(store_dir, 'meta.jsonl')
    bm25_path = os.path.join(store_dir, 'bm25.pkl')

    existing_meta: List[Dict] = []
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    existing_meta.append(obj)
                except Exception:
                    continue

    added_bm25 = 0
    # If bm25 exists, extend with any new ids from current chunks
    if os.path.exists(bm25_path):
        with open(bm25_path, 'rb') as f:
            data = pickle.load(f)
        tokenized = data['tokenized']
        ids = data['ids']
        have = set(ids)
        new_from_run = [c for c in chunks if c.id not in have]
        if new_from_run:
            tokenized.extend([[t.lower() for t in _word_re.findall(c.text)] for c in new_from_run])
            ids.extend([c.id for c in new_from_run])
            added_bm25 = len(new_from_run)
        with open(bm25_path, 'wb') as f:
            pickle.dump({'tokenized': tokenized, 'ids': ids}, f)
        return {'parsed': len(chunks), 'new_bm25': added_bm25}

    # Else build from all known metadata plus current chunks
    all_objs = existing_meta + [asdict(c) for c in chunks]
    seen = {}
    for obj in all_objs:
        seen[obj['id']] = obj
    ordered = [seen[k] for k in sorted(seen.keys())]
    tokenized = [[t.lower() for t in _word_re.findall(obj['text'])] for obj in ordered]
    ids = [obj['id'] for obj in ordered]
    with open(bm25_path, 'wb') as f:
        pickle.dump({'tokenized': tokenized, 'ids': ids}, f)
    return {'parsed': len(chunks), 'new_bm25': len(ordered)}


# --------------------- CLI ------------------------
def cmd_index_jsonl(args):
    store = args.store
    chunks: List[Chunk] = []
    if args.posts:
        print(f"Ingesting Reddit posts from {args.posts} …")
        chunks.extend(ingest_posts_jsonl(args.posts))
    if args.comments:
        print(f"Ingesting Reddit comments from {args.comments} …")
        chunks.extend(ingest_comments_jsonl(args.comments))
    if not chunks:
        print("No submissions parsed. Provide --posts and/or --comments JSONL.")
        sys.exit(1)
    mode = (args.index_mode or 'both').lower()
    build_embeddings = mode in ('embed','both')
    build_bm25 = mode in ('keyword','both')
    print(f"Building indexes into {store} (parsed_submissions={len(chunks)}; mode={mode}) …")
    stats_total = {'parsed': len(chunks), 'new_meta': 0, 'new_embeddings': 0, 'new_bm25': 0}
    if build_embeddings:
        # Validate FAISS write schedule parameter if provided
        y = getattr(args, 'faiss_write_y', None)
        if y is not None:
            try:
                y = int(y)
            except Exception:
                print("--faiss-write-y must be an integer >= 2")
                sys.exit(2)
            if y < 2:
                print("--faiss-write-y must be an integer >= 2")
                sys.exit(2)
        s = build_embedding_index(chunks, store, faiss_write_y=y)
        stats_total['new_meta'] += s.get('new_meta', 0)
        stats_total['new_embeddings'] += s.get('new_embeddings', 0)
    if build_bm25:
        s = build_bm25_index(chunks, store)
        stats_total['new_bm25'] += s.get('new_bm25', 0)
    print(f"Done. ✅  parsed={stats_total['parsed']}, new_meta={stats_total['new_meta']}, new_embeddings={stats_total['new_embeddings']}, new_bm25={stats_total['new_bm25']}")


def cmd_ask(args):
    pipe = Pipeline(args.store)
    out = pipe.ask(args.query, k=args.k, mode=args.mode, action=args.action)
    mode = args.mode.lower()
    if mode == 'both':
        print("\n=== Answer (Embeddings) ===\n")
        print(out['embed']['answer'])
        print("\n=== Citations (Embeddings) ===")
        for c in out['embed']['citations']:
            sec = f" › {c['section']}" if c['section'] else ""
            print(f"- [{c['title']}]({c['source']}){sec}  (chunk={c['chunk_id']})")
        print("\n=== Answer (Keywords) ===\n")
        print(out['keyword']['answer'])
        if 'expanded_keywords' in out['keyword']:
            print("\nExpanded keywords:")
            print(out['keyword']['expanded_keywords'])
        print("\n=== Citations (Keywords) ===")
        for c in out['keyword']['citations']:
            sec = f" › {c['section']}" if c['section'] else ""
            print(f"- [{c['title']}]({c['source']}){sec}  (chunk={c['chunk_id']})")
    else:
        print("\n=== Answer ===\n")
        print(out['answer'])
        if 'expanded_keywords' in out:
            print("\nExpanded keywords:")
            print(out['expanded_keywords'])
        print("\n=== Citations ===")
        for c in out['citations']:
            sec = f" › {c['section']}" if c['section'] else ""
            print(f"- [{c['title']}]({c['source']}){sec}  (chunk={c['chunk_id']})")


# --------------------- Entry ----------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="RAG over Reddit posts/comments JSONL using Voyage embeddings")
    sub = ap.add_subparsers(dest="cmd")

    ap_j = sub.add_parser("index-jsonl", help="Ingest Reddit JSONL and build indexes")
    ap_j.add_argument("--posts", required=False, help="Path to posts JSONL (one JSON object per line)")
    ap_j.add_argument("--comments", required=False, help="Path to comments JSONL (one JSON object per line)")
    ap_j.add_argument("--store", default=".rag_store")
    ap_j.add_argument("--index-mode", choices=['embed','keyword','both'], default='both', help="Which index artifacts to build")
    ap_j.add_argument("--faiss-write-y", type=int, required=False, help="Write FAISS/idmap/meta every ceil(j * y/(y-1)) batches; y must be > 2")
    ap_j.set_defaults(func=cmd_index_jsonl)

    ap_q = sub.add_parser("ask", help="Ask a question against the index")
    ap_q.add_argument("--query", required=True)
    ap_q.add_argument("--k", type=int, default=10)
    ap_q.add_argument("--mode", choices=['embed','keyword','both'], default='embed')
    ap_q.add_argument("--action", choices=['retrieve','ask'], default='ask', help='retrieve-only or retrieve + answer')
    ap_q.add_argument("--store", default=".rag_store")
    ap_q.set_defaults(func=cmd_ask)

    args = ap.parse_args()
    if not args.cmd:
        ap.print_help()
        sys.exit(0)
    args.func(args)
