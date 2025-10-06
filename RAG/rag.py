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
import os, re, json, argparse, sys, pickle
from dataclasses import asdict
from typing import List, Dict, Optional

from dotenv import load_dotenv


# Load environment variables early
load_dotenv()


from models import Chunk
from ingest import ingest_posts_jsonl, ingest_comments_jsonl
from pipeline import Pipeline
from embedder import build_embedding_index


# --------------------- Utils ---------------------
_word_re = re.compile(r"\w+")

# --------------------- Index build ----------------
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
    stats_total = {
        'parsed': len(chunks),
        'new_meta': 0,
        'new_embeddings': 0,
        'new_bm25': 0,
        'skipped_existing': 0,
        'failed_batches': 0,
    }
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
        s = build_embedding_index(chunks, store, faiss_write_y=y, embed_workers=getattr(args, 'embed_workers', 1))
        stats_total['parsed'] = max(stats_total['parsed'], s.get('parsed', 0))
        for key in ('new_meta', 'new_embeddings', 'skipped_existing', 'failed_batches'):
            stats_total[key] += s.get(key, 0)
    if build_bm25:
        s = build_bm25_index(chunks, store)
        stats_total['new_bm25'] += s.get('new_bm25', 0)
    print(
        "Done. ✅  "
        f"parsed={stats_total['parsed']}, "
        f"new_meta={stats_total['new_meta']}, "
        f"new_embeddings={stats_total['new_embeddings']}, "
        f"skipped_existing={stats_total['skipped_existing']}, "
        f"failed_batches={stats_total['failed_batches']}, "
        f"new_bm25={stats_total['new_bm25']}"
    )


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
    ap_j.add_argument("--embed-workers", type=int, default=int(os.getenv("RAG_EMBED_WORKERS", "1")), help="Number of embedding workers (default from RAG_EMBED_WORKERS or 1)")
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
