from __future__ import annotations

import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
import threading
from typing import Dict, List, Optional, Sequence, Tuple

import faiss
import numpy as np
from tqdm import tqdm

try:
    from RAG.ai import embed_texts as voyage_embed_texts
    from RAG.models import Chunk
except ModuleNotFoundError:
    from ai import embed_texts as voyage_embed_texts
    from models import Chunk


EMBED_BATCH = int(os.getenv("RAG_EMBED_BATCH", "64"))
MAX_EMBED_RETRIES = int(os.getenv("RAG_EMBED_RETRIES", "3"))


@dataclass
class BuildStats:
    parsed: int = 0
    new_embeddings: int = 0
    new_meta: int = 0
    skipped_existing: int = 0
    failed_batches: int = 0

    def as_dict(self) -> Dict[str, int]:
        return {
            "parsed": self.parsed,
            "new_embeddings": self.new_embeddings,
            "new_meta": self.new_meta,
            "skipped_existing": self.skipped_existing,
            "failed_batches": self.failed_batches,
        }


class EmbeddingStore:
    def __init__(self, store_dir: str):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)

        self.faiss_path = self.store_dir / "faiss.index"
        self.idmap_path = self.store_dir / "idmap.json"
        self.meta_path = self.store_dir / "meta.jsonl"

        self.index: Optional[faiss.Index] = None
        self.idmap: Dict[str, int] = {}
        self.meta: Dict[str, Chunk] = {}

        self._meta_dirty = False
        self._index_dirty = False
        self._idmap_dirty = False

    def load(self) -> None:
        if self.faiss_path.exists():
            self.index = faiss.read_index(str(self.faiss_path))
        if self.idmap_path.exists():
            with open(self.idmap_path, "r", encoding="utf-8") as f:
                try:
                    self.idmap = json.load(f)
                except json.JSONDecodeError:
                    self.idmap = {}
        if self.meta_path.exists():
            with open(self.meta_path, "r", encoding="utf-8") as f:
                for index, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    self.meta[data["id"]] = Chunk(**data)
                    if data["id"] not in self.idmap:
                        self.idmap[data["id"]] = index
                        self._idmap_dirty = True

        # Repair obvious inconsistencies
        if self.index is not None and len(self.idmap) != self.index.ntotal:
            # If sizes mismatch, rebuild idmap from sorted ids we still have metadata for
            # to prevent downstream crashes. Subsequent runs will overwrite.
            bad_ids = set(self.idmap.keys()) - set(self.meta.keys())
            for bid in bad_ids:
                self.idmap.pop(bid, None)
                self._idmap_dirty = True

    def upsert_metadata(self, chunk: Chunk) -> bool:
        current = self.meta.get(chunk.id)
        if current == chunk:
            return False
        # If the incoming chunk looks newer (different updated_at) or content changed, overwrite.
        if current is None or current.updated_at != chunk.updated_at or current.text != chunk.text:
            self.meta[chunk.id] = chunk
            self._meta_dirty = True
            self._idmap_dirty = True
            return True
        # Otherwise refresh metadata fields without rewriting embeddings.
        merged = Chunk(**{**asdict(current), **asdict(chunk)})
        if merged != current:
            self.meta[chunk.id] = merged
            self._meta_dirty = True
            self._idmap_dirty = True
            return True
        return False

    def add_embeddings(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) == 0:
            return
        if vectors.ndim != 2:
            raise ValueError("Expected 2D ndarray for embeddings")
        if vectors.shape[0] != len(chunks):
            raise ValueError("Batch/vector count mismatch")

        vectors = vectors.astype("float32")
        if self.index is None:
            dim = vectors.shape[1]
            self.index = faiss.IndexFlatIP(dim)
        elif vectors.shape[1] != self.index.d:
            raise ValueError(
                f"Embedding dimension mismatch: existing={self.index.d}, new={vectors.shape[1]}"
            )

        start_row = self.index.ntotal
        self.index.add(vectors)
        for offset, chunk in enumerate(chunks):
            self.idmap[chunk.id] = start_row + offset
            # Ensure metadata is tracked even if not previously added
            self.meta.setdefault(chunk.id, chunk)
        self._index_dirty = True
        self._idmap_dirty = True

    def flush(self, *, force: bool = False) -> None:
        if not force and not (self._meta_dirty or self._index_dirty or self._idmap_dirty):
            return

        if self.index is not None and self._index_dirty:
            tmp_index = self.faiss_path.with_suffix(".index.tmp")
            faiss.write_index(self.index, str(tmp_index))
            os.replace(tmp_index, self.faiss_path)

        if self._meta_dirty:
            tmp_meta = self.meta_path.with_suffix(".jsonl.tmp")
            with open(tmp_meta, "w", encoding="utf-8") as f:
                for cid in sorted(self.meta.keys()):
                    f.write(json.dumps(asdict(self.meta[cid]), ensure_ascii=False) + "\n")
            os.replace(tmp_meta, self.meta_path)

        if self._idmap_dirty:
            tmp_idmap = self.idmap_path.with_suffix(".json.tmp")
            with open(tmp_idmap, "w", encoding="utf-8") as f:
                json.dump(self.idmap, f)
            os.replace(tmp_idmap, self.idmap_path)

        self._index_dirty = False
        self._meta_dirty = False
        self._idmap_dirty = False


class FlushScheduler:
    def __init__(self, total_batches: int, y: Optional[int]):
        self.points: List[int] = []
        if total_batches <= 0:
            return
        if y and y > 2:
            j = 1
            while j < total_batches:
                self.points.append(j)
                j = math.ceil(j * y / (y - 1))
        else:
            self.points = list(range(1, total_batches + 1))
        self._cursor = 0

    def maybe_flush(self, contiguous_committed: int) -> bool:
        if self._cursor >= len(self.points):
            return False
        if contiguous_committed < self.points[self._cursor]:
            return False
        while self._cursor < len(self.points) and contiguous_committed >= self.points[self._cursor]:
            self._cursor += 1
        return True


def _dedupe_chunks(chunks: Sequence[Chunk]) -> List[Chunk]:
    seen: Dict[str, Chunk] = {}
    for chunk in chunks:
        if not chunk.id:
            continue
        seen[chunk.id] = chunk
    return list(seen.values())


def _embed_with_retries(texts: Sequence[str], batch_size: int, log_errors: bool = True) -> np.ndarray:
    delay = 1.0
    last_err: Optional[Exception] = None
    for attempt in range(1, MAX_EMBED_RETRIES + 1):
        try:
            return voyage_embed_texts(list(texts), batch_size)
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if log_errors:
                print(f"[Attempt {attempt}/{MAX_EMBED_RETRIES}] Error occurred while embedding texts: {exc}\nRetrying in {delay:.0f}s...")
            if attempt == MAX_EMBED_RETRIES:
                break
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
    raise RuntimeError(f"Failed to embed batch after {MAX_EMBED_RETRIES} attempts: {last_err}")


def build_embedding_index(
    chunks: Sequence[Chunk],
    store_dir: str,
    *,
    faiss_write_y: Optional[int] = None,
    embed_workers: int = 1,
    batch_size: int = EMBED_BATCH,
) -> Dict[str, int]:
    stats = BuildStats(parsed=len(chunks))

    if not chunks:
        return stats.as_dict()

    store = EmbeddingStore(store_dir)
    store.load()

    deduped_chunks = _dedupe_chunks(chunks)
    existing_ids = set(store.idmap.keys())

    for chunk in deduped_chunks:
        if store.upsert_metadata(chunk):
            stats.new_meta += 1

    new_chunks: List[Chunk] = []
    for chunk in deduped_chunks:
        if chunk.id in existing_ids:
            stats.skipped_existing += 1
        else:
            new_chunks.append(chunk)

    if not new_chunks:
        store.flush()
        return stats.as_dict()

    # Sort new chunks by id to enforce deterministic processing regardless of source order
    new_chunks.sort(key=lambda c: c.id)

    chunk_batches: List[List[Chunk]] = []
    for start in range(0, len(new_chunks), batch_size):
        chunk_batches.append(new_chunks[start : start + batch_size])

    scheduler = FlushScheduler(total_batches=len(chunk_batches), y=faiss_write_y)

    pending: Dict[int, Tuple[List[Chunk], Optional[np.ndarray]]] = {}
    committed = 0
    lock = threading.Lock()

    def queue_result(batch_idx: int, chunks_batch: List[Chunk], vectors: Optional[np.ndarray]) -> None:
        nonlocal committed
        with lock:
            pending[batch_idx] = (chunks_batch, vectors)
            while committed + 1 in pending:
                next_idx = committed + 1
                chunks_ready, vecs_ready = pending.pop(next_idx)
                committed = next_idx
                if vecs_ready is None:
                    stats.failed_batches += len(chunks_ready)
                    continue
                store.add_embeddings(chunks_ready, vecs_ready)
                stats.new_embeddings += len(chunks_ready)
                if scheduler.maybe_flush(committed):
                    store.flush()

    def worker(batch_idx: int, items: List[Chunk]) -> Tuple[int, List[Chunk], Optional[np.ndarray]]:
        texts = [c.text if c.text else " " for c in items]
        try:
            vectors = _embed_with_retries(texts, batch_size)
            return batch_idx, items, vectors
        except Exception:
            return batch_idx, items, None

    with ThreadPoolExecutor(max_workers=max(1, int(embed_workers))) as executor:
        futures = {
            executor.submit(worker, idx + 1, batch): idx + 1
            for idx, batch in enumerate(chunk_batches)
        }

        with tqdm(total=len(new_chunks), desc="Embedding", unit="chunk") as pbar:
            for future in as_completed(futures):
                idx, items, vectors = future.result()
                queue_result(idx, items, vectors)
                if vectors is not None:
                    pbar.update(len(items))

    # Ensure all buffered state is persisted
    store.flush(force=True)

    return stats.as_dict()


__all__ = ["build_embedding_index", "EMBED_BATCH"]


