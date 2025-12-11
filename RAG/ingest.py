from __future__ import annotations
import json, os, time
from typing import List, Optional, Tuple
from dataclasses import dataclass

try:
    from RAG.models import Chunk
except ModuleNotFoundError:
    from models import Chunk


def _try_parse_json_line(line: str) -> Optional[dict]:
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except Exception:
        pass
    s = line.rstrip(", ")
    try:
        return json.loads(s)
    except Exception:
        pass
    if not s.startswith("{") and s.endswith("}"):
        try:
            return json.loads("{" + s + "}")
        except Exception:
            pass
    if not s.startswith("{") and not s.endswith("}"):
        try:
            return json.loads("{" + s + "}")
        except Exception:
            pass
    return None


def ingest_posts_jsonl(posts_path: str) -> List[Chunk]:
    chunks: List[Chunk] = []
    with open(posts_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            obj = _try_parse_json_line(line)
            if not obj:
                continue
            pid = (obj.get("id") or os.urandom(6).hex())
            title = (obj.get("title") or "").strip() or f"reddit_post_{pid}"
            selftext = (obj.get("selftext") or "").strip()
            body = (title + "\n\n" + selftext).strip() if selftext else title
            url = (obj.get("url") or "").strip()
            subreddit = (obj.get("subreddit") or "").strip()
            flair = (obj.get("link_flair_text") or "").strip()
            author = (obj.get("author") or "").strip()
            section = (f"r/{subreddit} [{flair}]" if flair else f"r/{subreddit}").strip() or "post"
            created = obj.get("created_utc")
            try:
                if isinstance(created, (int, float)):
                    updated_at = time.strftime("%Y-%m-%d", time.gmtime(int(created)))
                else:
                    updated_at = time.strftime("%Y-%m-%d", time.gmtime())
            except Exception:
                updated_at = time.strftime("%Y-%m-%d", time.gmtime())

            source = url if url else (f"reddit://{subreddit}/{pid}" if subreddit else f"reddit://{pid}")
            cid = str(pid)
            chunks.append(Chunk(
                id=cid,
                doc_id=pid,
                title=title,
                source=source,
                section=section,
                text=body,
                updated_at=updated_at,
                author=author if author else None
            ))
    return chunks


def ingest_comments_jsonl(comments_path: str) -> List[Chunk]:
    chunks: List[Chunk] = []
    with open(comments_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            obj = _try_parse_json_line(line)
            if not obj:
                continue
            cid_raw = (obj.get("id") or os.urandom(6).hex())
            body = (obj.get("body") or "").strip()
            if not body:
                continue
            subreddit = (obj.get("subreddit") or "").strip()
            author = (obj.get("author") or "").strip()
            link_id_raw = (obj.get("link_id") or "").strip()
            # Remove Reddit's "t3_" prefix from link_id to get the post ID
            post_id = link_id_raw.replace("t3_", "") if link_id_raw.startswith("t3_") else link_id_raw
            section = f"r/{subreddit} comment".strip() or "comment"
            created = obj.get("created_utc")
            try:
                if isinstance(created, (int, float)):
                    updated_at = time.strftime("%Y-%m-%d", time.gmtime(int(created)))
                else:
                    updated_at = time.strftime("%Y-%m-%d", time.gmtime())
            except Exception:
                updated_at = time.strftime("%Y-%m-%d", time.gmtime())

            title = f"comment_{cid_raw}"
            source = f"reddit://{subreddit}/comment/{cid_raw}" if subreddit else f"reddit://comment/{cid_raw}"
            cid = str(cid_raw)
            chunks.append(Chunk(
                id=cid,
                doc_id=cid_raw,
                title=title,
                source=source,
                section=section,
                text=body,
                updated_at=updated_at,
                author=author if author else None,
                post_id=post_id if post_id else None
            ))
    return chunks


