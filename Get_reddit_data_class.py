#!/usr/bin/env python3
"""
Arctic Shift subreddit downloader (class-based, simple).

Endpoints per README:
  - /api/posts/search
  - /api/comments/search
  - /api/subreddits/search (optional helper)

Examples:
  # posts for r/programming on 2025-09-13 UTC
  python get_data_class.py -s programming --after 2025-09-13 --before 2025-09-14 --what submissions

  # posts + comments
  python get_data_class.py -s programming --after 2025-09-13 --before 2025-09-14 --what both
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import requests


BASE = "https://arctic-shift.photon-reddit.com"


def _iso_date(s: str) -> str:
    """Accept YYYY-MM-DD or ISO with time and return normalized string."""
    try:
        return dt.datetime.fromisoformat(s).isoformat() if "T" in s else dt.date.fromisoformat(s).isoformat()
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid date/time: {s}") from e


class ArcticShiftAPI:
    """Minimal client for Arctic Shift HTTP API (no pagination)."""

    def __init__(self, base_url: str = BASE, timeout: int = 60, session: Optional[requests.Session] = None):
        self.base = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()

    # ---------- low-level ----------
    def _get_json(self, path: str, params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        url = f"{self.base}{path}"
        r = self.session.get(url, params=params, timeout=self.timeout)
        r.raise_for_status()
        return r.json(), r.headers

    @staticmethod
    def _norm_fields(fields: Optional[Union[List[str], str]]) -> Optional[str]:
        if fields is None:
            return None
        if isinstance(fields, str):
            return fields
        return ",".join(fields)

    # ---------- high-level: search ----------
    def search_posts(
        self,
        *,
        subreddit: Optional[str] = None,
        author: Optional[str] = None,
        after: Optional[str] = None,
        before: Optional[str] = None,
        limit: Union[int, str] = "auto",
        fields: Optional[Union[List[str], str]] = None,
        **filters: Any,
    ) -> List[Dict[str, Any]]:
        """
        Call /api/posts/search (limit: 1..100 or 'auto' per README).
        Extra filters like title, selftext, link_flair_text are passed via **filters.
        """
        params: Dict[str, Any] = {}
        if subreddit: params["subreddit"] = subreddit.lstrip("r/")
        if author:    params["author"] = author.lstrip("u/")
        if after:     params["after"] = after
        if before:    params["before"] = before
        if limit is not None: params["limit"] = limit  # int or "auto"
        if fields:    params["fields"] = self._norm_fields(fields)
        params.update(filters)
        data, headers = self._get_json("/api/posts/search", params)
        self._maybe_print_ratelimit(headers)
        return self._extract_items(data)

    def search_comments(
        self,
        *,
        subreddit: Optional[str] = None,
        author: Optional[str] = None,
        after: Optional[str] = None,
        before: Optional[str] = None,
        limit: Union[int, str] = "auto",
        fields: Optional[Union[List[str], str]] = None,
        **filters: Any,
    ) -> List[Dict[str, Any]]:
        """
        Call /api/comments/search (limit: 1..100 or 'auto').
        Extra filters like body, link_id, parent_id via **filters.
        """
        params: Dict[str, Any] = {}
        if subreddit: params["subreddit"] = subreddit.lstrip("r/")
        if author:    params["author"] = author.lstrip("u/")
        if after:     params["after"] = after
        if before:    params["before"] = before
        if limit is not None: params["limit"] = limit
        if fields:    params["fields"] = self._norm_fields(fields)
        params.update(filters)
        data, headers = self._get_json("/api/comments/search", params)
        self._maybe_print_ratelimit(headers)
        return self._extract_items(data)

    def search_subreddits(
        self,
        *,
        subreddit: Optional[str] = None,
        subreddit_prefix: Optional[str] = None,
        limit: int = 25,
        fields: Optional[Union[List[str], str]] = None,
        **filters: Any,
    ) -> List[Dict[str, Any]]:
        """Optional helper: /api/subreddits/search."""
        params: Dict[str, Any] = {"limit": limit}
        if subreddit:        params["subreddit"] = subreddit.lstrip("r/")
        if subreddit_prefix: params["subreddit_prefix"] = subreddit_prefix
        if fields:           params["fields"] = self._norm_fields(fields)
        params.update(filters)
        data, _ = self._get_json("/api/subreddits/search", params)
        return self._extract_items(data)

    # ---------- utility ----------
    @staticmethod
    def _extract_items(data: Any) -> List[Dict[str, Any]]:
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("data", "results", "items"):
                if key in data and isinstance(data[key], list):
                    return data[key]
            if all(isinstance(v, dict) for v in data.values()):
                return list(data.values())
        return []

    @staticmethod
    def _maybe_print_ratelimit(headers: Dict[str, Any]) -> None:
        rl = {k: v for k, v in headers.items() if k.lower().startswith("x-ratelimit")}
        if rl:
            print(f"[rate-limit] {rl}")

    @staticmethod
    def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ---------- convenience: download to files ----------
    def download_subreddit_window(
        self,
        *,
        subreddit: str,
        after: Optional[str],
        before: Optional[str],
        what: str = "submissions",  # "submissions" | "comments" | "both"
        post_fields: Optional[List[str]] = None,
        comment_fields: Optional[List[str]] = None,
        limit: Union[int, str] = "auto",
        outdir: Union[str, Path] = "out",
        filename_prefix: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Fetch posts/comments for a window and write NDJSON files.
        Returns dict with counts and file paths.
        """
        post_fields = post_fields or [
            "id","created_utc","author","subreddit","score","title","selftext","url","num_comments","over_18","spoiler","link_flair_text"
        ]
        comment_fields = comment_fields or [
            "id","created_utc","author","subreddit","score","body","link_id","parent_id"
        ]

        sub = subreddit.lstrip("r/")
        ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        outdir = Path(outdir)
        prefix = (filename_prefix or sub)

        result: Dict[str, Any] = {"posts": 0, "comments": 0, "files": {}}

        if what in ("submissions", "both"):
            print(f"[fetch] posts subreddit={sub} after={after} before={before} limit={limit}")
            posts = self.search_posts(subreddit=sub, after=after, before=before, limit=limit, fields=post_fields)
            fpath = outdir / f"{prefix}.posts.{ts}.jsonl"
            self.write_jsonl(fpath, posts)
            result["posts"] = len(posts)
            result["files"]["posts"] = str(fpath)

        if what in ("comments", "both"):
            print(f"[fetch] comments subreddit={sub} after={after} before={before} limit={limit}")
            comments = self.search_comments(subreddit=sub, after=after, before=before, limit=limit, fields=comment_fields)
            fpath = outdir / f"{prefix}.comments.{ts}.jsonl"
            self.write_jsonl(fpath, comments)
            result["comments"] = len(comments)
            result["files"]["comments"] = str(fpath)

        return result


# ---------------- CLI wrapper ----------------
def main():
    p = argparse.ArgumentParser(description="Download subreddit data from Arctic Shift API (class-based).")
    p.add_argument("-s", "--subreddit", required=True, help="Subreddit name (without r/)")
    p.add_argument("--after",  type=_iso_date, help="Start (YYYY-MM-DD or ISO 8601).")
    p.add_argument("--before", type=_iso_date, help="End (YYYY-MM-DD or ISO 8601).")
    p.add_argument("--what", choices=["submissions","comments","both"], default="submissions", help="What to fetch.")
    p.add_argument("--limit", default="auto", help='1-100 or "auto" (default).')
    p.add_argument("--outdir", default="out", help="Output directory.")
    p.add_argument("--prefix", default=None, help="Filename prefix (defaults to subreddit).")
    args = p.parse_args()

    api = ArcticShiftAPI()
    res = api.download_subreddit_window(
        subreddit=args.subreddit,
        after=args.after,
        before=args.before,
        what=args.what,
        limit=args.limit,
        outdir=args.outdir,
        filename_prefix=args.prefix,
    )
    print(f"[done] posts={res['posts']} comments={res['comments']}")
    for k, v in res["files"].items():
        print(f"[file] {k}: {v}")


if __name__ == "__main__":
    try:
        main()
    except requests.HTTPError as e:
        # Surface server message (trimmed) for easier debugging
        print(f"[http-error] {e.response.status_code} {e.response.text[:400]}")
        raise
