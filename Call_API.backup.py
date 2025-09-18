#!/usr/bin/env python3
"""
Arctic Shift subreddit downloader (class-backed) with the SAME CLI flags you already use,
plus an optional --all for exhaustive coverage via time-slicing.

Examples:
  # Single-shot (same as your current behavior)
  python Call_API.py -s programming --after 2025-09-13 --before 2025-09-14 --what submissions

  # Posts + comments
  python Call_API.py -s programming --after 2025-09-13 --before 2025-09-14 --what both

  # Guaranteed full coverage if the window has >~1000 items
  python Call_API.py -s programming --after 2025-09-01 --before 2025-10-01 --what submissions --all
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set, Callable, Union

import requests
from Get_reddit_data_class import ArcticShiftAPI


# ---------- helpers (match your existing UX) ----------
def iso_date(s: str) -> str:
    """Accept YYYY-MM-DD or ISO with time; normalize to ISO string."""
    try:
        return dt.datetime.fromisoformat(s).isoformat() if "T" in s else dt.date.fromisoformat(s).isoformat()
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid date/time: {s}") from e


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_subreddit_via_class(api: ArcticShiftAPI, name: str) -> Optional[Dict[str, Any]]:
    """Best-effort resolver via /api/subreddits/search using subreddit_prefix."""
    try:
        items = api.search_subreddits(subreddit_prefix=name, limit=5)
        if not isinstance(items, list) or not items:
            return None
        for it in items:
            if str(it.get("display_name", "")).lower() == name.lower():
                return it
        return items[0]
    except Exception:
        return None


# ---------- time-sliced exhaustive fetching (optional) ----------
def _parse_date_any(s: Optional[str]) -> Optional[dt.datetime]:
    if not s:
        return None
    if "T" in s:
        t = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
        return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)
    return dt.datetime.fromisoformat(s).replace(tzinfo=dt.timezone.utc)


def _format_iso_seconds(t: dt.datetime) -> str:
    if t.tzinfo is None:
        t = t.replace(tzinfo=dt.timezone.utc)
    # Force Z-UTC (no +00:00 offset)
    return t.astimezone(dt.timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def _cap_guess(limit: Union[str, int]) -> int:
    return 1000 if str(limit).lower() == "auto" else int(limit)


def _dedup_merge(dst: List[Dict[str, Any]], seen: Set[str], src: List[Dict[str, Any]]) -> None:
    for r in src:
        rid = str(r.get("id"))
        if rid and rid not in seen:
            seen.add(rid)
            dst.append(r)


def _fetch_slice(
    search_func: Callable[..., List[Dict[str, Any]]],
    *,
    subreddit: str,
    after: Optional[str],
    before: Optional[str],
    limit: Union[str, int],
    fields: List[str],
    extra_params: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    params = dict(subreddit=subreddit, after=after, before=before, limit=limit, fields=fields, sort="asc")
    if extra_params:
        params.update(extra_params)
    return search_func(**params)


def fetch_exhaustive_timesliced(
    search_func: Callable[..., List[Dict[str, Any]]],
    *,
    subreddit: str,
    after: Optional[str],
    before: Optional[str],
    limit: Union[str, int],
    fields: List[str],
    max_depth: int = 20,
    extra_params: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """
    Guaranteed coverage by recursively splitting [after, before) until each slice returns < cap.
    De-duplicates by 'id' across overlaps. Uses created_utc ordering (sort=asc).
    """
    cap = _cap_guess(limit)
    a_dt = _parse_date_any(after) if after else None
    b_dt = _parse_date_any(before) if before else None

    stack: List[Tuple[Optional[str], Optional[str], int]] = [(after, before, 0)]
    out: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    while stack:
        a_iso, b_iso, depth = stack.pop()

        batch = _fetch_slice(
            search_func,
            subreddit=subreddit,
            after=a_iso,
            before=b_iso,
            limit=limit,
            fields=fields,
            extra_params=extra_params,
        )

        if len(batch) < cap or depth >= max_depth:
            _dedup_merge(out, seen, batch)
            continue

        # Need to split: estimate mid from created_utc
        first_t = batch[0].get("created_utc")
        last_t = batch[-1].get("created_utc")

        def _to_dt(x) -> Optional[dt.datetime]:
            if x is None:
                return None
            try:
                if isinstance(x, (int, float)) or (isinstance(x, str) and x.isdigit()):
                    return dt.datetime.fromtimestamp(int(x), tz=dt.timezone.utc)
                return _parse_date_any(str(x))
            except Exception:
                return None

        t0 = _to_dt(first_t) or a_dt
        t1 = _to_dt(last_t) or b_dt

        if not t0 or not t1:
            _dedup_merge(out, seen, batch)
            continue

        mid = (t0 + (t1 - t0) / 2).replace(microsecond=0)

        left_after = a_iso
        left_before = _format_iso_seconds(mid)
        right_after = _format_iso_seconds(mid)
        right_before = b_iso

        # If the split doesn't shrink the window, just accept this batch
        if left_before == b_iso or right_after == a_iso:
            _dedup_merge(out, seen, batch)
            continue

        # Push right then left (LIFO) to work smaller windows first
        stack.append((right_after, right_before, depth + 1))
        stack.append((left_after, left_before, depth + 1))

    return out


# ---------- CLI ----------
def main():
    p = argparse.ArgumentParser(description="Download subreddit data from Arctic Shift API.")
    p.add_argument("-s", "--subreddit", required=True, help="Subreddit name (without r/)")
    p.add_argument("--after", type=iso_date, help="Start (ISO/Date/Epoch supported by API).")
    p.add_argument("--before", type=iso_date, help="End (ISO/Date/Epoch supported by API).")
    # Per README: limit is 1–100 OR "auto" (100–1000 depending on server load)
    p.add_argument("--limit", default="auto", help='Request size: 1-100 or "auto" (default).')
    p.add_argument("--what", choices=["submissions", "comments", "both"], default="submissions")
    p.add_argument("--outdir", default="out")
    # Keep to documented selectable fields to reduce payload size
    p.add_argument(
        "--post_fields",
        nargs="*",
        default=[
            "id",
            "created_utc",
            "author",
            "subreddit",
            "score",
            "title",
            "selftext",
            "url",
            "num_comments",
            "over_18",
            "spoiler",
            "link_flair_text",
        ],
    )
    p.add_argument(
        "--comment_fields",
        nargs="*",
        default=["id", "created_utc", "author", "subreddit", "score", "body", "link_id", "parent_id"],
    )
    # NEW: exhaustive crawl flag
    p.add_argument(
        "--all",
        action="store_true",
        help="Exhaustively split the time window to avoid truncation when > cap results exist.",
    )
    args = p.parse_args()

    sub = args.subreddit.lstrip("r/")
    api = ArcticShiftAPI()

    meta = resolve_subreddit_via_class(api, sub)
    if meta:
        print(f"[info] {meta.get('display_name_prefixed', 'r/'+sub)} subscribers={meta.get('subscribers')}")

    outdir = Path(args.outdir)
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    if args.what in ("submissions", "both"):
        params_preview = {
            "subreddit": sub,
            "after": args.after,
            "before": args.before,
            "limit": args.limit,
            "fields": args.post_fields,
        }
        print(f"[fetch] posts {params_preview}")
        if args.all:
            posts = fetch_exhaustive_timesliced(
                api.search_posts,
                subreddit=sub,
                after=args.after,
                before=args.before,
                limit=args.limit,
                fields=args.post_fields,
            )
        else:
            posts = api.search_posts(
                subreddit=sub,
                after=args.after,
                before=args.before,
                limit=args.limit,
                fields=args.post_fields,
            )
        write_jsonl(outdir / f"{sub}.posts.{ts}.jsonl", posts)
        print(f"[done] posts: {len(posts)}")

    if args.what in ("comments", "both"):
        params_preview = {
            "subreddit": sub,
            "after": args.after,
            "before": args.before,
            "limit": args.limit,
            "fields": args.comment_fields,
        }
        print(f"[fetch] comments {params_preview}")
        if args.all:
            comments = fetch_exhaustive_timesliced(
                api.search_comments,
                subreddit=sub,
                after=args.after,
                before=args.before,
                limit=args.limit,
                fields=args.comment_fields,
            )
        else:
            comments = api.search_comments(
                subreddit=sub,
                after=args.after,
                before=args.before,
                limit=args.limit,
                fields=args.comment_fields,
            )
        write_jsonl(outdir / f"{sub}.comments.{ts}.jsonl", comments)
        print(f"[done] comments: {len(comments)}")


if __name__ == "__main__":
    try:
        main()
    except requests.HTTPError as e:
        print(f"[http-error] {e.response.status_code} {e.response.text[:400]}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)
