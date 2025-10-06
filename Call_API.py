#!/usr/bin/env python3
"""
Arctic Shift subreddit downloader (class-backed) with:
- SAME CLI flags as before (+ --all for exhaustive paging),
- NO timestamp in output filenames,
- INCREMENTAL appends: re-runs fetch only missing hourly slices of the requested window,
  based on what's already present in the existing JSONL file,
- HUMAN-READABLE LOGGING: every request prints a compact, readable line with params + result count,
  every covered slice prints a human-readable skip, and summaries are friendly,
- END-OF-RUN VALIDATION: checks that IDs are unique in the output files and reports status.

Examples:
  # Single-shot (will append new data, skipping already-covered hours)
  python Call_API.py -s programming --after 2025-09-13 --before 2025-09-14 --what submissions

  # Posts + comments, time-sliced to ensure no misses; still incremental
  python Call_API.py -s hair --after 2025-09-01 --before 2025-10-01 --what both --all
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set, Callable, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from Get_reddit_data_class import ArcticShiftAPI


# ---------- helpers (match your existing UX) ----------
def iso_date(s: str) -> str:
    """Accept YYYY-MM-DD or ISO with time; normalize to ISO string."""
    try:
        return dt.datetime.fromisoformat(s).isoformat() if "T" in s else dt.date.fromisoformat(s).isoformat()
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid date/time: {s}") from e


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


def write_jsonl_append_dedup(path: Path, rows: List[Dict[str, Any]]) -> int:
    """
    Append rows to JSONL, de-duplicating by 'id' against existing file contents.
    Returns number of new rows written.
    """
    existing_ids: Set[str] = set()
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                    rid = str(r.get("id"))
                    if rid:
                        existing_ids.add(rid)
                except Exception:
                    continue

    new_count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            rid = str(r.get("id"))
            if rid and rid in existing_ids:
                continue
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if rid:
                existing_ids.add(rid)
            new_count += 1
    return new_count


# ---------- human-readable logging ----------
def _kind(which: str) -> str:
    return "posts" if "post" in which else "comments"


def _fmt_int(n: Optional[int]) -> str:
    try:
        return f"{int(n):,}"
    except Exception:
        return str(n)


def _fmt_dt(s: Optional[str]) -> Optional[dt.datetime]:
    if not s:
        return None
    try:
        if "T" in s:
            t = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
            return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)
        return dt.datetime.fromisoformat(s).replace(tzinfo=dt.timezone.utc)
    except Exception:
        return None


def _fmt_range_human(after_iso: Optional[str], before_iso: Optional[str]) -> str:
    a = _fmt_dt(after_iso)
    b = _fmt_dt(before_iso)
    def fmt(t: Optional[dt.datetime]) -> str:
        if not t:
            return "?"
        return t.astimezone(dt.timezone.utc).strftime("%Y-%m-%d %H:%M")
    return f"{fmt(a)} → {fmt(b)} UTC"


def _fields_summary(fields: Any) -> str:
    if isinstance(fields, list):
        preview = ", ".join(fields[:3]) + (", …" if len(fields) > 3 else "")
        return f"{len(fields)} fields ({preview})"
    if isinstance(fields, str):
        parts = [p.strip() for p in fields.split(",") if p.strip()]
        preview = ", ".join(parts[:3]) + (", …" if len(parts) > 3 else "")
        return f"{len(parts)} fields ({preview})"
    return "fields=?"

def _log_request(which: str, params: Dict[str, Any], results: Optional[int] = None) -> None:
    kind = _kind(which)
    rng  = _fmt_range_human(params.get("after"), params.get("before"))
    sub  = params.get("subreddit")
    lim  = params.get("limit")
    srt  = params.get("sort", "asc")
    fld  = _fields_summary(params.get("fields"))
    res  = f" — results={_fmt_int(results)}" if results is not None else ""
    print(f"🔎 Request {kind}: r/{sub} | {rng} | limit={lim} | sort={srt} | {fld}{res}")

def _log_skip(which: str, after_iso: str, before_iso: str, reason: str = "covered") -> None:
    kind = _kind(which)
    rng  = _fmt_range_human(after_iso, before_iso)
    print(f"⏭️  Skip {kind}: {rng} — {reason}")

def _log_done(kind: str, count: int, path: Path) -> None:
    print(f"✅ Appended {_fmt_int(count)} {kind} → {path}")

def _log_up_to_date(kind: str) -> None:
    print(f"⏭️  {kind.capitalize()}: up to date for requested window")

def _log_validate(label: str, total: int, unique: int, dup: int, examples: List[str]) -> None:
    status = "OK" if dup == 0 else "DUPLICATES_FOUND"
    base = f"🧪 Validate {label}: lines={_fmt_int(total)}, unique_ids={_fmt_int(unique)}, duplicate_ids={_fmt_int(dup)} — {status}"
    if dup == 0:
        print(base)
    else:
        print(base + f" (examples: {examples[:10]})")


# ---------- time & bins ----------
# Use true 1-hour bins for coverage detection to avoid gaps/overfetching
BIN_SECONDS = 24 * 3600  # 1 hour bins to detect coverage & gaps

def _epoch_from_created(created: Any) -> Optional[int]:
    """created_utc can be epoch seconds or ISO; return epoch seconds int."""
    if created is None:
        return None
    if isinstance(created, (int, float)) or (isinstance(created, str) and created.isdigit()):
        try:
            return int(float(created))
        except Exception:
            return None
    t = _fmt_dt(str(created))
    return int(t.timestamp()) if t else None

def _hour_bin(ts_epoch: int) -> int:
    return ts_epoch // BIN_SECONDS

def _format_iso_Z(epoch_sec: int) -> str:
    return dt.datetime.utcfromtimestamp(epoch_sec).strftime("%Y-%m-%dT%H:%M:%SZ")

def _format_iso_seconds(t: dt.datetime) -> str:
    """Force Z-UTC (no +00:00 offset) for API friendliness."""
    if t.tzinfo is None:
        t = t.replace(tzinfo=dt.timezone.utc)
    return t.astimezone(dt.timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")

def _bins_from_file(jsonl_path: Path) -> Set[int]:
    """Scan existing JSONL and return set of covered hour bins (by created_utc)."""
    covered: Set[int] = set()
    if not jsonl_path.exists():
        return covered
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            ep = _epoch_from_created(r.get("created_utc"))
            if ep is None:
                continue
            covered.add(_hour_bin(ep))
    return covered

def _coverage_sidecar_path(jsonl_path: Path) -> Path:
    """Return path to a sidecar JSON file that stores explicitly covered bins."""
    return jsonl_path.with_suffix(jsonl_path.suffix + ".coverage.json")

def _read_covered_bins(path: Path) -> Set[int]:
    """Read covered bin indices from sidecar file, if present."""
    if not path.exists():
        return set()
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return {int(x) for x in data}
        return set()
    except Exception:
        return set()

# ---------- interval coverage (bin-agnostic) ----------
def _normalize_iso(s: Optional[str]) -> Optional[str]:
    t = _fmt_dt(s) if s else None
    return _format_iso_seconds(t) if t else None

def _read_covered_intervals(path: Path) -> List[Tuple[str, str]]:
    """Read covered intervals from sidecar. Supports both interval and legacy bin formats."""
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        # New format: {"intervals": [[after, before], ...]}
        if isinstance(data, dict) and isinstance(data.get("intervals"), list):
            out: List[Tuple[str, str]] = []
            for pair in data["intervals"]:
                if isinstance(pair, (list, tuple)) and len(pair) == 2:
                    a = _normalize_iso(pair[0])
                    b = _normalize_iso(pair[1])
                    if a and b:
                        out.append((a, b))
            return _merge_intervals(out)
        # Legacy: list of bins (hours or days depending on historical BIN_SECONDS)
        # Heuristic per bin: if value < 100_000, treat as days; else hours.
        if isinstance(data, list) and all(isinstance(x, (int, float)) for x in data):
            intervals: List[Tuple[str, str]] = []
            for raw in data:
                b = int(raw)
                seconds_per_bin = 86400 if b < 100_000 else 3600
                start_iso = _format_iso_Z(b * seconds_per_bin)
                end_iso   = _format_iso_Z((b + 1) * seconds_per_bin)
                intervals.append((start_iso, end_iso))
            return _merge_intervals(intervals)
    except Exception:
        return []

def _write_covered_intervals(path: Path, intervals: List[Tuple[str, str]]) -> None:
    merged = _merge_intervals(intervals)
    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump({"intervals": merged}, f, ensure_ascii=False)
    except Exception:
        pass

def _merge_intervals(intervals: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    if not intervals:
        return []
    items: List[Tuple[dt.datetime, dt.datetime]] = []
    for a, b in intervals:
        ta = _fmt_dt(a)
        tb = _fmt_dt(b)
        if not ta or not tb:
            continue
        items.append((ta, tb))
    items.sort(key=lambda x: x[0])
    merged: List[Tuple[dt.datetime, dt.datetime]] = []
    for a, b in items:
        if not merged:
            merged.append((a, b))
            continue
        la, lb = merged[-1]
        if a <= lb:  # overlap or touch
            merged[-1] = (la, max(lb, b))
        else:
            merged.append((a, b))
    return [(_format_iso_seconds(a), _format_iso_seconds(b)) for a, b in merged]

def _subtract_intervals(total: List[Tuple[str, str]], covered: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Return gaps of total minus covered (all in ISO)."""
    cov = _merge_intervals(covered)
    gaps: List[Tuple[str, str]] = []
    for a, b in total:
        ta = _fmt_dt(a)
        tb = _fmt_dt(b)
        if not ta or not tb or tb <= ta:
            continue
        cursor = ta
        for ca, cb in [(x[0], x[1]) for x in cov]:
            cta = _fmt_dt(ca)
            ctb = _fmt_dt(cb)
            if not cta or not ctb:
                continue
            if ctb <= cursor:
                continue
            if cta >= tb:
                break
            if cta > cursor:
                gaps.append((_format_iso_seconds(cursor), _format_iso_seconds(min(cta, tb))))
            cursor = max(cursor, ctb)
            if cursor >= tb:
                break
        if cursor < tb:
            gaps.append((_format_iso_seconds(cursor), _format_iso_seconds(tb)))
    return gaps

def _min_max_created_in_window(jsonl_path: Path, after_iso: Optional[str], before_iso: Optional[str]) -> Optional[Tuple[str, str]]:
    """Scan JSONL and compute [min_created, max_created] within the requested window.
    Returns None if no rows in window."""
    if not jsonl_path.exists():
        return None
    a_dt = _fmt_dt(after_iso) if after_iso else None
    b_dt = _fmt_dt(before_iso) if before_iso else None
    min_ep: Optional[int] = None
    max_ep: Optional[int] = None
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            ep = _epoch_from_created(r.get("created_utc"))
            if ep is None:
                continue
            t = dt.datetime.utcfromtimestamp(ep).replace(tzinfo=dt.timezone.utc)
            if a_dt and t < a_dt:
                continue
            if b_dt and t >= b_dt:
                continue
            if min_ep is None or ep < min_ep:
                min_ep = ep
            if max_ep is None or ep > max_ep:
                max_ep = ep
    if min_ep is None or max_ep is None:
        return None
    return _format_iso_Z(min_ep), _format_iso_Z(max_ep)

def _split_intervals_for_workers(intervals: List[Tuple[str, str]], workers: int) -> List[Tuple[str, str]]:
    """Split a set of [after,before) intervals into approximately equal-duration pieces.
    Ensures the total number of pieces equals 'workers'. Splits the largest intervals first
    proportionally so that all pieces are roughly the same size.
    """
    if workers <= 1 or not intervals:
        return intervals
    # Normalize and compute durations
    norm: List[Tuple[dt.datetime, dt.datetime, float]] = []
    for a_iso, b_iso in intervals:
        a = _fmt_dt(a_iso)
        b = _fmt_dt(b_iso)
        if not a or not b or b <= a:
            continue
        dur = (b - a).total_seconds()
        if dur > 0:
            norm.append((a, b, dur))
    if not norm:
        return intervals
    # Sort by duration desc (largest first)
    norm.sort(key=lambda x: x[2], reverse=True)
    total = sum(d for _, _, d in norm)
    remaining_workers = max(1, workers)
    remaining_total = total
    pieces: List[Tuple[str, str]] = []
    for idx, (a, b, dur) in enumerate(norm):
        rem_intervals = len(norm) - idx - 1
        if remaining_workers <= 0:
            break
        if idx == len(norm) - 1:
            count = remaining_workers
        else:
            # Proportional allocation, at least 1, leave at least 1 for each remaining interval
            ideal = dur / remaining_total * remaining_workers if remaining_total > 0 else 1
            count = max(1, int(round(ideal)))
            count = min(count, remaining_workers - max(0, rem_intervals))
            count = max(1, count)
        # Split [a,b) into 'count' pieces
        step = (b - a) / count
        for i in range(count):
            start = a + step * i
            end = a + step * (i + 1)
            pieces.append((_format_iso_seconds(start), _format_iso_seconds(end)))
        remaining_workers -= count
        remaining_total -= dur
    return pieces if pieces else intervals

def _hour_bins_for_range(after_iso: Optional[str], before_iso: Optional[str]) -> List[int]:
    """Return list of hour bins for [after, before) (UTC)."""
    if not after_iso or not before_iso:
        return []
    a = _fmt_dt(after_iso)
    b = _fmt_dt(before_iso)
    if not a or not b:
        return []
    a_epoch = int(a.timestamp())
    b_epoch = int(b.timestamp())
    if b_epoch <= a_epoch:
        return []
    start_bin = _hour_bin(a_epoch)
    end_bin_exclusive = _hour_bin(b_epoch - 1) + 1
    return list(range(start_bin, end_bin_exclusive))

# ---------- exhaustive fetching (pager) ----------
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
    """
    One slice request used by the pager. Logs with human-readable line AFTER the call.
    """
    params = dict(subreddit=subreddit, after=after, before=before, limit=limit, fields=fields, sort="asc")
    if extra_params:
        params.update(extra_params)
    rows = search_func(**params)
    _log_request(search_func.__name__, params, results=len(rows))
    return rows

def fetch_exhaustive_timesliced(
    search_func: Callable[..., List[Dict[str, Any]]],
    *,
    subreddit: str,
    after: Optional[str],
    before: Optional[str],
    limit: Union[str, int],
    fields: List[str],
    max_depth: int = 100,
    extra_params: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """
    Guaranteed coverage by recursively splitting [after, before) until each slice returns < cap.
    De-duplicates by 'id' across overlaps. Uses created_utc ordering (sort=asc).
    """
    cap = _cap_guess(limit)
    a_dt = _fmt_dt(after) if after else None
    b_dt = _fmt_dt(before) if before else None

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
        last_t  = batch[-1].get("created_utc")

        def _to_dt(x) -> Optional[dt.datetime]:
            ep = _epoch_from_created(x)
            return dt.datetime.fromtimestamp(ep, tz=dt.timezone.utc) if ep is not None else None

        t0 = _to_dt(first_t) or a_dt
        t1 = _to_dt(last_t)  or b_dt

        if not t0 or not t1:
            _dedup_merge(out, seen, batch)
            continue

        mid = (t0 + (t1 - t0) / 2).replace(microsecond=0)

        # Split exactly at mid without artificial overlap; dedup + interval coverage handle edges
        left_after   = a_iso
        left_before  = _format_iso_seconds(mid)
        right_after  = _format_iso_seconds(mid)
        right_before = b_iso

        # If the split doesn't shrink the window, just accept this batch
        if left_before == b_iso or right_after == a_iso:
            _dedup_merge(out, seen, batch)
            continue

        # Push right then left (LIFO) to work smaller windows first
        stack.append((right_after, right_before, depth + 1))
        stack.append((left_after,  left_before,  depth + 1))

    return out


# ---------- successive paging for auto limit ----------
def fetch_successive_paged(
    search_func: Callable[..., List[Dict[str, Any]]],
    *,
    subreddit: str,
    after: Optional[str],
    before: Optional[str],
    limit: Union[str, int],
    fields: List[str],
    extra_params: Dict[str, Any] | None = None,
    max_pages: int = 10_000,
) -> List[Dict[str, Any]]:
    """
    Forward page using created_utc boundary when limit='auto'.
    Continues requesting with after := last(created_utc) until no rows or after >= before.
    De-duplicates by id across pages.
    """
    out: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    current_after = after
    before_dt = _fmt_dt(before) if before else None

    for _ in range(max_pages):
        batch = _fetch_slice(
            search_func,
            subreddit=subreddit,
            after=current_after,
            before=before,
            limit=limit,
            fields=fields,
            extra_params=extra_params,
        )
        if not batch:
            break

        _dedup_merge(out, seen, batch)

        last_ct = batch[-1].get("created_utc")
        last_ep = _epoch_from_created(last_ct)
        if last_ep is None:
            break
        next_after = _format_iso_Z(last_ep)
        if current_after and next_after == current_after:
            # No progress on boundary; avoid infinite loop
            break
        current_after = next_after

        if before_dt is not None:
            ca_dt = _fmt_dt(current_after)
            if ca_dt and ca_dt >= before_dt:
                break

    return out

# ---------- simple call wrapper (non-pager) ----------
def _call_with_log(which: str, func: Callable[..., List[Dict[str, Any]]], params: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = func(**params)
    _log_request(which, params, results=len(rows))
    return rows


# ---------- end-of-run validation ----------
def _validate_unique_ids(jsonl_path: Path, label: str) -> None:
    """
    Scan a JSONL file and verify that 'id' values are unique.
    Prints a summary line and, if duplicates found, prints up to 10 example IDs.
    """
    if not jsonl_path.exists():
        print(f"🧪 Validate {label}: file not found ({jsonl_path})")
        return
    seen: Set[str] = set()
    dups: Set[str] = set()
    total = 0
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            try:
                obj = json.loads(line)
            except Exception:
                continue
            rid = obj.get("id")
            if rid is None:
                continue
            rid = str(rid)
            if rid in seen:
                dups.add(rid)
            else:
                seen.add(rid)
    _log_validate(label, total, len(seen), len(dups), list(dups))


# ---------- CLI ----------
def main():
    p = argparse.ArgumentParser(description="Download subreddit data from Arctic Shift API (incremental, no timestamp filenames).")
    p.add_argument("-s", "--subreddit", required=True, help="Subreddit name (without r/)")
    p.add_argument("--after", type=iso_date, help="Start (ISO/Date/Epoch supported by API).")
    p.add_argument("--before", type=iso_date, help="End (ISO/Date/Epoch supported by API).")
    # Per README: limit is 1–100 OR "auto" (100–1000 depending on server load)
    p.add_argument("--limit", default="auto", help='Request size: 1-100 or "auto" (default).')
    p.add_argument("--what", choices=["submissions", "comments", "both"], default="submissions")
    p.add_argument("--outdir", default="out")
    p.add_argument("--workers", type=int, default=4, help="Max concurrent intervals to fetch per kind")
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
    # exhaustive crawl flag
    p.add_argument(
        "--all",
        action="store_true",
        help="Exhaustively split the time window to avoid truncation when > cap results exist.",
    )
    args = p.parse_args()

    sub = args.subreddit.lstrip("r/")
    api = ArcticShiftAPI()

    # Pass through real 'auto' to leverage server-side larger batches; otherwise clamp 1-100
    if str(args.limit).lower() == "auto":
        eff_limit = "auto"
    else:
        try:
            eff_limit = max(1, min(100, int(args.limit)))
        except Exception:
            eff_limit = 100

    meta = resolve_subreddit_via_class(api, sub)
    if meta:
        print(f"ℹ️  Info: {meta.get('display_name_prefixed', 'r/'+sub)} — subscribers={_fmt_int(meta.get('subscribers'))}")

    outdir = Path(args.outdir)
    out_posts = outdir / f"{sub}.posts.jsonl"
    out_comments = outdir / f"{sub}.comments.jsonl"

    # Compute target hour-bins for requested window
    target_bins = _hour_bins_for_range(args.after, args.before)

    # -------- POSTS --------
    if args.what in ("submissions", "both"):
        sidecar_posts = _coverage_sidecar_path(out_posts)
        covered_intervals = _read_covered_intervals(sidecar_posts)
        total_interval = [(_normalize_iso(args.after), _normalize_iso(args.before))]
        # Seed coverage for auto-mode using min/max created_utc already in JSONL (within window)
        if eff_limit == "auto":
            observed = _min_max_created_in_window(out_posts, args.after, args.before)
            if observed:
                covered_intervals.append(observed)
        missing_intervals = _subtract_intervals(total_interval, covered_intervals)

        # Log covered intervals
        for a_iso, b_iso in _merge_intervals(covered_intervals):
            _log_skip("posts", a_iso, b_iso, reason="covered")

        total_new = 0
        if missing_intervals:
            def _post_worker(interval: Tuple[str, str]) -> Tuple[List[Dict[str, Any]], Tuple[str, str]]:
                a_iso, b_iso = interval
                worker_api = ArcticShiftAPI()
                if eff_limit == "auto":
                    rows = fetch_successive_paged(
                        worker_api.search_posts, subreddit=sub, after=a_iso, before=b_iso,
                        limit=eff_limit, fields=args.post_fields,
                    )
                else:
                    rows = fetch_exhaustive_timesliced(
                        worker_api.search_posts, subreddit=sub, after=a_iso, before=b_iso,
                        limit=eff_limit, fields=args.post_fields,
                    )
                return rows, interval

            # If limit='auto' and multiple workers, split intervals into ~equal pieces
            plan = missing_intervals
            if eff_limit == "auto" and args.workers > 1:
                plan = _split_intervals_for_workers(missing_intervals, args.workers)

            with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
                futures = [ex.submit(_post_worker, iv) for iv in plan]
                for fut in as_completed(futures):
                    rows, iv = fut.result()
                    total_new += write_jsonl_append_dedup(out_posts, rows)
                    covered_intervals.append(iv)
                    _write_covered_intervals(sidecar_posts, covered_intervals)

        if not missing_intervals:
            _log_up_to_date("posts")
        _log_done("posts", total_new, out_posts)

    # -------- COMMENTS --------
    if args.what in ("comments", "both"):
        sidecar_comments = _coverage_sidecar_path(out_comments)
        covered_intervals = _read_covered_intervals(sidecar_comments)
        total_interval = [(_normalize_iso(args.after), _normalize_iso(args.before))]
        if eff_limit == "auto":
            observed = _min_max_created_in_window(out_comments, args.after, args.before)
            if observed:
                covered_intervals.append(observed)
        missing_intervals = _subtract_intervals(total_interval, covered_intervals)

        # Log covered intervals
        for a_iso, b_iso in _merge_intervals(covered_intervals):
            _log_skip("comments", a_iso, b_iso, reason="covered")

        total_new = 0
        if missing_intervals:
            def _comment_worker(interval: Tuple[str, str]) -> Tuple[List[Dict[str, Any]], Tuple[str, str]]:
                a_iso, b_iso = interval
                worker_api = ArcticShiftAPI()
                if eff_limit == "auto":
                    rows = fetch_successive_paged(
                        worker_api.search_comments, subreddit=sub, after=a_iso, before=b_iso,
                        limit=eff_limit, fields=args.comment_fields,
                    )
                else:
                    rows = fetch_exhaustive_timesliced(
                        worker_api.search_comments, subreddit=sub, after=a_iso, before=b_iso,
                        limit=eff_limit, fields=args.comment_fields,
                    )
                return rows, interval

            plan = missing_intervals
            if eff_limit == "auto" and args.workers > 1:
                plan = _split_intervals_for_workers(missing_intervals, args.workers)

            with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
                futures = [ex.submit(_comment_worker, iv) for iv in plan]
                for fut in as_completed(futures):
                    rows, iv = fut.result()
                    total_new += write_jsonl_append_dedup(out_comments, rows)
                    covered_intervals.append(iv)
                    _write_covered_intervals(sidecar_comments, covered_intervals)

        if not missing_intervals:
            _log_up_to_date("comments")
        _log_done("comments", total_new, out_comments)

    # -------- VALIDATION --------
    if args.what in ("submissions", "both"):
        _validate_unique_ids(out_posts, "posts")
    if args.what in ("comments", "both"):
        _validate_unique_ids(out_comments, "comments")


if __name__ == "__main__":
    try:
        main()
    except requests.HTTPError as e:
        print(f"❗ HTTP error: {e.response.status_code} {e.response.text[:400]}")
        sys.exit(2)
    except Exception as e:
        print(f"❗ Error: {e}")
        sys.exit(1)
