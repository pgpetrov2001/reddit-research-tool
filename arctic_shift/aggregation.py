from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .api import ArcticShiftAPI
from .tokens import normalize_created, parse_iso_ts


FREQUENCY_ORDER = ["week", "month", "year"]


def _cache_path(base_dir: Path, subreddit: str, kind: str, frequency: str) -> Path:
    fname = f"{subreddit}.{kind}.agg.{frequency}.json"
    return base_dir / fname


def _load_cache(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def _validate_cache(cached: Optional[Dict]) -> bool:
    if cached is None or not isinstance(cached, dict):
        return False
    coverage = cached.get("coverage")
    buckets = cached.get("buckets")
    if not isinstance(coverage, dict) or not isinstance(buckets, list):
        return False

    cover_after = coverage.get("after")
    cover_before = coverage.get("before")
    if cover_after is None or cover_before is None:
        return False
    try:
        parse_iso_ts(cover_after)
        parse_iso_ts(cover_before)
    except ValueError:
        return False

    for entry in buckets:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            return False
        ts, count = entry
        if not isinstance(ts, str):
            return False
        try:
            parse_iso_ts(ts)
            int(count)
        except (ValueError, TypeError):
            return False
    return True


def _save_cache(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)


def _buckets_to_list(buckets: Sequence[Dict[str, str]]) -> List[Tuple[str, int]]:
    out: Dict[str, int] = {}
    for bucket in buckets:
        ts = normalize_created(bucket.get("created_utc"))
        if not ts:
            continue
        count = bucket.get("count", 0)
        try:
            count_int = int(count)
        except Exception:
            continue
        if ts in out and out[ts] != count_int:
            print(
                f"WARNING: Duplicate bucket for timestamp {ts} has mismatched counts: "
                f"{out[ts]} vs {count_int}. Keeping first value.",
                flush=True
            )
        # Deduplicate - keep first occurrence
        if ts not in out:
            out[ts] = count_int
    return sorted(out.items(), key=lambda x: parse_iso_ts(x[0]))


def _merge_buckets(existing: Sequence[Tuple[str, int]], additions: Sequence[Tuple[str, int]]) -> List[Tuple[str, int]]:
    combined: Dict[str, int] = {ts: cnt for ts, cnt in existing}
    for ts, cnt in additions:
        if ts in combined:
            # Duplicate timestamp - verify counts match
            if combined[ts] != cnt:
                print(
                    f"WARNING: Merging cache with new data - duplicate bucket for timestamp {ts} "
                    f"has mismatched counts: {combined[ts]} vs {cnt}. Keeping cached value.",
                    flush=True
                )
            # Keep existing (cached) value, discard duplicate
        else:
            combined[ts] = cnt
    return sorted(combined.items(), key=lambda x: parse_iso_ts(x[0]))


def ensure_histogram(
    api: ArcticShiftAPI,
    cache_dir: Path,
    *,
    frequency: str,
    kind: str,
    subreddit: str,
    after: Optional[str],
    before: Optional[str],
) -> List[Tuple[str, int]]:
    cache_path = _cache_path(cache_dir, subreddit, kind, frequency)
    cached = _load_cache(cache_path)

    if cached is None or not _validate_cache(cached):
        if cached:
            print(f"Invalid cache found for {subreddit}.{kind}.agg.{frequency}.json, refetching", flush=True)
        buckets = api.aggregate(kind, subreddit=subreddit, after=after, before=before, frequency=frequency)
        bucket_list = _buckets_to_list(buckets)
        payload = {
            "frequency": frequency,
            "coverage": {"after": after, "before": before},
            "buckets": bucket_list,
        }
        _save_cache(cache_path, payload)
        return bucket_list

    coverage = cached.get("coverage", {})
    cover_after = coverage.get("after")
    cover_before = coverage.get("before")
    bucket_list = cached.get("buckets", [])

    additions: List[Tuple[str, int]] = []

    if after and parse_iso_ts(after) < parse_iso_ts(cover_after):
        print(f"  Fetching additional data before cached range (before {cover_after})", flush=True)
        front_before = cover_after
        buckets = api.aggregate(kind, subreddit=subreddit, after=after, before=front_before, frequency=frequency)
        additions.extend(_buckets_to_list(buckets))
        cover_after = after

    if before and parse_iso_ts(before) > parse_iso_ts(cover_before):
        print(f"  Fetching additional data after cached range (after {cover_before})", flush=True)
        tail_after = cover_before
        buckets = api.aggregate(kind, subreddit=subreddit, after=tail_after, before=before, frequency=frequency)
        additions.extend(_buckets_to_list(buckets))
        cover_before = before

    if additions:
        bucket_list = _merge_buckets(bucket_list, additions)
        cached["buckets"] = bucket_list
        cached["coverage"] = {"after": cover_after, "before": cover_before}
        _save_cache(cache_path, cached)
        print(f"  Updated cache with {len(additions)} new buckets", flush=True)

    return bucket_list




