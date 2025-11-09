from __future__ import annotations

import datetime as dt
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import requests

from .validation import validate_bucket

from .tokens import format_iso, parse_iso_ts


BASE_URL = "https://arctic-shift.photon-reddit.com"
DEFAULT_TIMEOUT = 60
AGGREGATION_TIMEOUT = 10
MAX_RETRIES = 5
BACKOFF_BASE = 2.0
AGGREGRATE_FALLBACK_ORDER = ["week", "month", "year"]
MAX_AGGREGATION_SPLIT_DEPTH = 6


class ArcticShiftError(RuntimeError):
    pass

class ArcticShiftSplitError(ArcticShiftError):
    pass

class ArcticShiftAPI:
    def __init__(self, base_url: str = BASE_URL, timeout: int = DEFAULT_TIMEOUT, session: Optional[requests.Session] = None):
        self.base = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()

    def _get_json_with_retries(self, path: str, params: Dict[str, Any], retries: int = MAX_RETRIES, backoff: float = BACKOFF_BASE) -> Dict[str, Any]:
        url = f"{self.base}{path}"
        delay = 1.0
        for attempt in range(1, retries + 1):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
                if isinstance(data, dict) and data.get("error"):
                    # Handle API-provided errors such as timeouts
                    raise ArcticShiftError(str(data.get("error")))
                return {"data": data, "headers": resp.headers}
            except (requests.RequestException, ArcticShiftError) as exc:
                if attempt == retries:
                    raise
                time.sleep(delay)
                delay *= backoff
        raise ArcticShiftError("Failed to fetch data after retries")

    @staticmethod
    def _normalize_fields(fields: Optional[Union[str, Iterable[str]]]) -> Optional[str]:
        if fields is None:
            return None
        if isinstance(fields, str):
            return fields
        return ",".join(fields)

    def search(self, kind: str, *, subreddit: str, after: Optional[str], before: Optional[str], limit: Union[int, str] = "auto", fields: Optional[Union[str, Iterable[str]]] = None, extra: Optional[Dict[str, Any]] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        path = "/api/posts/search" if kind == "posts" else "/api/comments/search"
        params: Dict[str, Any] = {
            "subreddit": subreddit.lstrip("r/") if subreddit else None,
            "after": after,
            "before": before,
            "limit": limit,
            "fields": self._normalize_fields(fields),
            "sort": "asc",
        }
        if extra:
            params.update(extra)
        cleaned = {k: v for k, v in params.items() if v is not None}
        result = self._get_json_with_retries(path, cleaned)
        return self._extract_items(result["data"]), result["headers"]

    def aggregate(self, kind: str, *, subreddit: str, after: Optional[str], before: Optional[str], frequency: str) -> List[Dict[str, Any]]:
        try:
            # Store original boundaries for filtering at the end
            original_after = after
            original_before = before
            
            # Align interval endpoints to frequency boundaries
            if after and before:
                aligned_after, aligned_before = self._align_interval_endpoints(after, before, frequency)
                print(f"Aligned interval from [{after}, {before}) to [{aligned_after}, {aligned_before})", flush=True)
                after = aligned_after
                before = aligned_before
            
            # Fetch data with aligned boundaries
            buckets = self._aggregate_with_split(
                kind,
                subreddit=subreddit,
                after=after,
                before=before,
                frequency=frequency,
                depth=0,
            )
            
            # Filter to only include buckets within original interval
            if original_after or original_before:
                buckets = self._filter_buckets_to_interval(buckets, original_after, original_before)
            
            return buckets
        except Exception as exc:  # noqa: BLE001
            raise ArcticShiftError(f"aggregate failed: {exc}") from exc

    @staticmethod
    def _extract_items(data: Any) -> List[Dict[str, Any]]:
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("data", "results", "items"):
                val = data.get(key)
                if isinstance(val, list):
                    return val
        return []

    def _aggregate_with_split(
        self,
        kind: str,
        *,
        subreddit: str,
        after: Optional[str],
        before: Optional[str],
        frequency: str,
        depth: int,
    ) -> List[Dict[str, Any]]:
        try:
            print(f"Aggregating {subreddit}.{kind}.agg.{frequency}.json in interval {after} to {before}; depth {depth}", flush=True)
            return self._aggregate_request(
                kind,
                subreddit=subreddit,
                after=after,
                before=before,
                frequency=frequency,
            )
        except requests.HTTPError as exc:
            # Only split on 422 errors
            if isinstance(exc, requests.HTTPError):
                if exc.response is None or exc.response.status_code != 422:
                    raise
            
            if depth >= MAX_AGGREGATION_SPLIT_DEPTH:
                raise ArcticShiftSplitError(f"Maximum split depth of {MAX_AGGREGATION_SPLIT_DEPTH} reached for {subreddit}.{kind}.agg.{frequency}.json in interval {after} to {before}")
            
            if not after or not before:
                raise ArcticShiftSplitError(f"No after or before provided for {subreddit}.{kind}.agg.{frequency}.json")
            
            splits = self._split_range(after, before, frequency)
            if not splits:
                raise ArcticShiftSplitError(f"No splits found for {subreddit}.{kind}.agg.{frequency}.json in interval {after} to {before}")
            
            combined: List[Dict[str, Any]] = []
            for sub_after, sub_before in splits:
                combined.extend(
                    self._aggregate_with_split(
                        kind,
                        subreddit=subreddit,
                        after=sub_after,
                        before=sub_before,
                        frequency=frequency,
                        depth=depth + 1,
                    )
                )
            return self._collapse_buckets(combined)

    def _aggregate_request(
        self,
        kind: str,
        *,
        subreddit: str,
        after: Optional[str],
        before: Optional[str],
        frequency: str,
    ) -> List[Dict[str, Any]]:
        path = "/api/posts/search/aggregate" if kind == "posts" else "/api/comments/search/aggregate"
        params: Dict[str, Any] = {
            "subreddit": subreddit.lstrip("r/") if subreddit else None,
            "after": after,
            "before": before,
            "aggregate": "created_utc",
            "frequency": frequency,
        }
        cleaned = {k: v for k, v in params.items() if v is not None}
        
        # Single attempt with short timeout for aggregation requests
        url = f"{self.base}{path}"
        resp = self.session.get(url, params=cleaned, timeout=AGGREGATION_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        
        if isinstance(data, dict) and data.get("error"):
            raise ArcticShiftError(str(data.get("error")))

        buckets = self._extract_items(data)
        if not buckets:
            raise ArcticShiftError(f"API returned no buckets for {subreddit}.{kind}.agg.{frequency}.json in interval {after} to {before}")
        return buckets

    @staticmethod
    def _align_interval_endpoints(after: str, before: str, frequency: str) -> Tuple[str, str]:
        """
        Align interval endpoints to frequency boundaries, expanding to ensure full coverage.
        
        Args:
            after: Start of interval (ISO format)
            before: End of interval (ISO format)
            frequency: 'week', 'month', or 'year'
        
        Returns:
            Tuple of (aligned_after, aligned_before) where:
            - aligned_after is floored (rounds down to include the period containing 'after')
            - aligned_before is ceiled (rounds up to include the period containing 'before')
        """
        start = parse_iso_ts(after)
        end = parse_iso_ts(before)
        
        # Floor the start (expand backwards to frequency boundary)
        aligned_start = ArcticShiftAPI._align_to_frequency(start, frequency, "floor")
        
        # Ceil the end (expand forwards to frequency boundary)
        aligned_end = ArcticShiftAPI._align_to_frequency(end, frequency, "ceil")
        
        return format_iso(aligned_start), format_iso(aligned_end)
    
    @staticmethod
    def _filter_buckets_to_interval(
        buckets: List[Dict[str, Any]],
        after: Optional[str],
        before: Optional[str]
    ) -> List[Dict[str, Any]]:
        """
        Filter buckets to only include those within the specified interval [after, before).
        
        Args:
            buckets: List of bucket dictionaries with 'created_utc' timestamps
            after: Start of interval (inclusive), None means no lower bound
            before: End of interval (exclusive), None means no upper bound
        
        Returns:
            Filtered list of buckets
        """
        if not after and not before:
            return buckets
        
        filtered = []
        after_dt = parse_iso_ts(after) if after else None
        before_dt = parse_iso_ts(before) if before else None
        
        for bucket in buckets:
            result = validate_bucket(bucket)
            if result is None:
                continue
            ts_str, _ = result
            bucket_dt = parse_iso_ts(ts_str)
            
            # Check if bucket is within interval [after, before)
            if after_dt and bucket_dt < after_dt:
                continue
            if before_dt and bucket_dt >= before_dt:
                continue
            
            filtered.append(bucket)
        
        return filtered
    
    @staticmethod
    def _align_to_frequency(dt_obj: dt.datetime, frequency: str, direction: str) -> dt.datetime:
        """
        Align datetime to frequency boundary.
        
        Args:
            dt_obj: datetime to align
            frequency: 'week', 'month', or 'year'
            direction: 'floor' (round down) or 'ceil' (round up)
        
        Returns:
            Aligned datetime at 00:00:00 UTC
        """
        # Normalize to midnight UTC
        dt_obj = dt_obj.replace(hour=0, minute=0, second=0, microsecond=0)
        
        if frequency == "week":
            # Align to Monday (weekday 0 in Python)
            days_since_monday = dt_obj.weekday()
            if direction == "floor":
                return dt_obj - dt.timedelta(days=days_since_monday)
            else:  # ceil
                if days_since_monday == 0:
                    return dt_obj
                return dt_obj + dt.timedelta(days=7 - days_since_monday)
        
        elif frequency == "month":
            # Align to 1st of month
            if direction == "floor":
                return dt_obj.replace(day=1)
            else:  # ceil
                if dt_obj.day == 1:
                    return dt_obj
                # Go to next month
                if dt_obj.month == 12:
                    return dt_obj.replace(year=dt_obj.year + 1, month=1, day=1)
                else:
                    return dt_obj.replace(month=dt_obj.month + 1, day=1)
        
        elif frequency == "year":
            # Align to January 1st
            if direction == "floor":
                return dt_obj.replace(month=1, day=1)
            else:  # ceil
                if dt_obj.month == 1 and dt_obj.day == 1:
                    return dt_obj
                return dt_obj.replace(year=dt_obj.year + 1, month=1, day=1)
        
        return dt_obj
    
    @staticmethod
    def _split_range(after: str, before: str, frequency: str) -> Optional[List[Tuple[str, str]]]:
        """
        Split time range in half, aligning split point to frequency boundaries.
        
        Returns overlapping ranges to ensure no data is missed due to rounding.
        The overlap will be deduplicated by _collapse_buckets.
        """
        start = parse_iso_ts(after)
        end = parse_iso_ts(before)
        
        if start >= end:
            return None
        
        # Find midpoint
        mid_timestamp = start.timestamp() + (end.timestamp() - start.timestamp()) / 2
        mid = dt.datetime.fromtimestamp(mid_timestamp, tz=dt.timezone.utc)
        mid_str = format_iso(mid)
        
        # Align the midpoint to create overlapping ranges
        # Use _align_interval_endpoints which expands the interval
        mid_aligned_after, mid_aligned_before = ArcticShiftAPI._align_interval_endpoints(
            mid_str, mid_str, frequency
        )
        
        mid_floor = parse_iso_ts(mid_aligned_after)
        mid_ceil = parse_iso_ts(mid_aligned_before)
        
        # If floor alignment is too close to start or end, can't split
        if mid_floor <= start or mid_floor >= end:
            return None
        
        # Ensure ceiling doesn't exceed end boundary
        if mid_ceil > end:
            mid_ceil = mid_floor
        
        # Create overlapping ranges:
        # First range: [after, mid_ceil) covers everything up to and including mid boundary
        # Second range: [mid_floor, before) covers everything from mid boundary onward
        # Any overlap will be deduplicated by _collapse_buckets
        return [
            (after, format_iso(mid_ceil)),
            (format_iso(mid_floor), before),
        ]

    @staticmethod
    def _collapse_buckets(buckets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Deduplicate buckets from overlapping ranges.
        
        When the same timestamp appears multiple times (due to overlapping ranges),
        keep only one entry. Verify that counts match, as they represent the same data.
        """
        merged: Dict[str, Dict[str, Any]] = {}
        for bucket in buckets:
            result = validate_bucket(bucket)
            if result is None:
                continue
            ts, count_int = result
            if ts not in merged:
                merged_bucket = dict(bucket)
                merged_bucket["count"] = count_int
                merged[ts] = merged_bucket
            else:
                # Duplicate timestamp - verify counts match
                existing_count = int(merged[ts].get("count", 0))
                if existing_count != count_int:
                    print(
                        f"WARNING: Duplicate bucket for timestamp {ts} has mismatched counts: "
                        f"{existing_count} vs {count_int}. Keeping first value.",
                        flush=True
                    )
                # Keep the first occurrence, discard duplicate
        return [
            merged[ts] for ts in sorted(merged.keys(), key=lambda item: parse_iso_ts(item))
        ]

