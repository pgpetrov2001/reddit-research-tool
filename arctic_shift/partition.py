from __future__ import annotations

import datetime as dt
from math import floor
from typing import List, Optional, Sequence, Tuple


ISO_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def parse_iso(timestamp: str) -> dt.datetime:
    return dt.datetime.fromisoformat(timestamp.replace("Z", "+00:00")).astimezone(dt.timezone.utc)


def format_iso(ts: dt.datetime) -> str:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=dt.timezone.utc)
    return ts.astimezone(dt.timezone.utc).strftime(ISO_FORMAT)


def frequency_to_timedelta(frequency: Optional[str]) -> Optional[dt.timedelta]:
    """Convert frequency string to timedelta."""
    if frequency is None:
        return None
    freq_map = {
        "hour": dt.timedelta(hours=1),
        "day": dt.timedelta(days=1),
        "week": dt.timedelta(weeks=1),
        "month": dt.timedelta(days=30),  # Approximate
        "year": dt.timedelta(days=365),  # Approximate
    }
    return freq_map.get(frequency.lower(), dt.timedelta(days=1))  # Default to 1 day if unknown


def interpolate_dates(date1: str, date2: str, progress: float) -> str:
    date1_dt = parse_iso(date1)
    date2_dt = parse_iso(date2)
    return format_iso(date1_dt + (date2_dt - date1_dt) * progress)


def clamp_date(date: str, after: Optional[str], before: Optional[str] = None) -> str:
    date_dt = parse_iso(date)
    after_dt = parse_iso(after) if after else None
    before_dt = parse_iso(before) if before else None
    if after_dt and date_dt < after_dt:
        return after
    if before_dt and date_dt >= before_dt:
        return before
    return date


def quantile_cutoffs(numbers: Sequence[int], workers: int) -> List[int]:
    if not numbers:
        return []
    target = sum(numbers) / workers
    cuts = []
    i = -1
    s = 0
    while i < len(numbers):
        while i < len(numbers) and s < target:
            i += 1
            if i >= len(numbers):
                break
            s += numbers[i]
        if i >= len(numbers):
            break
        ratio = (numbers[i] - (s - target)) / numbers[i]
        cuts.append(i + ratio)
        s = (1 - ratio) * numbers[i]

    return cuts[:(workers - 1)]


def partitions_from_cuts(after: str, before: str, buckets: List[Tuple[str, int]], quantile_cuts: List[int]) -> List[Tuple[str, str]]:
    bounds = []
    for cut in quantile_cuts:
        i = floor(cut)
        date = interpolate_dates(
            buckets[i][0],
            buckets[i + 1][0] if i + 1 < len(buckets) else before,
            cut - i,
        )
        date = clamp_date(date, after)
        bounds.append(date)
    
    bounds = [after] + bounds + [before]

    # Convert bounds list to partition tuples
    partitions: List[Tuple[Optional[str], Optional[str]]] = []
    for i in range(len(bounds) - 1):
        partitions.append((bounds[i], bounds[i + 1]))

    return partitions


def compute_partitions(
    *,
    after: str,
    before: str,
    workers: int,
    histogram: Sequence[Tuple[str, int]],
) -> Tuple[List[Tuple[Optional[str], Optional[str]]], List[Optional[int]]]:
    total = sum(count for _, count in histogram)
    if workers <= 1:
        return [(after, before)], [total]

    if not histogram:
        raise ValueError("Cannot partition empty histogram with multiple workers")

    cuts = quantile_cutoffs([count for _, count in histogram], workers)
    partitions = partitions_from_cuts(after, before, histogram, cuts)
    
    # quantile_cutoffs returns workers cuts, creating workers+1 partitions
    # Calculate expected totals per partition (approximate, since we split evenly)
    avg_per_partition = total / len(partitions)
    totals = [int(avg_per_partition) for _ in range(len(partitions))]
    # Adjust last partition to account for rounding
    totals[-1] = total - sum(totals[:-1])

    return partitions, totals

# TODO: fix the 1 more partition returned
def compute_naive_partitions(
    after: Optional[str],
    before: Optional[str],
    workers: int,
) -> Tuple[List[Tuple[Optional[str], Optional[str]]], List[Optional[int]]]:
    """
    Compute partitions by splitting the time range evenly into equal time intervals.
    Used as a fallback when histogram computation fails.
    Returns (partitions, totals) where partitions are (after, before) tuples and totals are None (unknown).
    """
    if workers <= 1:
        return [(after, before)], [None]
    
    if after is None or before is None:
        # Can't split unbounded ranges evenly, return single partition
        return [(after, before)], [None]
    
    after_dt = parse_iso(after)
    before_dt = parse_iso(before)
    total_duration = before_dt - after_dt
    
    if total_duration.total_seconds() <= 0:
        return [(after, before)], [None]
    
    # Split into equal time intervals
    interval_duration = total_duration / workers
    partitions: List[Tuple[Optional[str], Optional[str]]] = []
    
    for i in range(workers):
        part_after_dt = after_dt + (interval_duration * i)
        part_before_dt = after_dt + (interval_duration * (i + 1))
        
        # Last partition should end exactly at 'before'
        if i == workers - 1:
            part_before_dt = before_dt
        
        part_after = format_iso(part_after_dt)
        part_before = format_iso(part_before_dt)
        partitions.append((part_after, part_before))
    
    return partitions, [None] * workers

