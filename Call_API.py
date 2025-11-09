#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from arctic_shift import ArcticShiftAPI, ArcticShiftError
from arctic_shift.partition import compute_naive_partitions, compute_partitions
from arctic_shift.workers import (
    WorkerPlan,
    discover_worker_files,
    prepare_worker_files,
    worker_files_range_same,
    run_workers,
)
from arctic_shift.progress import ProgressTracker, get_resume_counts
from arctic_shift.merge import merge_sorted_jsonl
from arctic_shift.aggregation import ensure_histogram, FREQUENCY_ORDER


def parse_iso_date(text: str) -> str:
    try:
        if "T" in text:
            return dt.datetime.fromisoformat(text).replace(tzinfo=dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        return dt.date.fromisoformat(text).strftime("%Y-%m-%dT00:00:00Z")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid date/time: {text}") from exc


def compute_histogram(
    api: ArcticShiftAPI,
    cache_dir: Path,
    kind: str,
    subreddit: str,
    after: Optional[str],
    before: Optional[str],
    workers: int,
) -> Tuple[List[Tuple[str, int]], Optional[str]]:
    if workers <= 1:
        return [], None
    print(f"[{kind}] Computing histogram for partitioning ({workers} workers)...", flush=True)
    for freq in FREQUENCY_ORDER:
        try:
            print(f"[{kind}] Trying frequency: {freq}", flush=True)
            buckets = ensure_histogram(
                api,
                cache_dir,
                frequency=freq,
                kind=kind,
                subreddit=subreddit,
                after=after,
                before=before,
            )
            total = sum(cnt for _, cnt in buckets)
            print(f"[{kind}] Histogram computed successfully with frequency '{freq}': {len(buckets)} buckets, {total} total items", flush=True)
            return buckets, freq
        except Exception as exc:  # noqa: BLE001
            print(f"[{kind}] WARNING: Failed to compute histogram for frequency {freq}: {exc}", flush=True)
            continue
    raise ArcticShiftError("Failed to compute histogram for any frequency")


def planned_partitions(
    api: ArcticShiftAPI,
    cache_dir: Path,
    kind: str,
    subreddit: str,
    after: Optional[str],
    before: Optional[str],
    workers: int,
    force_histogram: bool = False,
) -> List[WorkerPlan]:
    try:
        histogram, frequency = compute_histogram(api, cache_dir, kind, subreddit, after, before, workers)
    except ArcticShiftError as exc:
        if force_histogram:
            print(f"[{kind}] ERROR: Failed to compute histogram: {exc}", flush=True)
            print(f"[{kind}] --force-histogram is enabled, cannot fall back to naive partitioning", flush=True)
            raise
        print(f"[{kind}] WARNING: Failed to compute histogram: {exc}", flush=True)
        print(f"[{kind}] Falling back to naive time-based partitioning", flush=True)
        parts, totals = compute_naive_partitions(after, before, workers)
    else:
        parts, totals = compute_partitions(
            after=after, before=before, workers=workers, histogram=histogram
        )
    
    total_expected = sum(t for t in totals if t is not None) if totals else 0
    print(f"[{kind}] Partitioned into {len(parts)} partitions, expected total: {total_expected}", flush=True)
    return [WorkerPlan(interval=part, expected=totals[idx]) for idx, part in enumerate(parts)]


def merge_worker_outputs(paths: List[Path], dest: Path, kind: str) -> Tuple[int, int, int]:
    """Merge worker outputs and return (existing_count, new_count, duplicate_count)."""
    print(f"[{kind}] Merging {len(paths)} worker files...", flush=True)
    return merge_sorted_jsonl(paths, dest)


def download_kind(kind: str, api_factory, args) -> None:
    print(f"\n{'='*60}", flush=True)
    print(f"[{kind}] Starting download for subreddit: {args.subreddit}", flush=True)
    if args.after:
        print(f"[{kind}] After: {args.after}", flush=True)
    if args.before:
        print(f"[{kind}] Before: {args.before}", flush=True)
    print(f"[{kind}] Workers: {args.workers}", flush=True)
    print(f"{'='*60}", flush=True)
    
    base_dir = Path(args.outdir)
    worker_dir = base_dir / f"{kind}_workers"
    worker_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{kind}] Discovering existing worker files...", flush=True)
    existing = discover_worker_files(worker_dir)
    print(f"[{kind}] Found {len(existing)} existing worker files", flush=True)
    
    if args.workers > 0 and len(existing) == args.workers and worker_files_range_same(existing, args.after, args.before):
        print(f"[{kind}] Existing worker files match configuration, reusing partitions", flush=True)
        plans = [WorkerPlan(interval=(info[1], info[2]), expected=info[3]) for info in existing]
    else:
        print(f"[{kind}] Computing new partitions...", flush=True)
        api = api_factory()
        cache_dir = base_dir
        plans = planned_partitions(api, cache_dir, kind, args.subreddit, args.after, args.before, args.workers, force_histogram=args.force_histogram)
        print(f"[{kind}] Preparing worker files (redistributing if needed)...", flush=True)
        prepare_worker_files(kind, base_dir, plans, existing, subreddit=args.subreddit)
        print(f"[{kind}] Worker files prepared", flush=True)

    # Print partitions before starting workers
    print(f"\n[{kind}] Partitions:", flush=True)
    total_estimate = sum(p.expected for p in plans if p.expected is not None)
    for idx, plan in enumerate(plans, start=1):
        after_str = plan.interval[0] or "∞"
        before_str = plan.interval[1] or "∞"
        expected_str = str(plan.expected) if plan.expected is not None else "?"
        print(f"  Worker {idx}: {after_str} to {before_str} (expected: {expected_str})", flush=True)
    if total_estimate:
        print(f"[{kind}] Total expected items: {total_estimate}", flush=True)
    print(f"[{kind}] Starting {len(plans)} worker(s)...", flush=True)
    
    # Initialize progress with resume counts if files exist
    initial_counts = get_resume_counts(kind, worker_dir, plans)
    progress = ProgressTracker(kind, plans, total=total_estimate or None, initial_counts=initial_counts)
    # Print initial progress
    progress.print_initial()
    try:
        results = run_workers(
            api_factory=api_factory,
            kind=kind,
            subreddit=args.subreddit,
            fields=args.post_fields if kind == "posts" else args.comment_fields,
            out_dir=worker_dir,
            plans=plans,
            progress_cb=progress.update,
            completion_cb=progress.mark_completed,
        )
    finally:
        progress.close()

    if not results:
        print(f"[{kind}] No {kind} fetched", flush=True)
        return

    total_fetched = sum(r.count for r in results)
    print(f"[{kind}] Fetching complete: {total_fetched} {kind} from {total_estimate} expected items with {len(results)} worker(s)", flush=True)

    final_path = base_dir / f"{args.subreddit}.{kind}.jsonl"
    existing_count, new_count, duplicate_count = merge_worker_outputs([res.file_path for res in results], final_path, kind)
    
    print(f"[{kind}] Merge complete → {final_path}", flush=True)
    print(f"[{kind}]   Existing items: {existing_count}", flush=True)
    print(f"[{kind}]   New items added: {new_count}", flush=True)
    if duplicate_count > 0:
        print(f"[{kind}]   Duplicates removed: {duplicate_count}", flush=True)
    print(f"[{kind}]   Total items: {existing_count + new_count}", flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download subreddit data from Arctic Shift API with worker partitioning.")
    parser.add_argument("-s", "--subreddit", required=True)
    parser.add_argument("--after", type=parse_iso_date, help="ISO8601 start, inclusive")
    parser.add_argument("--before", type=parse_iso_date, help="ISO8601 end, exclusive")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker splits")
    parser.add_argument("--what", choices=["posts", "comments", "both"], default="posts")
    parser.add_argument("--outdir", help="Output directory")
    parser.add_argument("--force-histogram", action="store_true", help="Force histogram-based partitioning, fail if histogram cannot be computed")
    parser.add_argument(
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
    parser.add_argument(
        "--comment_fields",
        nargs="*",
        default=[
            "id",
            "created_utc",
            "author",
            "subreddit",
            "score",
            "body",
            "link_id",
            "parent_id",
        ],
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    print(f"Starting download for subreddit: {args.subreddit}", flush=True)
    print(f"Output directory: {args.outdir}", flush=True)
    
    if args.what in ("posts", "both"):
        download_kind("posts", ArcticShiftAPI, args)
    if args.what in ("comments", "both"):
        download_kind("comments", ArcticShiftAPI, args)
    
    print("\nDownload complete!", flush=True)


if __name__ == "__main__":
        main()
