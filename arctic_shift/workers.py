from __future__ import annotations

import datetime as dt
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

from .api import ArcticShiftAPI
from .fetcher import FetchResult, IntervalFetcher
from .tokens import normalize_created, parse_iso_ts, token_to_value, value_to_token


@dataclass
class WorkerPlan:
    interval: Tuple[Optional[str], Optional[str]]
    expected: Optional[int]


def worker_path_for(kind: str, worker_dir: Path, worker_id: int, interval: Tuple[Optional[str], Optional[str]], expected: Optional[int]) -> Path:
    start = value_to_token(interval[0], "min")
    end = value_to_token(interval[1], "max")
    suffix = f"{start}__{end}"
    if expected is not None:
        suffix += f"__{expected}"
    slug = f"{kind}_worker{worker_id:02d}"
    return worker_dir / f"{slug}__{suffix}.jsonl"


def discover_worker_files(worker_dir: Path) -> List[Tuple[Path, Optional[str], Optional[str], Optional[int]]]:
    files = sorted(worker_dir.glob("*_worker*__*.jsonl"))
    info: List[Tuple[Path, Optional[str], Optional[str], Optional[int]]] = []
    for f in files:
        parts = f.stem.split("__")
        if len(parts) < 3:
            continue
        start_token = parts[1]
        end_token = parts[2]
        start = token_to_value(start_token, "min")
        end = token_to_value(end_token, "max")
        expected = None
        if len(parts) > 3:
            try:
                expected = int(parts[3])
            except ValueError:
                expected = None
        info.append((f, start, end, expected))
    return info


def worker_files_range_same(
    worker_files: List[Tuple[Path, Optional[str], Optional[str], Optional[int]]],
    after: Optional[str],
    before: Optional[str],
) -> bool:
    if not worker_files:
        return False

    def sort_key(entry: Tuple[Path, Optional[str], Optional[str], Optional[int]]):
        start = entry[1]
        if not start:
            return parse_iso_ts("1970-01-01T00:00:00Z")
        try:
            return parse_iso_ts(start)
        except ValueError:
            return parse_iso_ts("1970-01-01T00:00:00Z")

    worker_files.sort(key=sort_key)
    first_after = worker_files[0][1]
    last_before = worker_files[-1][2]
    return first_after == after and last_before == before


def _locate_partition(ts: dt.datetime, boundaries: List[Tuple[Optional[dt.datetime], Optional[dt.datetime]]]) -> Optional[int]:
    """Find the partition index for a timestamp, or None if outside all partitions."""
    for idx, (start_dt, end_dt) in enumerate(boundaries):
        if (start_dt is None or ts >= start_dt) and (end_dt is None or ts < end_dt):
            return idx
    
    # Assign to first/last partition if unbounded
    if boundaries:
        if boundaries[0][0] is None:  # First has unbounded start
            return 0
        if boundaries[-1][1] is None:  # Last has unbounded end
            return len(boundaries) - 1
    return None


def _read_and_classify_rows(
    existing_files: Sequence[Path],
    boundaries: List[Tuple[Optional[dt.datetime], Optional[dt.datetime]]],
    num_partitions: int,
) -> List[List[Tuple[str, dt.datetime]]]:
    """
    Read JSON objects from old worker files and distribute them into the new partitions.
    
    Returns a list where each element is a list of tuples: (json_line, timestamp).
    json_line is the full JSON object as a string, timestamp is parsed for sorting.

    Each list consists of all the elements from the old worker file that belong to the same partition.
    """
    partition_data: List[List[Tuple[str, dt.datetime]]] = [[] for _ in range(num_partitions)]
    
    for old_path in existing_files:
        if not old_path.exists():
            print(f"[WARNING] Worker file {old_path} which was detected a moment ago no longer exists")
            continue
        
        try:
            with old_path.open("r", encoding="utf-8") as reader:
                for line_number, json_line in enumerate(reader, start=1):
                    json_line = json_line.strip()
                    if not json_line:
                        continue
                    
                    try:
                        # Parse JSON to extract created_utc timestamp for classification
                        json_obj = json.loads(json_line)
                        created_iso = normalize_created(json_obj.get("created_utc"))
                        if not created_iso:
                            raise ValueError(f"Missing or invalid created_utc timestamp")
                        timestamp = parse_iso_ts(created_iso)
                    except (json.JSONDecodeError, ValueError) as e:
                        print(f"[WARNING] Invalid JSON line in existing worker file {old_path} on line {line_number}. Error: {e}. JSON line: {json_line}")
                        continue
                    
                    # Classify this JSON object into the appropriate partition
                    plan_idx = _locate_partition(timestamp, boundaries)
                    if plan_idx is not None and plan_idx < num_partitions:
                        # Store the original JSON line with its timestamp for sorting
                        partition_data[plan_idx].append((json_line, timestamp))
        except OSError as e:
            print(f"[WARNING] Failed to read worker file {old_path}: {e}")
    
    return partition_data


def _delete_existing_worker_files(existing_files: Sequence[Path]) -> List[Path]:
    deleted = []
    for path in existing_files:
        try:
            path.unlink()
            deleted.append(path)
        except OSError as e:
            print(f"[WARNING] Failed to delete worker file {path}: {e}")
    return deleted


def _write_worker_files(target_paths: List[Path], partition_data: List[List[Tuple[str, dt.datetime]]]) -> None:
    """
    Sort JSON objects by timestamp and write them to their corresponding worker files.
    
    Each tuple contains (json_line, timestamp) where json_line is the full JSON object.
    """
    for path, json_objects in zip(target_paths, partition_data):
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        
        # Sort by timestamp to maintain chronological order
        json_objects.sort(key=lambda x: x[1])

        with tmp.open("w", encoding="utf-8") as handle:
            for json_line, _ in json_objects:
                handle.write(json_line + "\n")
            handle.flush()
        
        if path.exists():
            path.unlink()
        tmp.replace(path)


def prepare_worker_files(
    kind: str,
    base_dir: Path,
    plans: Sequence[WorkerPlan],
    existing_files: Sequence[Tuple[Path, Optional[str], Optional[str], Optional[int]]],
    subreddit: Optional[str] = None,
) -> None:
    """
    Redistribute data from old worker files into new partition files.
    
    If no worker files exist but subreddit is provided, attempts to
    read from the final output file ({subreddit}.{kind}.jsonl) and redistribute from there.
    """
    if not plans:
        return
    
    worker_dir = base_dir / f"{kind}_workers"
    target_paths = [worker_path_for(kind, worker_dir, idx + 1, plan.interval, plan.expected) for idx, plan in enumerate(plans)]
    
    # Collect file paths to redistribute
    file_paths: List[Path] = []
    
    # Extract paths from existing worker files
    if existing_files:
        file_paths = [path for path, *_ in existing_files]
    # If no worker files, check for final output file
    elif subreddit:
        final_file = base_dir / f"{subreddit}.{kind}.jsonl"
        if final_file.exists():
            file_paths = [final_file]
        else:
            # No files to redistribute, create empty worker files
            for path in target_paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch(exist_ok=True)
            return
    else:
        # No files to redistribute, create empty worker files
        for path in target_paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch(exist_ok=True)
        return
    
    boundaries = [
        (parse_iso_ts(start) if start else None, parse_iso_ts(end) if end else None)
        for start, end in (plan.interval for plan in plans)
    ]
    print(f"Reading from {len(file_paths)} old files: {', '.join(path.name for path in file_paths)}")
    partition_data = _read_and_classify_rows(file_paths, boundaries, len(plans))
    nl = '\n'
    print(f"Classified old rows for the new files: {nl.join(f'{path.name}: {len(data)} old rows' for path, data in zip(target_paths, partition_data))}")
    _write_worker_files(target_paths, partition_data)
    print(f"Wrote the {len(target_paths)} new worker files using the old data.")
    deleted = _delete_existing_worker_files(set(file_paths) - set(target_paths)) # deletes old worker files
    print(f"Deleted {len(deleted)} existing worker files: {', '.join(str(path) for path in deleted)}")


def run_workers(
    *,
    api_factory: Callable[[], ArcticShiftAPI],
    kind: str,
    subreddit: str,
    fields: Optional[Sequence[str]],
    out_dir: Path,
    plans: Sequence[WorkerPlan],
    progress_cb: Callable[[int, int], None],
    completion_cb: Optional[Callable[[int], None]] = None,
) -> List[FetchResult]:
    results: List[FetchResult] = []
    with ThreadPoolExecutor(max_workers=len(plans) or 1) as executor:
        futures = []
        for idx, plan in enumerate(plans, start=1):
            api = api_factory()
            fetcher = IntervalFetcher(
                api=api,
                kind=kind,
                subreddit=subreddit,
                fields=fields,
                out_dir=out_dir,
                worker_id=idx,
                progress_cb=progress_cb,
                completion_cb=completion_cb,
            )
            futures.append(executor.submit(fetcher.run, plan.interval, plan.expected))
        for fut in as_completed(futures):
            results.append(fut.result())
    return results

