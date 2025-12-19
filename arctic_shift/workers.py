from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

from .api import ArcticShiftAPI
from .carryover import CarryoverManager, CarryoverSegment
from .fetcher import FetchResult, IntervalFetcher
from .tokens import parse_iso_ts, token_to_value, value_to_token


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


def prepare_worker_files(
    kind: str,
    base_dir: Path,
    plans: Sequence[WorkerPlan],
    existing_files: Sequence[Tuple[Path, Optional[str], Optional[str], Optional[int]]],
    overall_after: Optional[str],
    overall_before: Optional[str],
    subreddit: Optional[str] = None,
) -> CarryoverManager:
    """
    Prepare carryover metadata by renaming old worker files to *.jsonl.old and
    recording which slices must be replayed for the new worker plans.
    """
    manager = CarryoverManager()
    if not plans:
        return manager

    worker_dir = base_dir / f"{kind}_workers"
    worker_dir.mkdir(parents=True, exist_ok=True)

    # Rename existing worker files so they become .old carryover sources.
    renamed_sources: List[Tuple[Path, Optional[str], Optional[str]]] = []
    for path, start, end, _ in existing_files:
        renamed = _ensure_old_suffix(path)
        renamed_sources.append((renamed, start, end))

    if not renamed_sources and subreddit:
        final_file = base_dir / f"{subreddit}.{kind}.jsonl"
        if final_file.exists():
            renamed_sources.append(
                (_ensure_old_suffix(final_file, copy_source=True), overall_after, overall_before)
            )

    if not renamed_sources:
        return manager

    for worker_id, plan in enumerate(plans, start=1):
        plan_start, plan_end = plan.interval
        for source_path, source_start, source_end in renamed_sources:
            overlap_start = _later_start(plan_start, source_start)
            overlap_end = _earlier_end(plan_end, source_end)
            if _has_overlap(overlap_start, overlap_end):
                segment = CarryoverSegment.from_bounds(
                    worker_id=worker_id,
                    source_path=source_path,
                    start=overlap_start,
                    end=overlap_end,
                )
                manager.add_segment(segment)

    if any(manager.segments_for_worker(idx + 1) for idx in range(len(plans))):
        sources = {path.name for path, *_ in renamed_sources}
        print(f"[carryover] Prepared segments from {len(sources)} old file(s): {', '.join(sorted(sources))}")
    else:
        print("[carryover] No overlapping data found in old worker files.")

    return manager


def _ensure_old_suffix(path: Path, *, copy_source: bool = False) -> Path:
    if path.suffix == ".old":
        return path
    old_path = path.with_suffix(path.suffix + ".old")
    if old_path.exists():
        return old_path
    try:
        if copy_source:
            shutil.copy2(path, old_path)
        else:
            path.rename(old_path)
    except FileNotFoundError:
        pass
    return old_path


def _later_start(a: Optional[str], b: Optional[str]) -> Optional[str]:
    if a is None:
        return b
    if b is None:
        return a
    return a if parse_iso_ts(a) >= parse_iso_ts(b) else b


def _earlier_end(a: Optional[str], b: Optional[str]) -> Optional[str]:
    if a is None:
        return b
    if b is None:
        return a
    return a if parse_iso_ts(a) <= parse_iso_ts(b) else b


def _has_overlap(start: Optional[str], end: Optional[str]) -> bool:
    if start is None or end is None:
        return True
    return parse_iso_ts(start) < parse_iso_ts(end)


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
    initial_count_cb: Optional[Callable[[int, int], None]] = None,
    carryover_manager: Optional[CarryoverManager] = None,
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
                initial_count_cb=initial_count_cb,
                carryover_segments=carryover_manager.segments_for_worker(idx) if carryover_manager else None,
                carryover_manager=carryover_manager,
            )
            futures.append(executor.submit(fetcher.run, plan.interval, plan.expected))
        for fut in as_completed(futures):
            results.append(fut.result())
    return results

