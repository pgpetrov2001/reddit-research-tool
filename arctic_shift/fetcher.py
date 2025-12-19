from __future__ import annotations

import json
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

from .api import ArcticShiftAPI
from .carryover import CarryoverManager, CarryoverSegment
from .tokens import normalize_created, parse_iso_ts, value_to_token


@dataclass
class FetchResult:
    count: int
    file_path: Path

class IntervalFetcher:
    def __init__(
        self,
        api: ArcticShiftAPI,
        kind: str,
        subreddit: str,
        fields: Optional[Sequence[str]],
        out_dir: Path,
        worker_id: int,
        progress_cb: Callable[[int, int], None],
        completion_cb: Optional[Callable[[int], None]] = None,
        initial_count_cb: Optional[Callable[[int, int], None]] = None,
        carryover_segments: Optional[Sequence[CarryoverSegment]] = None,
        carryover_manager: Optional[CarryoverManager] = None,
    ) -> None:
        self.api = api
        self.kind = kind
        self.subreddit = subreddit
        self.fields = list(fields) if fields else None
        self.out_dir = out_dir
        self.worker_id = worker_id
        self.progress_cb = progress_cb
        self.completion_cb = completion_cb
        self.initial_count_cb = initial_count_cb
        self.carryover_segments = list(carryover_segments or [])
        self.carryover_manager = carryover_manager
        self.lock = threading.Lock()
        self.total_written = 0
        self.last_timestamp = None
        self.last_timestamp_dt: Optional[datetime] = None
        self.plan_start: Optional[str] = None
        self.plan_end: Optional[str] = None
        self.plan_start_dt: Optional[datetime] = None
        self.plan_end_dt: Optional[datetime] = None
        self.pending_segments: Deque[CarryoverSegment] = deque()

    def run(self, interval: Tuple[Optional[str], Optional[str]], expected_total: Optional[int] = None) -> FetchResult:
        after, before = interval
        slug = f"{self.kind}_worker{self.worker_id:02d}"
        start = value_to_token(after, "min")
        end = value_to_token(before, "max")
        suffix = f"{start}__{end}"
        if expected_total is not None:
            suffix += f"__{expected_total}"
        path = self.out_dir / f"{slug}__{suffix}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)

        self.plan_start, self.plan_end = interval
        self.plan_start_dt = parse_iso_ts(after) if after else None
        self.plan_end_dt = parse_iso_ts(before) if before else None
        self.pending_segments = deque(sorted(self.carryover_segments, key=lambda seg: seg.start_dt or datetime.min))

        self._try_resume(path)
        with path.open("a", encoding="utf-8") as handle:
            self._flush_ready_segments(handle)
            if self.initial_count_cb:
                self.initial_count_cb(self.worker_id, self.total_written)
            self._fetch_interval(after, before, handle)
        
        # Signal completion
        if self.completion_cb:
            self.completion_cb(self.worker_id)
        
        return FetchResult(count=self.total_written, file_path=path)

    def _try_resume(self, path: Path):
        if not path.exists():
            return
        self.total_written = 0
        last_line = None
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.total_written += 1
                    last_line = line
        if not last_line:
            return
        try:
            row = json.loads(last_line)
        except json.JSONDecodeError:
            return
        self._set_last_timestamp(normalize_created(row.get("created_utc")))

    def _fetch_interval(self, after: str, before: str, handle) -> int:
        total_new = 0
        while True:
            cursor = self.last_timestamp or after
            if before and cursor and parse_iso_ts(cursor) >= parse_iso_ts(before):
                break
            rows = self._fetch_once(cursor, before)
            if not rows:
                break
            rows.sort(key=lambda r: self._created_ts(r))
            boundary_dt = self._next_boundary_dt()
            appended = 0
            hit_boundary = False
            for row in rows:
                created_iso = normalize_created(row.get("created_utc"))
                if not created_iso:
                    continue
                created_dt = parse_iso_ts(created_iso)
                if self.plan_end_dt and created_dt >= self.plan_end_dt:
                    hit_boundary = False
                    rows = []
                    break
                if boundary_dt and created_dt >= boundary_dt:
                    hit_boundary = True
                    break
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                appended += 1
                total_new += 1
                self.total_written += 1
                self._set_last_timestamp(created_iso)
            if appended:
                self.progress_cb(self.worker_id, appended)
            if hit_boundary:
                self._force_flush_next_segment(handle)
                continue
            if not appended:
                break
        return total_new

    def _fetch_once(self, after: Optional[str], before: Optional[str]) -> List[Dict[str, Any]]:
        rows, _ = self.api.search(
            self.kind,
            subreddit=self.subreddit,
            after=after,
            before=before,
            limit="auto",
            fields=self.fields,
        )
        return rows

    @staticmethod
    def _created_ts(row: Dict[str, Any]) -> float:
        created = row.get("created_utc")
        if isinstance(created, (int, float)):
            return float(created)
        if isinstance(created, str):
            if created.isdigit():
                return float(created)
            return parse_iso_ts(created).timestamp()
        raise ValueError("missing created_utc")

    def _set_last_timestamp(self, value: Optional[str]) -> None:
        self.last_timestamp = value
        self.last_timestamp_dt = parse_iso_ts(value) if value else None

    def _current_position_dt(self) -> Optional[datetime]:
        return self.last_timestamp_dt or self.plan_start_dt

    def _next_boundary_dt(self) -> Optional[datetime]:
        if self.pending_segments:
            return self.pending_segments[0].start_dt
        return self.plan_end_dt

    def _flush_ready_segments(self, handle) -> None:
        while self.pending_segments:
            next_segment = self.pending_segments[0]
            if next_segment.start_dt and self._current_position_dt() and self._current_position_dt() < next_segment.start_dt:
                break
            self.pending_segments.popleft()
            self._drain_segment(next_segment, handle)

    def _force_flush_next_segment(self, handle) -> None:
        if not self.pending_segments:
            return
        segment = self.pending_segments.popleft()
        self._drain_segment(segment, handle)
        self._flush_ready_segments(handle)

    def _drain_segment(self, segment: CarryoverSegment, handle) -> None:
        if not self.carryover_manager:
            return
        self.carryover_manager.wait_for_turn(segment)
        flushed = 0
        try:
            if not segment.source_path.exists():
                return
            with segment.source_path.open("r", encoding="utf-8") as reader:
                for raw_line in reader:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    created_iso = normalize_created(obj.get("created_utc"))
                    if not created_iso:
                        continue
                    created_dt = parse_iso_ts(created_iso)
                    if segment.start_dt and created_dt < segment.start_dt:
                        continue
                    if segment.end_dt and created_dt >= segment.end_dt:
                        break
                    if self.plan_start_dt and created_dt < self.plan_start_dt:
                        continue
                    if self.plan_end_dt and created_dt >= self.plan_end_dt:
                        break
                    handle.write(line + "\n")
                    flushed += 1
                    self.total_written += 1
                    self._set_last_timestamp(created_iso)
            if flushed:
                self.progress_cb(self.worker_id, flushed)
        finally:
            self.carryover_manager.mark_completed(segment)

