from __future__ import annotations

from collections import defaultdict
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .tokens import parse_iso_ts


def _to_datetime(value: Optional[str]) -> Optional[datetime]:
    if value is None:
        return None
    return parse_iso_ts(value)


@dataclass(frozen=True)
class CarryoverSegment:
    """Represents the slice of an old worker file that must be replayed into a new worker."""

    worker_id: int
    source_path: Path
    start: Optional[str]
    end: Optional[str]
    start_dt: Optional[datetime]
    end_dt: Optional[datetime]

    @classmethod
    def from_bounds(
        cls,
        *,
        worker_id: int,
        source_path: Path,
        start: Optional[str],
        end: Optional[str],
    ) -> "CarryoverSegment":
        return cls(
            worker_id=worker_id,
            source_path=source_path,
            start=start,
            end=end,
            start_dt=_to_datetime(start),
            end_dt=_to_datetime(end),
        )


class CarryoverManager:
    """Coordinates access to .old worker files shared across new workers."""

    def __init__(self) -> None:
        self._worker_segments: Dict[int, List[CarryoverSegment]] = defaultdict(list)
        self._file_locks: Dict[Path, threading.Lock] = {}
        self._file_counts: Dict[Path, int] = defaultdict(int)
        self._lock = threading.Lock()

    def add_segment(self, segment: CarryoverSegment) -> None:
        with self._lock:
            self._worker_segments[segment.worker_id].append(segment)
            self._file_counts[segment.source_path] += 1
            self._file_locks.setdefault(segment.source_path, threading.Lock())

    def segments_for_worker(self, worker_id: int) -> List[CarryoverSegment]:
        with self._lock:
            segments = list(self._worker_segments.get(worker_id, ()))
        segments.sort(key=lambda seg: (seg.start_dt or datetime.min, seg.end_dt or datetime.max))
        return segments

    def wait_for_turn(self, segment: CarryoverSegment) -> None:
        with self._lock:
            file_lock = self._file_locks.setdefault(segment.source_path, threading.Lock())
        file_lock.acquire()

    def mark_completed(self, segment: CarryoverSegment) -> None:
        file_lock = self._file_locks.get(segment.source_path)
        if file_lock:
            try:
                file_lock.release()
            except RuntimeError:
                pass

        delete_file = False
        with self._lock:
            if segment.source_path in self._file_counts:
                self._file_counts[segment.source_path] -= 1
                if self._file_counts[segment.source_path] <= 0:
                    delete_file = True
                    self._file_counts.pop(segment.source_path, None)
                    self._file_locks.pop(segment.source_path, None)
        if delete_file:
            self._delete_file(segment.source_path)

    @staticmethod
    def _delete_file(path: Path) -> None:
        try:
            path.unlink()
        except OSError:
            pass




