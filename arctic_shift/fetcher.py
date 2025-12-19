from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .api import ArcticShiftAPI
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
        self.lock = threading.Lock()
        self.total_written = 0
        self.last_timestamp = None

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

        self._try_resume(path)
        if self.initial_count_cb:
            self.initial_count_cb(self.worker_id, self.total_written)
        with path.open("a", encoding="utf-8") as handle:
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
        self.last_timestamp = normalize_created(row.get("created_utc"))

    def _fetch_interval(self, after: str, before: str, handle) -> int:
        total_new = 0
        if self.last_timestamp is None:
            self.last_timestamp = after
        while True:
            if self.last_timestamp >= before:
                break
            rows = self._fetch_once(self.last_timestamp, before)
            if not rows:
                break
            rows.sort(key=lambda r: self._created_ts(r))
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            total_new += len(rows)
            self.progress_cb(self.worker_id, len(rows))
            self.last_timestamp = normalize_created(rows[-1].get("created_utc"))
            if self.last_timestamp is None:
                raise RuntimeError(f"Encountered empty 'created_utc' timestamp on fetched {self.kind} while fetching interval [{after}, {before}) in worker {self.worker_id}")
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

