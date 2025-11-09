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
    ) -> None:
        self.api = api
        self.kind = kind
        self.subreddit = subreddit
        self.fields = list(fields) if fields else None
        self.out_dir = out_dir
        self.worker_id = worker_id
        self.progress_cb = progress_cb
        self.completion_cb = completion_cb
        self.lock = threading.Lock()

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

        written = self._resume_count(path)
        total_written = written
        with path.open("a", encoding="utf-8") as handle:
            total_written += self._fetch_interval(after, before, handle, total_written)
        
        # Signal completion
        if self.completion_cb:
            self.completion_cb(self.worker_id)
        
        return FetchResult(count=total_written, file_path=path)

    def _resume_count(self, path: Path) -> int:
        if not path.exists():
            return 0
        count = 0
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
        return count

    def _fetch_interval(self, after: Optional[str], before: Optional[str], handle, already_written: int) -> int:
        total_new = 0
        current_after = after
        if already_written > 0:
            current_after = self._resume_after(handle.name)
        current_before = before
        while True:
            if current_after and current_before and current_after >= current_before:
                break
            rows = self._fetch_once(current_after, current_before)
            if not rows:
                break
            rows.sort(key=lambda r: self._created_ts(r))
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            total_new += len(rows)
            self.progress_cb(self.worker_id, len(rows))
            current_after = normalize_created(rows[-1].get("created_utc"))
            if current_after is None:
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

    def _resume_after(self, path_name: str) -> Optional[str]:
        last_line = None
        with open(path_name, "r", encoding="utf-8") as reader:
            for line in reader:
                line = line.strip()
                if line:
                    last_line = line
        if not last_line:
            return None
        try:
            row = json.loads(last_line)
        except json.JSONDecodeError:
            return None
        return normalize_created(row.get("created_utc"))

