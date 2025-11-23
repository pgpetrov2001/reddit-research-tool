from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class Chunk:
    id: str
    doc_id: str
    title: str
    source: str
    section: Optional[str]
    text: str
    updated_at: Optional[str]
    author: Optional[str] = None


@dataclass
class Candidate:
    chunk: Chunk
    score: float
    # Optional per-method scores can be added later if needed
    # scores: Dict[str, float] | None = None


