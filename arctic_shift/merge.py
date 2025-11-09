from __future__ import annotations

import heapq
import json
import shutil
import tempfile
from pathlib import Path
from typing import Iterator, List, Set, Tuple

def merge_sorted_jsonl(paths: List[Path], dest: Path) -> Tuple[int, int, int]:
    """
    Merge sorted JSONL files, removing duplicates.
    
    Returns:
        Tuple of (existing_count, new_count, duplicate_count)
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    streams = []
    temp_file = None
    
    existing_count = 0
    seen_ids: Set[str] = set()
    duplicate_count = 0
    written_count = 0

    try:
        # If dest exists, create a temporary copy to avoid loading into memory
        if dest.exists():
            # Count existing rows
            with dest.open("r", encoding="utf-8") as existing_file:
                for existing_count, _ in enumerate(existing_file, 1):
                    pass
            temp_file = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False, suffix=".jsonl")
            with dest.open("r", encoding="utf-8") as existing_file:
                shutil.copyfileobj(existing_file, temp_file)
            temp_file.close()  # Close for writing, will reopen for reading
            temp_file = Path(temp_file.name)

        # Open all input files (including temp file if it exists)
        all_paths = [temp_file] + paths if temp_file else paths
        for path in all_paths:
            streams.append(path.open("r", encoding="utf-8"))

        iterators = [iter_stream(s) for s in streams]

        with dest.open("w", encoding="utf-8") as out:
            for item in heapq.merge(*iterators, key=lambda x: x["created_utc"]):
                # Check for duplicates by ID
                item_id = item.get("id")
                if item_id:
                    if item_id in seen_ids:
                        duplicate_count += 1
                        if duplicate_count <= 10:  # Warn about first 10 duplicates
                            print(f"[WARNING] Duplicate ID found: {item_id} (created_utc: {item.get('created_utc')})")
                        continue
                    seen_ids.add(item_id)
                
                out.write(json.dumps(item, ensure_ascii=False) + "\n")
                written_count += 1
        
        if duplicate_count > 10:
            print(f"[WARNING] ... and {duplicate_count - 10} more duplicates (total: {duplicate_count})")
        elif duplicate_count > 0:
            print(f"[WARNING] Total duplicates removed: {duplicate_count}")
        
        new_count = written_count - existing_count
        return existing_count, new_count, duplicate_count
    finally:
        for s in streams:
            s.close()
        # Clean up temporary file
        if temp_file and temp_file.exists():
            temp_file.unlink()


def iter_stream(stream) -> Iterator[dict]:
    for line in stream:
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue

