from typing import Any, Dict, Optional, Tuple


def validate_bucket(bucket: Dict[str, Any]) -> Optional[Tuple[str, int]]:
    if not isinstance(bucket, dict):
        return None
    ts = bucket.get("created_utc")
    if ts is None:
        return None
    count = bucket.get("count", 0)
    try:
        count_int = int(count)
    except (TypeError, ValueError):
        return None
    return ts, count_int