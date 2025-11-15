from __future__ import annotations

import datetime as dt
from dateutil.relativedelta import relativedelta
from typing import Optional


ISO_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def _to_datetime(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(dt.timezone.utc)


def parse_iso_ts(value: str) -> dt.datetime:
    return _to_datetime(value)


def value_to_datetime(value: Optional[str]) -> Optional[dt.datetime]:
    if value is None:
        return None
    return _to_datetime(value)


def format_iso(ts: dt.datetime) -> str:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=dt.timezone.utc)
    return ts.astimezone(dt.timezone.utc).strftime(ISO_FORMAT)


def duration_from_string(s: str):
    s = s.lower()
    if s == "day":
        return dt.timedelta(days=1)
    elif s == "week":
        return dt.timedelta(weeks=1)
    elif s == "month":
        return relativedelta(months=1)
    elif s == "year":
        return relativedelta(years=1)
    else:
        raise ValueError(f"Unknown duration: {s}")


def normalize_created(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        ts = dt.datetime.fromtimestamp(float(value), tz=dt.timezone.utc)
        return format_iso(ts)
    if isinstance(value, str):
        if value.isdigit():
            ts = dt.datetime.fromtimestamp(float(value), tz=dt.timezone.utc)
            return format_iso(ts)
        return value
    return None


def value_to_token(value: Optional[str], default: str) -> str:
    token = value if value else default
    return token.replace(":", "~")


def _rehydrate_colonless(token: str) -> str:
    if "T" not in token or token.endswith(":") or token.count(":") > 0:
        return token
    if token.endswith("Z"):
        core = token[:-1]
        date, time_part = core.split("T", 1)
        pieces = time_part.split("-")
        if len(pieces) >= 3:
            hh, mm = pieces[0], pieces[1]
            rest = "-".join(pieces[2:])
            rebuilt = f"{date}T{hh}:{mm}:{rest}Z"
            return rebuilt
    return token


def token_to_value(token: str, default: Optional[str] = None) -> Optional[str]:
    if token == default:
        return None
    restored = token.replace("~", ":")
    if ":" not in restored:
        restored = _rehydrate_colonless(restored)
    return restored

