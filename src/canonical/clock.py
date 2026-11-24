"""Strict UTC timestamp and canonical identity helpers."""
from datetime import datetime, timezone
import hashlib
import json
import math


class CanonicalError(RuntimeError):
    def __init__(self, code, detail=""):
        self.code = code
        super().__init__(code + (": "+detail if detail else ""))


def utc(value):
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise CanonicalError("INVALID_UTC_TIMESTAMP") from exc
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise CanonicalError("NAIVE_OR_INVALID_TIMESTAMP")
    try:
        if value.utcoffset() is None:
            raise CanonicalError("NAIVE_OR_INVALID_TIMESTAMP")
        return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")
    except (ValueError, TypeError, OverflowError) as exc:
        raise CanonicalError("INVALID_UTC_TIMESTAMP") from exc


def now_utc():
    return utc(datetime.now(timezone.utc))


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def finite_positive(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise CanonicalError("INVALID_EXPECTED_GOALS")
    return float(value)

