"""
In-memory API cache with TTL support.

Predictions are expensive to compute, so we cache them per league and
invalidate them when training refreshes the model set.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional


MODEL_HEALTH_CACHE_KEY = "model_health"


def prediction_cache_key(league: Optional[str], limit: Optional[int] = None) -> str:
    league_key = str(league or "ALL").strip().upper() or "ALL"
    return f"predictions_{league_key}"


def slip_cache_key(league: Optional[str], min_prob: float, max_selections: int) -> str:
    league_key = str(league or "ALL").strip().upper() or "ALL"
    return f"slip_{league_key}_{min_prob:.4f}_{int(max_selections)}"


class PredictionCache:
    def __init__(self, ttl_seconds: int = 600) -> None:
        self._store: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self.ttl = ttl_seconds

    def get(self, key: str) -> Any:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None

            entry_ttl = int(entry.get("ttl", self.ttl))
            if time.monotonic() - float(entry["ts"]) > entry_ttl:
                del self._store[key]
                return None

            return entry["data"]

    def set(self, key: str, data: Any, ttl_seconds: Optional[int] = None) -> None:
        with self._lock:
            self._store[key] = {
                "data": data,
                "ts": time.monotonic(),
                "ttl": int(ttl_seconds or self.ttl),
            }

    def invalidate(self, key: Optional[str] = None) -> None:
        with self._lock:
            if key:
                self._store.pop(key, None)
            else:
                self._store.clear()

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            now = time.monotonic()
            return {
                "cached_keys": list(self._store.keys()),
                "entry_ages_seconds": {
                    key: round(now - float(value["ts"])) for key, value in self._store.items()
                },
                "entry_ttls_seconds": {
                    key: int(value.get("ttl", self.ttl)) for key, value in self._store.items()
                },
            }


prediction_cache = PredictionCache(ttl_seconds=600)
