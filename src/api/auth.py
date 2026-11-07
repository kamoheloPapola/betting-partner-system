"""Shared-secret API key guard for destructive endpoints."""
from __future__ import annotations

import os

from fastapi import Header, HTTPException, status

_UNSET = object()
_api_key: object = _UNSET


def _get_configured_key() -> str | None:
    global _api_key
    if _api_key is _UNSET:
        _api_key = os.environ.get("PITCHSENSE_API_KEY", "").strip() or None
    return _api_key  # type: ignore[return-value]


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """FastAPI dependency. Raises 403 if key is wrong or env var is unset."""
    configured = _get_configured_key()
    if configured is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API key auth is not configured on this server.",
        )
    if x_api_key != configured:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing X-Api-Key header.",
        )
