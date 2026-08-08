from __future__ import annotations

import hmac
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, NoReturn

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.monitoring.alert_pipeline import dispatch_alert


CLI_AUTH_TOKENS_ENV = "CLI_AUTH_TOKENS"
CLI_ADMIN_ROLE = "admin"

logger = logging.getLogger(__name__)
_bearer_scheme = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class CliPrincipal:
    role: str


def _request_context(request: Request) -> tuple[str, str, str]:
    timestamp = datetime.now(timezone.utc).isoformat()
    caller_ip = request.client.host if request.client else "unknown"
    return timestamp, caller_ip, request.url.path


def _log_denied(request: Request, *, reason: str) -> None:
    timestamp, caller_ip, route = _request_context(request)
    logger.warning(
        "cli_auth_denied timestamp=%s caller_ip=%s route=%s reason=%s",
        timestamp,
        caller_ip,
        route,
        reason,
    )
    dispatch_alert(
        source="cli_auth_denial",
        severity="WARNING",
        message="CLI authorization denied",
        context={
            "caller_ip": caller_ip,
            "route": route,
            "reason": reason,
            "status_code": 401,
        },
    )


def _log_forbidden(request: Request, *, role: str) -> None:
    timestamp, caller_ip, route = _request_context(request)
    logger.warning(
        "cli_auth_forbidden timestamp=%s caller_ip=%s route=%s role=%s",
        timestamp,
        caller_ip,
        route,
        role,
    )
    dispatch_alert(
        source="cli_auth_denial",
        severity="WARNING",
        message="CLI authorization forbidden",
        context={
            "caller_ip": caller_ip,
            "route": route,
            "role": role,
            "status_code": 403,
        },
    )


def _configured_tokens() -> Dict[str, str]:
    raw_tokens = os.getenv(CLI_AUTH_TOKENS_ENV, "")
    if not raw_tokens:
        return {}

    try:
        parsed = json.loads(raw_tokens)
    except (TypeError, json.JSONDecodeError):
        logger.error("Invalid %s configuration; CLI API access is disabled", CLI_AUTH_TOKENS_ENV)
        return {}

    if not isinstance(parsed, dict):
        logger.error("%s must be a JSON object; CLI API access is disabled", CLI_AUTH_TOKENS_ENV)
        return {}

    configured: Dict[str, str] = {}
    for token, role in parsed.items():
        if isinstance(token, str) and token and isinstance(role, str) and role:
            configured[token] = role.strip().lower()
    return configured


def _role_for_token(candidate: str, configured_tokens: Dict[str, str]) -> str | None:
    candidate_bytes = candidate.encode("utf-8")
    matched_role = None
    for token, role in configured_tokens.items():
        if hmac.compare_digest(candidate_bytes, token.encode("utf-8")):
            matched_role = role
    return matched_role


def _unauthorized(request: Request, *, reason: str) -> NoReturn:
    _log_denied(request, reason=reason)
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="unauthorized",
        headers={"WWW-Authenticate": "Bearer"},
    )


def require_cli_admin(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> CliPrincipal:
    if credentials is None or credentials.scheme.lower() != "bearer":
        _unauthorized(request, reason="missing_or_malformed_credentials")

    configured_tokens = _configured_tokens()
    role = _role_for_token(credentials.credentials, configured_tokens)
    if role is None:
        _unauthorized(request, reason="invalid_credentials")

    if role != CLI_ADMIN_ROLE:
        _log_forbidden(request, role=role)
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="forbidden")

    return CliPrincipal(role=role)
