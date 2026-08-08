"""HMAC signing and fail-closed verification for serialized model artifacts."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, NoReturn

from src.core.exceptions import PredictionSystemError
from src.monitoring.alert_pipeline import dispatch_alert


ARTIFACT_SIGNING_KEY_ENV = "ARTIFACT_SIGNING_KEY"
ARTIFACT_INTEGRITY_FIELD = "artifact_integrity"
SIGNATURE_SCHEME = "hmac-sha256-v1"
MINIMUM_SIGNING_KEY_BYTES = 32

logger = logging.getLogger(__name__)


class ArtifactSigningError(PredictionSystemError):
    """Raised when an artifact cannot be signed for publication."""


class ArtifactVerificationError(PredictionSystemError):
    """Raised when an existing artifact fails authenticity verification."""


def _signing_key() -> bytes:
    raw_key = os.getenv(ARTIFACT_SIGNING_KEY_ENV, "")
    key = raw_key.encode("utf-8")
    if len(key) < MINIMUM_SIGNING_KEY_BYTES:
        raise ArtifactSigningError(
            f"{ARTIFACT_SIGNING_KEY_ENV} must contain at least "
            f"{MINIMUM_SIGNING_KEY_BYTES} bytes"
        )
    return key


def _normalized_filename(value: Any) -> str:
    return str(value or "").strip().replace("\\", "/")


def _signed_context(
    *,
    filename: str,
    model_name: str,
    version: str,
    league: Any,
) -> Dict[str, Any]:
    return {
        "filename": _normalized_filename(filename),
        "league": None if league in {None, ""} else str(league),
        "model_name": str(model_name or "").strip(),
        "version": str(version or "").strip(),
    }


def _context_bytes(context: Dict[str, Any]) -> bytes:
    return json.dumps(
        context,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _artifact_mac(key: bytes, context: Dict[str, Any], artifact_bytes: bytes) -> str:
    signer = hmac.new(key, digestmod=hashlib.sha256)
    signer.update(_context_bytes(context))
    signer.update(b"\x00")
    signer.update(artifact_bytes)
    return signer.hexdigest()


def _is_sha256_hex(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value)


def _reject(
    path: Path,
    *,
    version: str,
    reason: str,
    detail: str,
) -> NoReturn:
    logger.error(
        "artifact_verification_failed path=%s version=%s reason=%s",
        path,
        version or "unknown",
        reason,
    )
    dispatch_alert(
        source="artifact_verification_failure",
        severity="CRITICAL",
        message="Artifact verification rejected an unsafe artifact",
        context={
            "path": str(path),
            "version": version or "unknown",
            "reason": reason,
        },
    )
    raise ArtifactVerificationError(
        f"Artifact verification failed: {detail}",
        context={"path": str(path), "version": version, "reason": reason},
    )


def sign_artifact(
    path: Path,
    *,
    filename: str,
    model_name: str,
    version: str,
    league: Any = None,
) -> Dict[str, Any]:
    """Return manifest integrity metadata for a fully-written artifact."""
    context = _signed_context(
        filename=filename,
        model_name=model_name,
        version=version,
        league=league,
    )
    if not all((context["filename"], context["model_name"], context["version"])):
        raise ArtifactSigningError(
            "Artifact filename, model name, and version are required for signing",
            context={"path": str(path), "version": context["version"]},
        )

    try:
        artifact_bytes = path.read_bytes()
        key = _signing_key()
    except ArtifactSigningError:
        logger.error(
            "artifact_signing_failed path=%s version=%s reason=invalid_signing_key",
            path,
            context["version"],
        )
        raise
    except OSError as exc:
        logger.error(
            "artifact_signing_failed path=%s version=%s reason=read_failure",
            path,
            context["version"],
        )
        raise ArtifactSigningError(
            "Artifact could not be read for signing",
            context={"path": str(path), "version": context["version"]},
        ) from exc

    return {
        "scheme": SIGNATURE_SCHEME,
        **context,
        "artifact_sha256": hashlib.sha256(artifact_bytes).hexdigest(),
        "signature": _artifact_mac(key, context, artifact_bytes),
    }


def verify_artifact(
    path: Path,
    manifest_metadata: Dict[str, Any],
    *,
    requested_version: str,
) -> bytes:
    """Verify and return the exact bytes that are safe to deserialize."""
    requested_version = str(requested_version or "").strip()
    integrity = manifest_metadata.get(ARTIFACT_INTEGRITY_FIELD)
    if not isinstance(integrity, dict):
        _reject(
            path,
            version=requested_version,
            reason="missing_signature",
            detail="manifest signature is missing",
        )

    if integrity.get("scheme") != SIGNATURE_SCHEME:
        _reject(
            path,
            version=requested_version,
            reason="unsupported_signature_scheme",
            detail="signature scheme is missing or unsupported",
        )

    expected_digest = integrity.get("artifact_sha256")
    expected_signature = integrity.get("signature")
    if not _is_sha256_hex(expected_digest) or not _is_sha256_hex(expected_signature):
        _reject(
            path,
            version=requested_version,
            reason="missing_signature",
            detail="manifest signature or artifact digest is missing or malformed",
        )

    manifest_version = str(manifest_metadata.get("version") or "").strip()
    signed_version = str(integrity.get("version") or "").strip()
    if not requested_version or manifest_version != requested_version or signed_version != requested_version:
        _reject(
            path,
            version=requested_version,
            reason="version_mismatch",
            detail="requested, manifest, and signed versions do not match",
        )

    expected_context = _signed_context(
        filename=manifest_metadata.get("filename"),
        model_name=manifest_metadata.get("name"),
        version=manifest_version,
        league=manifest_metadata.get("league"),
    )
    signed_context = {
        key: integrity.get(key)
        for key in ("filename", "league", "model_name", "version")
    }
    if signed_context != expected_context:
        _reject(
            path,
            version=requested_version,
            reason="metadata_mismatch",
            detail="signed artifact identity does not match the manifest",
        )

    try:
        key = _signing_key()
    except ArtifactSigningError:
        _reject(
            path,
            version=requested_version,
            reason="missing_signing_key",
            detail="artifact verification key is unavailable",
        )

    try:
        artifact_bytes = path.read_bytes()
    except OSError as exc:
        logger.error(
            "artifact_verification_failed path=%s version=%s reason=read_failure",
            path,
            requested_version,
        )
        dispatch_alert(
            source="artifact_verification_failure",
            severity="CRITICAL",
            message="Artifact verification could not read an artifact",
            context={
                "path": str(path),
                "version": requested_version or "unknown",
                "reason": "read_failure",
            },
        )
        raise ArtifactVerificationError(
            "Artifact verification failed: artifact could not be read",
            context={
                "path": str(path),
                "version": requested_version,
                "reason": "read_failure",
            },
        ) from exc

    actual_digest = hashlib.sha256(artifact_bytes).hexdigest()
    if not hmac.compare_digest(actual_digest, expected_digest):
        _reject(
            path,
            version=requested_version,
            reason="digest_mismatch",
            detail="artifact bytes do not match the signed manifest",
        )

    actual_signature = _artifact_mac(key, expected_context, artifact_bytes)
    if not hmac.compare_digest(actual_signature, expected_signature):
        _reject(
            path,
            version=requested_version,
            reason="signature_mismatch",
            detail="artifact signature is invalid",
        )

    return artifact_bytes
