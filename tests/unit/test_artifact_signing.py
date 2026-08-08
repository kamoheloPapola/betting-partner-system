from __future__ import annotations

import json
import logging
import pickle
from unittest.mock import patch

import joblib
import pytest

from src.ml import registry as registry_module
from src.ml.artifact_signing import (
    ARTIFACT_INTEGRITY_FIELD,
    ArtifactSigningError,
    ArtifactVerificationError,
    sign_artifact,
)
from src.ml.registry import ModelRegistry


SIGNING_KEY = "test-signing-key-with-at-least-32-bytes"


class _SignedModel:
    def __init__(self, value: str) -> None:
        self.value = value


@pytest.fixture
def signed_registry(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_SIGNING_KEY", SIGNING_KEY)
    monkeypatch.setattr(registry_module, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(ModelRegistry, "MANIFEST_FILE", tmp_path / "manifest.json")
    monkeypatch.setattr(ModelRegistry, "BACKUP_FILE", tmp_path / "manifest.json.bak")
    ModelRegistry._instance = None
    ModelRegistry._manifest_cache = None

    registry = ModelRegistry()
    try:
        yield registry
    finally:
        ModelRegistry._instance = None
        ModelRegistry._manifest_cache = None


def _publish_model(registry, tmp_path, *, suffix=".pkl"):
    filename = f"signed-model{suffix}"
    artifact = tmp_path / filename
    model = _SignedModel("trusted")
    if suffix == ".joblib":
        joblib.dump(model, artifact)
    else:
        artifact.write_bytes(pickle.dumps(model))

    registry.register_model(
        "signed_model",
        "1.0.0",
        {
            "filename": filename,
            "status": "productive",
            "league": "PL",
        },
    )
    manifest_key = "signed_model_v1.0.0_PL"
    registry.set_active_model("signed_model", manifest_key, league="PL")
    return artifact, registry.manifest[manifest_key]


@pytest.mark.parametrize("suffix", [".pkl", ".joblib"])
def test_validly_signed_artifact_loads(signed_registry, tmp_path, suffix):
    _publish_model(signed_registry, tmp_path, suffix=suffix)

    loaded = signed_registry.load_model("signed_model", league="PL")

    assert isinstance(loaded, _SignedModel)
    assert loaded.value == "trusted"
    persisted_manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert ARTIFACT_INTEGRITY_FIELD in persisted_manifest["signed_model_v1.0.0_PL"]


@pytest.mark.parametrize("suffix", [".pkl", ".joblib"])
def test_single_byte_tamper_is_rejected_before_any_deserialization(
    signed_registry,
    tmp_path,
    caplog,
    suffix,
):
    artifact, _metadata = _publish_model(signed_registry, tmp_path, suffix=suffix)
    tampered = bytearray(artifact.read_bytes())
    tampered[-1] ^= 0x01
    artifact.write_bytes(tampered)

    with caplog.at_level(logging.ERROR, logger="src.ml.artifact_signing"):
        with patch.object(registry_module.pickle, "load") as pickle_load, patch.object(
            registry_module.joblib,
            "load",
        ) as joblib_load:
            with pytest.raises(ArtifactVerificationError, match="verification failed") as exc_info:
                signed_registry.load_model("signed_model", league="PL")

    pickle_load.assert_not_called()
    joblib_load.assert_not_called()
    assert exc_info.value.context["reason"] == "digest_mismatch"
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert f"path={artifact}" in messages
    assert "version=1.0.0" in messages
    assert "reason=digest_mismatch" in messages


def test_missing_signature_fails_closed_before_deserialization(signed_registry, tmp_path):
    _artifact, metadata = _publish_model(signed_registry, tmp_path)
    metadata.pop(ARTIFACT_INTEGRITY_FIELD)

    with patch.object(registry_module.pickle, "load") as deserialize:
        with pytest.raises(ArtifactVerificationError) as exc_info:
            signed_registry.load_model("signed_model", league="PL")

    deserialize.assert_not_called()
    assert exc_info.value.context["reason"] == "missing_signature"


def test_valid_signature_for_wrong_version_is_rejected(signed_registry, tmp_path):
    artifact, metadata = _publish_model(signed_registry, tmp_path)
    metadata[ARTIFACT_INTEGRITY_FIELD] = sign_artifact(
        artifact,
        filename=artifact.name,
        model_name="signed_model",
        version="0.9.0",
        league="PL",
    )

    with patch.object(registry_module.pickle, "load") as deserialize:
        with pytest.raises(ArtifactVerificationError) as exc_info:
            signed_registry.load_model("signed_model", league="PL")

    deserialize.assert_not_called()
    assert exc_info.value.context["reason"] == "version_mismatch"


def test_deserializer_uses_verified_bytes_if_file_changes_after_verification(
    signed_registry,
    tmp_path,
    monkeypatch,
):
    artifact, _metadata = _publish_model(signed_registry, tmp_path)
    real_verify = registry_module.verify_artifact

    def verify_then_replace(path, manifest_metadata, *, requested_version):
        verified_bytes = real_verify(
            path,
            manifest_metadata,
            requested_version=requested_version,
        )
        artifact.write_bytes(pickle.dumps(_SignedModel("replacement")))
        return verified_bytes

    monkeypatch.setattr(registry_module, "verify_artifact", verify_then_replace)

    loaded = signed_registry.load_model("signed_model", league="PL")

    assert loaded.value == "trusted"


def test_missing_signing_key_prevents_publication(signed_registry, tmp_path, monkeypatch):
    artifact = tmp_path / "unsigned.pkl"
    artifact.write_bytes(pickle.dumps(_SignedModel("unsigned")))
    monkeypatch.delenv("ARTIFACT_SIGNING_KEY")

    with pytest.raises(ArtifactSigningError, match="at least 32 bytes"):
        signed_registry.register_model(
            "unsigned_model",
            "1.0.0",
            {
                "filename": artifact.name,
                "status": "productive",
                "league": "PL",
            },
        )

    assert "unsigned_model_v1.0.0_PL" not in signed_registry.manifest
