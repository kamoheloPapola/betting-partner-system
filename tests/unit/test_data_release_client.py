from __future__ import annotations

import hashlib
import io
import json
import logging
import tarfile
from pathlib import Path

import pytest
import requests

from src.data import release_client


@pytest.fixture(autouse=True)
def reset_fetch_state():
    release_client._reset_fetch_state_for_tests()
    try:
        yield
    finally:
        release_client._reset_fetch_state_for_tests()


class FakeResponse:
    def __init__(
        self,
        status_code: int,
        *,
        payload=None,
        content: bytes = b"",
        text: str | None = None,
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self.content = content
        self.text = text if text is not None else (
            json.dumps(payload) if payload is not None else content.decode("utf-8", errors="replace")
        )

    def json(self):
        if self._payload is None:
            raise ValueError("no JSON payload")
        return self._payload


class RecordingSession:
    def __init__(self, *responses) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def request(self, method, url, **kwargs):
        self.calls.append({"method": method, "url": url, **kwargs})
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def _write_source_data(root: Path) -> None:
    matches = root / "processed" / "matches"
    matches.mkdir(parents=True)
    (matches / "PL_2025.csv").write_text("league,home\nPL,Alpha\n", encoding="utf-8")
    (matches / "PL_upcoming.csv").write_text("league,home\nPL,Beta\n", encoding="utf-8")
    (root / "processed" / "season_maps.json").write_text(
        json.dumps({"PL": {"2025": ["Alpha", "Beta"]}}),
        encoding="utf-8",
    )


def _archive_bytes(tmp_path: Path) -> bytes:
    source = tmp_path / "source-data"
    _write_source_data(source)
    return release_client.build_processed_data_archive(data_dir=source)


def _release_payload() -> dict:
    return {
        "id": 101,
        "assets": [
            {
                "id": 202,
                "name": release_client.DATA_RELEASE_ASSET_NAME,
            }
        ],
    }


def _asset_payload(archive_bytes: bytes) -> dict:
    return {
        "id": 202,
        "name": release_client.DATA_RELEASE_ASSET_NAME,
        "browser_download_url": "https://example.invalid/processed-data.tar.gz",
        "digest": f"sha256:{hashlib.sha256(archive_bytes).hexdigest()}",
    }


def test_archive_contains_only_explicit_public_allowlist(tmp_path) -> None:
    data_dir = tmp_path / "data"
    _write_source_data(data_dir)
    (data_dir / "processed" / "private-note.txt").write_text(
        "must not be public",
        encoding="utf-8",
    )

    archive_bytes = release_client.build_processed_data_archive(data_dir=data_dir)

    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
        assert sorted(archive.getnames()) == [
            "processed/matches/PL_2025.csv",
            "processed/matches/PL_upcoming.csv",
            "processed/season_maps.json",
        ]


def test_publish_creates_fixed_non_draft_release_and_uploads_asset(tmp_path) -> None:
    archive_bytes = _archive_bytes(tmp_path)
    digest = f"sha256:{hashlib.sha256(archive_bytes).hexdigest()}"
    session = RecordingSession(
        FakeResponse(404, payload={"message": "Not Found"}),
        FakeResponse(201, payload={"id": 101, "assets": []}),
        FakeResponse(
            201,
            payload={"id": 202, "name": release_client.DATA_RELEASE_ASSET_NAME, "digest": digest},
        ),
    )

    uploaded = release_client.GitHubDataReleaseClient(session=session).publish(
        archive_bytes,
        token="test-token",
    )

    assert uploaded["name"] == release_client.DATA_RELEASE_ASSET_NAME
    assert [call["method"] for call in session.calls] == ["GET", "POST", "POST"]
    create_call = session.calls[1]
    assert create_call["json"] == {
        "tag_name": "data-nightly",
        "name": "data-nightly",
        "body": "Latest processed match data for the prediction API.",
        "draft": False,
        "prerelease": False,
        "make_latest": "false",
    }
    upload_call = session.calls[2]
    assert upload_call["params"] == {"name": "processed-data.tar.gz"}
    assert upload_call["data"] == archive_bytes
    assert upload_call["headers"]["Content-Type"] == "application/octet-stream"
    assert all(call["timeout"] == (10.0, 120.0) for call in session.calls)


def test_publish_replaces_asset_without_recreating_release(tmp_path) -> None:
    archive_bytes = _archive_bytes(tmp_path)
    digest = f"sha256:{hashlib.sha256(archive_bytes).hexdigest()}"
    session = RecordingSession(
        FakeResponse(
            200,
            payload={
                "id": 101,
                "assets": [
                    {"id": 202, "name": "processed-data.tar.gz"},
                    {"id": 303, "name": "unrelated.txt"},
                ],
            },
        ),
        FakeResponse(204),
        FakeResponse(
            201,
            payload={"id": 404, "name": "processed-data.tar.gz", "digest": digest},
        ),
    )

    release_client.GitHubDataReleaseClient(session=session).publish(
        archive_bytes,
        token="test-token",
    )

    assert [call["method"] for call in session.calls] == ["GET", "DELETE", "POST"]
    assert session.calls[1]["url"].endswith("/releases/assets/202")
    assert session.calls[2]["url"].endswith("/releases/101/assets")


def test_fetch_verifies_digest_and_replaces_processed_data(tmp_path) -> None:
    archive_bytes = _archive_bytes(tmp_path)
    session = RecordingSession(
        FakeResponse(200, payload=_release_payload()),
        FakeResponse(200, payload=_asset_payload(archive_bytes)),
        FakeResponse(200, content=archive_bytes),
    )
    destination = tmp_path / "destination"
    old_matches = destination / "processed" / "matches"
    old_matches.mkdir(parents=True)
    (old_matches / "stale.csv").write_text("stale\n", encoding="utf-8")
    (destination / "processed" / "season_maps.json").write_text(
        "{}",
        encoding="utf-8",
    )

    release_client.GitHubDataReleaseClient(session=session).fetch_and_install(
        data_dir=destination
    )

    assert not (old_matches / "stale.csv").exists()
    assert (old_matches / "PL_2025.csv").read_text(encoding="utf-8").startswith("league,home")
    assert json.loads(
        (destination / "processed" / "season_maps.json").read_text(encoding="utf-8")
    ) == {"PL": {"2025": ["Alpha", "Beta"]}}
    assert all(call["timeout"] == (10.0, 120.0) for call in session.calls)
    assert "Authorization" not in session.calls[0]["headers"]
    assert "Authorization" not in session.calls[1]["headers"]
    assert "Authorization" not in session.calls[2]["headers"]


@pytest.mark.parametrize("failure_call", [1, 2, 3, 4])
def test_install_rolls_back_partial_swap_when_replace_fails(
    tmp_path,
    monkeypatch,
    failure_call,
) -> None:
    archive_bytes = _archive_bytes(tmp_path)
    destination = tmp_path / "destination"
    old_matches = destination / "processed" / "matches"
    old_matches.mkdir(parents=True)
    old_match = old_matches / "existing.csv"
    old_match.write_text("old match data\n", encoding="utf-8")
    old_season_map = destination / "processed" / "season_maps.json"
    old_season_map.write_text('{"old": true}', encoding="utf-8")
    real_replace = release_client.os.replace
    replace_calls = 0

    def fail_during_swap(source, target):
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == failure_call:
            raise OSError(f"simulated replace failure on call {failure_call}")
        real_replace(source, target)

    monkeypatch.setattr(release_client.os, "replace", fail_during_swap)

    with pytest.raises(
        release_client.DataReleaseError,
        match="Could not install processed data",
    ):
        release_client.install_processed_data_archive(
            archive_bytes,
            data_dir=destination,
        )

    assert old_match.read_text(encoding="utf-8") == "old match data\n"
    assert sorted(path.name for path in old_matches.iterdir()) == ["existing.csv"]
    assert old_season_map.read_text(encoding="utf-8") == '{"old": true}'
    assert not list(destination.glob(".data-release-staging-*"))
    assert not list((destination / "processed").glob(".matches-backup-*"))
    assert not list((destination / "processed").glob(".season-map-backup-*"))


def test_fetch_404_degrades_and_marks_data_unfresh(tmp_path, caplog) -> None:
    release_client._reset_fetch_state_for_tests()
    session = RecordingSession(FakeResponse(404, payload={"message": "Not Found"}))

    with caplog.at_level(logging.WARNING, logger=release_client.__name__):
        fetched = release_client.fetch_processed_data_once(
            data_dir=tmp_path,
            client=release_client.GitHubDataReleaseClient(session=session),
        )

    assert fetched is False
    assert release_client.data_release_fetch_failed() is True
    assert "has not been published yet" in caplog.text


def test_fetch_transient_missing_asset_uses_the_same_degraded_path(
    tmp_path,
    caplog,
) -> None:
    session = RecordingSession(
        FakeResponse(200, payload={"id": 101, "assets": []}),
    )

    with caplog.at_level(logging.WARNING, logger=release_client.__name__):
        fetched = release_client.fetch_processed_data_once(
            data_dir=tmp_path,
            client=release_client.GitHubDataReleaseClient(session=session),
        )

    assert fetched is False
    assert release_client.data_release_fetch_failed() is True
    assert "does not contain exactly one" in caplog.text


def test_corrupt_archive_fails_loudly_without_replacing_existing_data(
    tmp_path,
    caplog,
) -> None:
    release_client._reset_fetch_state_for_tests()
    corrupt = b"not a tar archive"
    session = RecordingSession(
        FakeResponse(200, payload=_release_payload()),
        FakeResponse(200, payload=_asset_payload(corrupt)),
        FakeResponse(200, content=corrupt),
    )
    existing = tmp_path / "processed" / "matches"
    existing.mkdir(parents=True)
    existing_file = existing / "existing.csv"
    existing_file.write_text("still here\n", encoding="utf-8")

    with caplog.at_level(logging.ERROR, logger=release_client.__name__):
        fetched = release_client.fetch_processed_data_once(
            data_dir=tmp_path,
            client=release_client.GitHubDataReleaseClient(session=session),
        )

    assert fetched is False
    assert existing_file.read_text(encoding="utf-8") == "still here\n"
    assert "archive is corrupt" in caplog.text.lower()


def test_digest_mismatch_is_rejected_before_extraction(tmp_path) -> None:
    archive_bytes = _archive_bytes(tmp_path)
    asset_payload = _asset_payload(archive_bytes)
    asset_payload["digest"] = "sha256:" + ("0" * 64)
    session = RecordingSession(
        FakeResponse(200, payload=_release_payload()),
        FakeResponse(200, payload=asset_payload),
        FakeResponse(200, content=archive_bytes),
    )

    with pytest.raises(release_client.DataReleaseIntegrityError, match="digest"):
        release_client.GitHubDataReleaseClient(session=session).download()


def test_timeout_bound_is_passed_to_http_layer_and_degrades(tmp_path, caplog) -> None:
    release_client._reset_fetch_state_for_tests()
    session = RecordingSession(requests.Timeout("simulated hang"))

    with caplog.at_level(logging.ERROR, logger=release_client.__name__):
        fetched = release_client.fetch_processed_data_once(
            data_dir=tmp_path,
            client=release_client.GitHubDataReleaseClient(session=session),
        )

    assert fetched is False
    assert session.calls[0]["timeout"] == (10.0, 120.0)
    assert "connect=10.0s, transfer=120.0s" in caplog.text


def test_process_guard_prevents_redundant_fetches(tmp_path) -> None:
    release_client._reset_fetch_state_for_tests()

    class SuccessfulClient:
        calls = 0

        def fetch_and_install(self, *, data_dir):
            self.calls += 1

    client = SuccessfulClient()

    assert release_client.fetch_processed_data_once(data_dir=tmp_path, client=client) is True
    assert release_client.fetch_processed_data_once(data_dir=tmp_path, client=client) is True
    assert client.calls == 1


def test_process_guard_allows_retry_after_failure(tmp_path) -> None:
    class FlakyClient:
        calls = 0

        def fetch_and_install(self, *, data_dir):
            self.calls += 1
            if self.calls == 1:
                raise release_client.DataReleaseNotFound("temporary 404")

    client = FlakyClient()

    assert release_client.fetch_processed_data_once(data_dir=tmp_path, client=client) is False
    assert release_client.fetch_processed_data_once(data_dir=tmp_path, client=client) is True
    assert client.calls == 2


def test_reload_season_maps_updates_in_process_mapping(tmp_path) -> None:
    from src.utils import naming

    original = {
        league: {season: set(teams) for season, teams in seasons.items()}
        for league, seasons in naming.LEAGUE_TEAMS.items()
    }
    replacement = tmp_path / "season_maps.json"
    replacement.write_text(
        json.dumps({"PL": {"2026": ["New Club"]}}),
        encoding="utf-8",
    )
    try:
        naming.reload_season_maps(replacement)
        assert naming.LEAGUE_TEAMS == {"PL": {2026: {"New Club"}}}
    finally:
        naming.LEAGUE_TEAMS.clear()
        naming.LEAGUE_TEAMS.update(original)
