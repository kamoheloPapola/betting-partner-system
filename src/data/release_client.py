"""Publish and fetch processed match data through a fixed GitHub Release."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import shutil
import tarfile
import tempfile
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Optional

import requests
from dotenv import dotenv_values

from src.config import DATA_DIR


logger = logging.getLogger(__name__)

GITHUB_OWNER = "kamoheloPapola"
GITHUB_REPOSITORY = "betting-partner-system"
GITHUB_API_BASE = "https://api.github.com"
GITHUB_UPLOADS_BASE = "https://uploads.github.com"
DATA_RELEASE_TAG = "data-nightly"
DATA_RELEASE_ASSET_NAME = "processed-data.tar.gz"

# Match the 10s/120s artifact-transfer bounds already settled in the reviewed,
# still-unmerged registry work so later lineage reconciliation adds no new values.
CONNECT_TIMEOUT_SECONDS = 10.0
TRANSFER_TIMEOUT_SECONDS = 120.0
REQUEST_TIMEOUT = (CONNECT_TIMEOUT_SECONDS, TRANSFER_TIMEOUT_SECONDS)

REPO_ROOT = Path(__file__).resolve().parents[2]
SEASON_MAP_ARCHIVE_PATH = PurePosixPath("processed/season_maps.json")
MATCH_ARCHIVE_ROOT = PurePosixPath("processed/matches")


class DataReleaseError(RuntimeError):
    """Base error for processed-data Release operations."""


class DataReleaseNotFound(DataReleaseError):
    """Raised when the fixed data Release or its asset does not exist yet."""


class DataReleaseIntegrityError(DataReleaseError):
    """Raised when downloaded data fails digest or archive validation."""


@dataclass(frozen=True)
class DataReleaseFetchState:
    attempted: bool = False
    successful: bool = False
    error: Optional[str] = None


_fetch_lock = threading.Lock()
_fetch_state = DataReleaseFetchState()


def get_data_release_fetch_state() -> DataReleaseFetchState:
    """Return the process-local startup fetch state."""
    return _fetch_state


def data_release_fetch_failed() -> bool:
    """Return whether startup attempted the Release fetch and it failed."""
    state = get_data_release_fetch_state()
    return state.attempted and not state.successful


def _set_fetch_state(*, successful: bool, error: Optional[str] = None) -> None:
    global _fetch_state
    _fetch_state = DataReleaseFetchState(
        attempted=True,
        successful=successful,
        error=error,
    )


def _reset_fetch_state_for_tests() -> None:
    """Reset the process-local guard for isolated tests."""
    global _fetch_state
    _fetch_state = DataReleaseFetchState()


def _response_json(response: requests.Response, *, operation: str) -> Mapping[str, Any]:
    try:
        payload = response.json()
    except (ValueError, json.JSONDecodeError) as exc:
        raise DataReleaseError(
            f"GitHub {operation} returned invalid JSON"
        ) from exc
    if not isinstance(payload, Mapping):
        raise DataReleaseError(f"GitHub {operation} returned a non-object response")
    return payload


def _api_error(response: requests.Response, *, operation: str) -> DataReleaseError:
    body = response.text.strip()
    detail = body[:500] if body else "empty response body"
    return DataReleaseError(
        f"GitHub {operation} failed with HTTP {response.status_code}: {detail}"
    )


def resolve_publish_token(*, repo_root: Path = REPO_ROOT) -> str:
    """Resolve a repository-scoped Release token without logging its value."""
    token = (os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN") or "").strip()
    if token:
        return token

    local_env = repo_root / ".env.local"
    if local_env.is_file():
        values = dotenv_values(local_env)
        token = str(values.get("GH_TOKEN") or values.get("GITHUB_TOKEN") or "").strip()
        if token:
            return token

    raise DataReleaseError(
        "GH_TOKEN or GITHUB_TOKEN is required to publish processed data"
    )


def build_processed_data_archive(*, data_dir: Path = DATA_DIR) -> bytes:
    """Build an archive from the explicit public processed-data allowlist."""
    matches_dir = data_dir / "processed" / "matches"
    season_maps_path = data_dir / "processed" / "season_maps.json"
    match_files = sorted(
        path
        for path in matches_dir.glob("*.csv")
        if path.is_file() and not path.is_symlink()
    )

    if not match_files:
        raise DataReleaseError(f"No processed match CSV files found in {matches_dir}")
    if not season_maps_path.is_file() or season_maps_path.is_symlink():
        raise DataReleaseError(
            f"Required season map is missing: {season_maps_path}"
        )

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for source in match_files:
            archive.add(
                source,
                arcname=str(MATCH_ARCHIVE_ROOT / source.name),
                recursive=False,
            )
        archive.add(
            season_maps_path,
            arcname=str(SEASON_MAP_ARCHIVE_PATH),
            recursive=False,
        )
    return buffer.getvalue()


def _validated_archive_files(archive_bytes: bytes) -> dict[PurePosixPath, bytes]:
    files: dict[PurePosixPath, bytes] = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
            for member in archive.getmembers():
                if not member.isfile():
                    raise DataReleaseIntegrityError(
                        f"Archive contains unsupported member type: {member.name}"
                    )
                member_path = PurePosixPath(member.name)
                is_match_csv = (
                    member_path.parent == MATCH_ARCHIVE_ROOT
                    and member_path.suffix.lower() == ".csv"
                    and member_path.name not in {"", ".", ".."}
                )
                if member_path != SEASON_MAP_ARCHIVE_PATH and not is_match_csv:
                    raise DataReleaseIntegrityError(
                        f"Archive contains unexpected path: {member.name}"
                    )
                if member_path in files:
                    raise DataReleaseIntegrityError(
                        f"Archive contains duplicate path: {member.name}"
                    )
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise DataReleaseIntegrityError(
                        f"Archive member cannot be read: {member.name}"
                    )
                files[member_path] = extracted.read()
    except (tarfile.TarError, OSError) as exc:
        raise DataReleaseIntegrityError(
            f"Processed-data archive is corrupt: {exc}"
        ) from exc

    match_paths = [path for path in files if path.parent == MATCH_ARCHIVE_ROOT]
    if not match_paths:
        raise DataReleaseIntegrityError("Archive contains no processed match CSV files")
    if SEASON_MAP_ARCHIVE_PATH not in files:
        raise DataReleaseIntegrityError("Archive is missing processed/season_maps.json")
    if any(not files[path] for path in match_paths):
        raise DataReleaseIntegrityError("Archive contains an empty match CSV file")
    try:
        json.loads(files[SEASON_MAP_ARCHIVE_PATH].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DataReleaseIntegrityError("Archive season map is not valid JSON") from exc
    return files


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def install_processed_data_archive(
    archive_bytes: bytes,
    *,
    data_dir: Path = DATA_DIR,
) -> None:
    """Validate in staging, then replace processed matches and the season map."""
    files = _validated_archive_files(archive_bytes)
    data_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=".data-release-staging-", dir=str(data_dir))
    )
    staged_matches = staging_root / "processed" / "matches"
    staged_season_map = staging_root / "processed" / "season_maps.json"
    staged_matches.mkdir(parents=True)
    for archive_path, content in files.items():
        if archive_path == SEASON_MAP_ARCHIVE_PATH:
            staged_season_map.parent.mkdir(parents=True, exist_ok=True)
            staged_season_map.write_bytes(content)
        else:
            (staged_matches / archive_path.name).write_bytes(content)

    target_matches = processed_dir / "matches"
    target_season_map = processed_dir / "season_maps.json"
    transaction_id = uuid.uuid4().hex
    matches_backup = processed_dir / f".matches-backup-{transaction_id}"
    season_map_backup = processed_dir / f".season-map-backup-{transaction_id}.json"
    matches_backed_up = False
    season_map_backed_up = False
    matches_installed = False
    season_map_installed = False

    # Extraction never touches the live tree. Each os.replace below is atomic,
    # but the two-target swap is not one atomic transaction; backups support
    # rollback when a caught filesystem error interrupts the sequence.
    try:
        if target_matches.exists():
            os.replace(target_matches, matches_backup)
            matches_backed_up = True
        if target_season_map.exists():
            os.replace(target_season_map, season_map_backup)
            season_map_backed_up = True

        os.replace(staged_matches, target_matches)
        matches_installed = True
        os.replace(staged_season_map, target_season_map)
        season_map_installed = True
    except OSError as exc:
        if matches_installed:
            _remove_path(target_matches)
        if season_map_installed:
            _remove_path(target_season_map)
        if matches_backed_up and matches_backup.exists():
            os.replace(matches_backup, target_matches)
        if season_map_backed_up and season_map_backup.exists():
            os.replace(season_map_backup, target_season_map)
        raise DataReleaseError(
            f"Could not install processed data into {processed_dir}: {exc}"
        ) from exc
    else:
        _remove_path(matches_backup)
        _remove_path(season_map_backup)
    finally:
        _remove_path(staging_root)


class GitHubDataReleaseClient:
    """Minimal GitHub Releases client for the fixed processed-data asset."""

    def __init__(self, *, session: Optional[requests.Session] = None) -> None:
        self.session = session or requests.Session()

    @staticmethod
    def _headers(*, token: Optional[str] = None, accept: str = "application/vnd.github+json") -> dict[str, str]:
        headers = {
            "Accept": accept,
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "betting-partner-system-data-release",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def _request(self, method: str, url: str, *, operation: str, **kwargs: Any) -> requests.Response:
        try:
            return self.session.request(
                method,
                url,
                timeout=REQUEST_TIMEOUT,
                **kwargs,
            )
        except requests.Timeout as exc:
            raise DataReleaseError(
                f"GitHub {operation} timed out "
                f"(connect={CONNECT_TIMEOUT_SECONDS}s, transfer={TRANSFER_TIMEOUT_SECONDS}s)"
            ) from exc
        except requests.RequestException as exc:
            raise DataReleaseError(f"GitHub {operation} failed: {exc}") from exc

    def publish(self, archive_bytes: bytes, *, token: str) -> Mapping[str, Any]:
        """Replace the fixed asset while preserving the Release ID."""
        release_url = (
            f"{GITHUB_API_BASE}/repos/{GITHUB_OWNER}/{GITHUB_REPOSITORY}"
            f"/releases/tags/{DATA_RELEASE_TAG}"
        )
        auth_headers = self._headers(token=token)
        lookup = self._request(
            "GET",
            release_url,
            operation="data Release lookup",
            headers=auth_headers,
        )
        if lookup.status_code == 404:
            create = self._request(
                "POST",
                f"{GITHUB_API_BASE}/repos/{GITHUB_OWNER}/{GITHUB_REPOSITORY}/releases",
                operation="data Release creation",
                headers=auth_headers,
                json={
                    "tag_name": DATA_RELEASE_TAG,
                    "name": DATA_RELEASE_TAG,
                    "body": "Latest processed match data for the prediction API.",
                    "draft": False,
                    "prerelease": False,
                    "make_latest": "false",
                },
            )
            if create.status_code != 201:
                raise _api_error(create, operation="data Release creation")
            release = _response_json(create, operation="data Release creation")
        elif lookup.status_code == 200:
            release = _response_json(lookup, operation="data Release lookup")
        else:
            raise _api_error(lookup, operation="data Release lookup")

        release_id = release.get("id")
        if not isinstance(release_id, int):
            raise DataReleaseError("GitHub data Release response is missing an integer id")

        assets = release.get("assets", [])
        if not isinstance(assets, list):
            raise DataReleaseError("GitHub data Release response has malformed assets")
        for asset in assets:
            if not isinstance(asset, Mapping) or asset.get("name") != DATA_RELEASE_ASSET_NAME:
                continue
            asset_id = asset.get("id")
            if not isinstance(asset_id, int):
                raise DataReleaseError("Existing processed-data asset is missing an integer id")
            deleted = self._request(
                "DELETE",
                f"{GITHUB_API_BASE}/repos/{GITHUB_OWNER}/{GITHUB_REPOSITORY}/releases/assets/{asset_id}",
                operation="existing processed-data asset deletion",
                headers=auth_headers,
            )
            if deleted.status_code != 204:
                raise _api_error(
                    deleted,
                    operation="existing processed-data asset deletion",
                )

        # Replacement preserves the Release ID but is delete-then-upload, so a
        # concurrent fetch can see a temporary 404. GitHub exposes the new asset
        # only after upload completes, never as partially uploaded bytes.
        uploaded = self._request(
            "POST",
            f"{GITHUB_UPLOADS_BASE}/repos/{GITHUB_OWNER}/{GITHUB_REPOSITORY}/releases/{release_id}/assets",
            operation="processed-data asset upload",
            headers={
                **self._headers(token=token, accept="application/vnd.github+json"),
                "Content-Type": "application/octet-stream",
            },
            params={"name": DATA_RELEASE_ASSET_NAME},
            data=archive_bytes,
        )
        if uploaded.status_code != 201:
            raise _api_error(uploaded, operation="processed-data asset upload")
        uploaded_asset = _response_json(uploaded, operation="processed-data asset upload")
        expected_digest = f"sha256:{hashlib.sha256(archive_bytes).hexdigest()}"
        if uploaded_asset.get("digest") != expected_digest:
            raise DataReleaseIntegrityError(
                "GitHub uploaded asset digest does not match the local archive"
            )
        return uploaded_asset

    def download(self) -> bytes:
        """Download and verify the public fixed-tag processed-data asset."""
        release = self._request(
            "GET",
            f"{GITHUB_API_BASE}/repos/{GITHUB_OWNER}/{GITHUB_REPOSITORY}/releases/tags/{DATA_RELEASE_TAG}",
            operation="data Release lookup",
            headers=self._headers(),
        )
        if release.status_code == 404:
            raise DataReleaseNotFound(
                f"GitHub Release {DATA_RELEASE_TAG} has not been published yet"
            )
        if release.status_code != 200:
            raise _api_error(release, operation="data Release lookup")
        release_payload = _response_json(release, operation="data Release lookup")
        assets = release_payload.get("assets", [])
        matching_assets = [
            asset
            for asset in assets
            if isinstance(asset, Mapping)
            and asset.get("name") == DATA_RELEASE_ASSET_NAME
        ] if isinstance(assets, list) else []
        if len(matching_assets) != 1:
            raise DataReleaseNotFound(
                f"GitHub Release {DATA_RELEASE_TAG} does not contain exactly one "
                f"{DATA_RELEASE_ASSET_NAME} asset"
            )
        asset_id = matching_assets[0].get("id")
        if not isinstance(asset_id, int):
            raise DataReleaseError("Processed-data asset has no integer id")
        metadata_response = self._request(
            "GET",
            f"{GITHUB_API_BASE}/repos/{GITHUB_OWNER}/{GITHUB_REPOSITORY}/releases/assets/{asset_id}",
            operation="processed-data asset metadata lookup",
            headers=self._headers(),
        )
        if metadata_response.status_code == 404:
            raise DataReleaseNotFound(
                f"GitHub asset {DATA_RELEASE_ASSET_NAME} disappeared during fetch"
            )
        if metadata_response.status_code != 200:
            raise _api_error(
                metadata_response,
                operation="processed-data asset metadata lookup",
            )
        asset = _response_json(
            metadata_response,
            operation="processed-data asset metadata lookup",
        )
        download_url = asset.get("browser_download_url")
        digest = asset.get("digest")
        if not isinstance(download_url, str) or not download_url:
            raise DataReleaseError("Processed-data asset has no download URL")
        if not isinstance(digest, str) or not digest.startswith("sha256:"):
            raise DataReleaseIntegrityError(
                "Processed-data asset metadata has no SHA-256 digest"
            )

        downloaded = self._request(
            "GET",
            download_url,
            operation="processed-data asset download",
            headers=self._headers(accept="application/octet-stream"),
            allow_redirects=True,
        )
        if downloaded.status_code != 200:
            raise _api_error(downloaded, operation="processed-data asset download")
        archive_bytes = downloaded.content
        actual_digest = f"sha256:{hashlib.sha256(archive_bytes).hexdigest()}"
        if actual_digest != digest:
            raise DataReleaseIntegrityError(
                "Downloaded processed-data archive does not match GitHub asset digest"
            )
        return archive_bytes

    def fetch_and_install(self, *, data_dir: Path = DATA_DIR) -> None:
        archive_bytes = self.download()
        install_processed_data_archive(archive_bytes, data_dir=data_dir)


def publish_processed_data_release(
    *,
    data_dir: Path = DATA_DIR,
    client: Optional[GitHubDataReleaseClient] = None,
) -> Mapping[str, Any]:
    """Package the allowlisted data and publish it to the fixed Release."""
    archive_bytes = build_processed_data_archive(data_dir=data_dir)
    token = resolve_publish_token()
    return (client or GitHubDataReleaseClient()).publish(
        archive_bytes,
        token=token,
    )


def fetch_processed_data_once(
    *,
    data_dir: Path = DATA_DIR,
    client: Optional[GitHubDataReleaseClient] = None,
    on_installed: Optional[Callable[[Path], None]] = None,
) -> bool:
    """Serialize fetch attempts and cache only a successful process result."""
    with _fetch_lock:
        if _fetch_state.successful:
            return True
        active_client = client or GitHubDataReleaseClient()
        try:
            active_client.fetch_and_install(data_dir=data_dir)
            if on_installed is not None:
                on_installed(data_dir)
        except DataReleaseNotFound as exc:
            logger.warning("[startup] Processed data Release is not currently available: %s", exc)
            _set_fetch_state(successful=False, error=str(exc))
            return False
        except Exception as exc:
            logger.exception("[startup] Processed-data Release fetch failed: %s", exc)
            _set_fetch_state(successful=False, error=str(exc))
            return False

        logger.info("[startup] Processed data installed from GitHub Release %s", DATA_RELEASE_TAG)
        _set_fetch_state(successful=True)
        return True
