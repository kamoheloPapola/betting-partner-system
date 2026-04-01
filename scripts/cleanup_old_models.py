from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Optional

from src.ml.model_db import ModelHistoryDB


MODEL_SUFFIXES = (".pkl", ".joblib")
LEAGUE_CODES = {"BL1", "FL1", "PD", "PL", "SA", "Global"}
TIMESTAMP_VERSION_PATTERN = re.compile(r"^\d{8}_\d{6}$")
SEMVER_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")
NAME_VERSION_PATTERN = re.compile(r"^(?P<name>.+?)_v(?P<version>\d{8}_\d{6}|\d+\.\d+\.\d+|\d+)$")


@dataclass(frozen=True)
class ManifestArtifact:
    manifest_key: str
    rel_path: str
    name: str
    league: str
    version: str
    status: str
    registered_at: Optional[str]
    trained_at: Optional[str]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class CleanupResult:
    manifest_path: Path
    model_root: Path
    keep_versions: int
    referenced_files: list[str]
    protected_files: list[str]
    retained_recent_files: list[str]
    kept_files: list[str]
    referenced_missing_files: list[str]
    deletion_candidates: list[str]
    total_candidate_bytes: int
    deleted_files: list[str]
    deleted_bytes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delete old model artifacts while preserving productive, shadow, and recent versions."
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Delete the identified old files after listing them.",
    )
    parser.add_argument(
        "--keep-versions",
        type=int,
        default=3,
        help="Number of recent versions to keep per model name + league combination.",
    )
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def models_dir(root: Path) -> Path:
    return root / "src" / "ml" / "models"


def manifest_path(model_root: Path) -> Path:
    return model_root / "manifest.json"


def normalize_model_path(raw_path: str, model_root: Path) -> str:
    candidate = raw_path.replace("\\", "/")

    if candidate.startswith("./"):
        candidate = candidate[2:]
    if candidate.startswith("src/ml/models/"):
        candidate = candidate[len("src/ml/models/") :]

    raw_fs_path = Path(raw_path)
    if raw_fs_path.is_absolute():
        try:
            return raw_fs_path.resolve().relative_to(model_root.resolve()).as_posix()
        except ValueError:
            return PurePosixPath(candidate).as_posix()

    return PurePosixPath(candidate).as_posix()


def collect_referenced_models(node: Any, model_root: Path) -> set[str]:
    referenced: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for nested in value.values():
                walk(nested)
            return

        if isinstance(value, list):
            for nested in value:
                walk(nested)
            return

        if isinstance(value, str) and value.lower().endswith(MODEL_SUFFIXES):
            referenced.add(normalize_model_path(value, model_root))

    walk(node)
    return referenced


def list_model_files(model_root: Path) -> list[Path]:
    return sorted(
        path
        for path in model_root.rglob("*")
        if path.is_file() and path.suffix.lower() in MODEL_SUFFIXES
    )


def _parse_datetime(value: Any) -> Optional[datetime]:
    if not value or not isinstance(value, str):
        return None

    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _parse_semver(version: str) -> Optional[tuple[int, int, int]]:
    if not SEMVER_PATTERN.fullmatch(version):
        return None
    return tuple(int(part) for part in version.split("."))  # type: ignore[return-value]


def _parse_timestamp_version(version: str) -> Optional[datetime]:
    if not TIMESTAMP_VERSION_PATTERN.fullmatch(version):
        return None
    try:
        return datetime.strptime(version, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _version_sort_key(version: str) -> tuple[int, Any]:
    timestamp_version = _parse_timestamp_version(version)
    if timestamp_version is not None:
        return (2, timestamp_version)

    semver = _parse_semver(version)
    if semver is not None:
        return (1, semver)

    return (0, version)


def _artifact_sort_key(artifact: ManifestArtifact) -> tuple[int, Any, tuple[int, Any], str]:
    ts = _parse_datetime(artifact.registered_at) or _parse_datetime(artifact.trained_at)
    if ts is not None:
        return (1, ts, _version_sort_key(artifact.version), artifact.rel_path)
    return (0, datetime.min.replace(tzinfo=timezone.utc), _version_sort_key(artifact.version), artifact.rel_path)


def _iter_manifest_artifacts(data: dict[str, Any], model_root: Path) -> list[ManifestArtifact]:
    artifacts: list[ManifestArtifact] = []

    for manifest_key, value in data.items():
        if manifest_key in {"active_models", "shadow_models"}:
            continue
        if not isinstance(value, dict):
            continue

        filename = value.get("filename")
        if not isinstance(filename, str) or not filename.lower().endswith(MODEL_SUFFIXES):
            continue

        rel_path = normalize_model_path(filename, model_root)
        artifacts.append(
            ManifestArtifact(
                manifest_key=str(manifest_key),
                rel_path=rel_path,
                name=str(value.get("name") or Path(rel_path).stem),
                league=str(value.get("league") or "Global"),
                version=str(value.get("version") or Path(rel_path).stem),
                status=str(value.get("status") or ""),
                registered_at=value.get("registered_at") if isinstance(value.get("registered_at"), str) else None,
                trained_at=value.get("trained_at") if isinstance(value.get("trained_at"), str) else None,
                metadata=value,
            )
        )

    return artifacts


def _collect_protected_manifest_keys(data: dict[str, Any]) -> set[str]:
    protected: set[str] = set()

    for manifest_key, value in data.items():
        if manifest_key in {"active_models", "shadow_models"}:
            continue
        if isinstance(value, dict) and value.get("status") == "productive":
            protected.add(str(manifest_key))

    active_models = data.get("active_models", {})
    if isinstance(active_models, dict):
        for value in active_models.values():
            if isinstance(value, str):
                protected.add(value)

    shadow_models = data.get("shadow_models", {})
    if isinstance(shadow_models, dict):
        for value in shadow_models.values():
            if isinstance(value, str):
                protected.add(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        protected.add(item)

    return protected


def _safe_file_size(model_root: Path, rel_path: str) -> int:
    path = model_root / rel_path
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def _format_rel_paths(paths: Iterable[str]) -> list[str]:
    return sorted(set(paths))


def build_cleanup_result(
    *,
    model_root: Path,
    keep_versions: int,
) -> tuple[CleanupResult, dict[str, ManifestArtifact]]:
    manifest = manifest_path(model_root)
    data = json.loads(manifest.read_text(encoding="utf-8"))

    referenced_paths = _format_rel_paths(collect_referenced_models(data, model_root))
    artifacts = _iter_manifest_artifacts(data, model_root)
    artifact_by_key = {artifact.manifest_key: artifact for artifact in artifacts}
    artifact_by_path = {artifact.rel_path: artifact for artifact in artifacts}

    protected_keys = _collect_protected_manifest_keys(data)
    protected_files = _format_rel_paths(
        artifact_by_key[key].rel_path
        for key in protected_keys
        if key in artifact_by_key
    )

    on_disk_paths = _format_rel_paths(
        path.relative_to(model_root).as_posix()
        for path in list_model_files(model_root)
    )
    on_disk_set = set(on_disk_paths)

    retained_recent: set[str] = set()
    grouped: dict[tuple[str, str], list[ManifestArtifact]] = {}
    for artifact in artifacts:
        if artifact.rel_path not in on_disk_set:
            continue
        grouped.setdefault((artifact.name, artifact.league), []).append(artifact)

    for scope_artifacts in grouped.values():
        ordered = sorted(scope_artifacts, key=_artifact_sort_key, reverse=True)
        retained_recent.update(artifact.rel_path for artifact in ordered[:keep_versions])

    retained_recent_files = _format_rel_paths(retained_recent)
    kept_files = _format_rel_paths(set(protected_files) | retained_recent)
    referenced_missing_files = _format_rel_paths(set(referenced_paths) - on_disk_set)
    deletion_candidates = _format_rel_paths(on_disk_set - set(kept_files))
    total_candidate_bytes = sum(_safe_file_size(model_root, rel_path) for rel_path in deletion_candidates)

    result = CleanupResult(
        manifest_path=manifest,
        model_root=model_root,
        keep_versions=keep_versions,
        referenced_files=referenced_paths,
        protected_files=protected_files,
        retained_recent_files=retained_recent_files,
        kept_files=kept_files,
        referenced_missing_files=referenced_missing_files,
        deletion_candidates=deletion_candidates,
        total_candidate_bytes=total_candidate_bytes,
        deleted_files=[],
        deleted_bytes=0,
    )
    return result, artifact_by_path


def _infer_cleanup_identity(rel_path: str, artifact: Optional[ManifestArtifact]) -> tuple[str, str, str]:
    if artifact is not None:
        return artifact.name, artifact.league or "Global", artifact.version

    path = PurePosixPath(rel_path)
    parts = path.parts
    filename_stem = path.stem
    league = "Global"

    if len(parts) >= 2 and parts[1] in LEAGUE_CODES:
        league = parts[1]

    for candidate in sorted(LEAGUE_CODES - {"Global"}, key=len, reverse=True):
        suffix = f"_{candidate}"
        if filename_stem.endswith(suffix):
            league = candidate
            filename_stem = filename_stem[: -len(suffix)]
            break

    match = NAME_VERSION_PATTERN.match(filename_stem)
    if match:
        return match.group("name"), league, match.group("version")

    return filename_stem, league, "unknown"


def _log_cleanup_event(
    history_db: ModelHistoryDB,
    *,
    rel_path: str,
    artifact: Optional[ManifestArtifact],
) -> None:
    model_name, league, version = _infer_cleanup_identity(rel_path, artifact)
    history_db.write_event(
        model_name=model_name,
        league=league or "Global",
        version=version,
        brier_score=None,
        ece=None,
        train_size=None,
        event_type="cleanup",
    )


def cleanup_models(
    *,
    root: Optional[Path] = None,
    model_root: Optional[Path] = None,
    keep_versions: int = 3,
    confirm: bool = False,
    history_db_path: Optional[Path] = None,
) -> CleanupResult:
    if keep_versions < 0:
        raise ValueError("--keep-versions must be zero or greater.")

    resolved_root = root or project_root()
    resolved_model_root = model_root or models_dir(resolved_root)
    result, artifact_by_path = build_cleanup_result(
        model_root=resolved_model_root,
        keep_versions=keep_versions,
    )

    if not confirm or not result.deletion_candidates:
        return result

    history_db = ModelHistoryDB(db_path=history_db_path) if history_db_path else ModelHistoryDB()
    deleted_files: list[str] = []
    deleted_bytes = 0

    for rel_path in result.deletion_candidates:
        target = resolved_model_root / rel_path
        file_size = _safe_file_size(resolved_model_root, rel_path)
        target.unlink()
        _log_cleanup_event(
            history_db,
            rel_path=rel_path,
            artifact=artifact_by_path.get(rel_path),
        )
        deleted_files.append(rel_path)
        deleted_bytes += file_size

    return CleanupResult(
        manifest_path=result.manifest_path,
        model_root=result.model_root,
        keep_versions=result.keep_versions,
        referenced_files=result.referenced_files,
        protected_files=result.protected_files,
        retained_recent_files=result.retained_recent_files,
        kept_files=result.kept_files,
        referenced_missing_files=result.referenced_missing_files,
        deletion_candidates=result.deletion_candidates,
        total_candidate_bytes=result.total_candidate_bytes,
        deleted_files=deleted_files,
        deleted_bytes=deleted_bytes,
    )


def format_size(num_bytes: int) -> str:
    value = float(num_bytes)
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{num_bytes} B"


def print_section(title: str, items: list[str]) -> None:
    print(f"\n{title} ({len(items)}):")
    if not items:
        print("  - none -")
        return

    for item in items:
        print(f"  {item}")


def print_summary(result: CleanupResult) -> None:
    print(f"Manifest: {result.manifest_path}")
    print(f"Model directory: {result.model_root}")
    print(f"Keep versions per model scope: {result.keep_versions}")
    print(f"Referenced model paths in manifest: {len(result.referenced_files)}")
    print(f"Protected files on disk (productive/active/shadow): {len(result.protected_files)}")
    print(f"Recent version files retained: {len(result.retained_recent_files)}")
    print(f"Files kept on disk after retention rules: {len(result.kept_files)}")
    print(f"Referenced files missing on disk: {len(result.referenced_missing_files)}")
    print(f"Deletion candidates on disk: {len(result.deletion_candidates)}")
    print(
        "Total size of deletion candidates: "
        f"{result.total_candidate_bytes} bytes ({format_size(result.total_candidate_bytes)})"
    )

    print_section("Protected files", result.protected_files)
    print_section("Recent versions retained", result.retained_recent_files)
    print_section("Referenced but missing", result.referenced_missing_files)
    print_section("Deletion candidates", result.deletion_candidates)


def main() -> int:
    args = parse_args()
    result = cleanup_models(
        keep_versions=int(args.keep_versions),
        confirm=bool(args.confirm),
    )
    print_summary(result)

    if not args.confirm:
        print("\nDry run only. Re-run with --confirm to delete the listed candidates.")
        return 0

    if not result.deleted_files:
        print("\nNothing to delete.")
        return 0

    print("\nDeleted files:")
    for rel_path in result.deleted_files:
        print(f"  deleted {rel_path}")

    print(
        f"\nDeleted {len(result.deleted_files)} files, "
        f"freeing {result.deleted_bytes} bytes ({format_size(result.deleted_bytes)})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
