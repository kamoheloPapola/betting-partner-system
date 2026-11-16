"""Validate a pinned historical source inventory and expose approved fixture metadata."""

from collections import defaultdict
import csv
from dataclasses import dataclass
from datetime import date, datetime
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from types import MappingProxyType

from .audit import LEAGUES, MetadataError, project_metadata


AUTHORITATIVE = "AUTHORITATIVE"
SUPERSEDED_IDENTICAL = "SUPERSEDED_IDENTICAL"
NON_AUTHORITATIVE_SNAPSHOT = "NON_AUTHORITATIVE_SNAPSHOT"
UNVERIFIED_PROVIDER = "UNVERIFIED_PROVIDER"
MISSING = "MISSING"
_HASH = re.compile(r"[0-9a-f]{64}")
_CSV_NAME = re.compile(r"[A-Za-z0-9_-]+\.csv", re.IGNORECASE)
_DATE_FORMATS = ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d")


class SourceManifestError(ValueError):
    """The pinned source inventory cannot be trusted or does not grant access."""


@dataclass(frozen=True)
class Authority:
    path: str
    sha256: str
    fixtures: tuple[tuple[str, str, str], ...]


@dataclass(frozen=True)
class SourceManifest:
    _authority: MappingProxyType
    _dispositions: MappingProxyType

    def authoritative(self, league: str, season: int) -> Authority:
        """Return verified metadata for an authoritative league-season only."""
        key = (league, season)
        if key not in self._authority:
            disposition = self._dispositions.get(key, "OUTSIDE_DECLARED_SCOPE")
            raise SourceManifestError(f"{league}/{season}: no authoritative source ({disposition})")
        return self._authority[key]


def _path_stat(path):
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
        raise SourceManifestError(f"Symlink or reparse point refused: {path}")
    return info


def _checked_stat(path):
    for parent in reversed(path.parents):
        _path_stat(parent)
    return _path_stat(path)


def _signature(info):
    # Windows path and descriptor stats can report different ctime semantics.
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def _held_bytes(path, expected_sha256):
    """Hash and retain the bytes to be parsed, avoiding a second open."""
    try:
        original = _checked_stat(path)
        if not stat.S_ISREG(original.st_mode):
            raise SourceManifestError(f"Not a regular file: {path}")
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            if _signature(original) != _signature(before):
                raise SourceManifestError(f"File changed before held read: {path}")
            raw = stream.read()
            after = os.fstat(stream.fileno())
            current = _checked_stat(path)
            if (_signature(before) != _signature(after) or _signature(after) != _signature(current)
                    or len(raw) != before.st_size):
                raise SourceManifestError(f"File changed during held read: {path}")
    except OSError as exc:
        raise SourceManifestError(f"Cannot read pinned file: {path}") from exc
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise SourceManifestError(f"SHA-256 mismatch: {path}")
    return raw


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise SourceManifestError(f"Duplicate manifest field: {key}")
        result[key] = value
    return result


def _year(value):
    if type(value) is not int or not 1900 <= value <= 9998:
        raise SourceManifestError("Season years must be integers between 1900 and 9998")
    return value


def _sha(value):
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise SourceManifestError("Expected a lowercase SHA-256 digest")
    return value


def _iso_date(value):
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise SourceManifestError("Manifest date must use YYYY-MM-DD")
    return date.fromisoformat(value)


def _historical_entry(entry):
    league = entry["league"]
    path = entry["path"]
    if not isinstance(path, str) or league not in LEAGUES:
        raise SourceManifestError("Invalid historical entry path or league")
    parts = PurePosixPath(path).parts
    if (len(parts) != 4 or parts[:3] != ("data", "historical", league)
            or "/".join(parts) != path or not _CSV_NAME.fullmatch(parts[-1])):
        raise SourceManifestError(f"Historical entry is outside its league CSV directory: {path}")
    _sha(entry["sha256"])
    season = _year(entry["declared_season_start"])
    _year(entry["filename_season"])
    if type(entry["row_count"]) is not int or entry["row_count"] <= 0:
        raise SourceManifestError(f"Invalid manifest row count: {path}")
    if _iso_date(entry["min_date"]) > _iso_date(entry["max_date"]):
        raise SourceManifestError(f"Invalid manifest date range: {path}")
    role = entry["role"]
    if role not in (AUTHORITATIVE, SUPERSEDED_IDENTICAL, NON_AUTHORITATIVE_SNAPSHOT, UNVERIFIED_PROVIDER):
        raise SourceManifestError(f"Invalid source role: {path}")
    if role == SUPERSEDED_IDENTICAL:
        if _sha(entry["identical_to"]) != entry["sha256"]:
            raise SourceManifestError(f"SUPERSEDED_IDENTICAL hash differs from identical_to: {path}")
    elif "identical_to" in entry:
        raise SourceManifestError(f"Unexpected identical_to: {path}")
    return league, season


def _inventory(root):
    found = set()
    for league in LEAGUES:
        directory = root / "data" / "historical" / league
        if not stat.S_ISDIR(_checked_stat(directory).st_mode):
            raise SourceManifestError(f"Not a league directory: {directory}")
        for path in directory.iterdir():
            if path.suffix.lower() == ".csv":
                if not stat.S_ISREG(_path_stat(path).st_mode):
                    raise SourceManifestError(f"Not a regular league CSV: {path}")
                found.add(path.relative_to(root).as_posix())
    return found


def _manifest_entries(document):
    if type(document["schema_version"]) is not int or document["schema_version"] != 1:
        raise SourceManifestError("Unsupported source manifest schema")
    bounds = document["season_scope"]
    if set(bounds) != set(LEAGUES):
        raise SourceManifestError("A season scope is required for each league")
    scope = set()
    for league, years in bounds.items():
        first, last = _year(years["first"]), _year(years["last"])
        if first > last:
            raise SourceManifestError(f"Inverted season scope: {league}")
        scope.update((league, year) for year in range(first, last + 1))

    entries = {}
    authorities = {}
    dispositions = {}
    hashes = defaultdict(list)

    def disposition(key, role):
        if key not in scope:
            raise SourceManifestError(f"Disposition outside declared scope: {key}")
        if key in dispositions:
            raise SourceManifestError(f"Multiple dispositions for {key}: {dispositions[key]} and {role}")
        dispositions[key] = role

    for entry in document["files"]:
        key = _historical_entry(entry)
        path, role = entry["path"], entry["role"]
        if key not in scope or path in entries:
            raise SourceManifestError(f"Duplicate path or file outside season scope: {path}")
        entries[path] = entry
        hashes[entry["sha256"]].append(entry)
        if role in (AUTHORITATIVE, UNVERIFIED_PROVIDER):
            disposition(key, role)
        if role == AUTHORITATIVE:
            authorities[key] = entry

    for section, role in (("known_gaps", MISSING), ("unverified_seasons", UNVERIFIED_PROVIDER)):
        for entry in document[section]:
            if entry["status"] != role:
                raise SourceManifestError(f"Invalid disposition in {section}")
            disposition((entry["league"], _year(entry["declared_season_start"])), role)
    if scope - dispositions.keys():
        raise SourceManifestError(f"Missing league-season dispositions: {sorted(scope - dispositions.keys())}")

    for group in hashes.values():
        if len(group) == 1 and group[0]["role"] != SUPERSEDED_IDENTICAL:
            continue
        authority = [entry for entry in group if entry["role"] == AUTHORITATIVE]
        copies = [entry for entry in group if entry["role"] == SUPERSEDED_IDENTICAL]
        if len(authority) != 1 or len(copies) != len(group) - 1:
            raise SourceManifestError("Equal hashes must have one authoritative file and explicitly superseded copies")
        target = authority[0]
        for copy in copies:
            if (copy["league"], copy["declared_season_start"]) != (target["league"], target["declared_season_start"]):
                raise SourceManifestError("SUPERSEDED_IDENTICAL copy must share the authority's league-season")
    return entries, authorities, dispositions


def _metadata(path, entry, raw):
    """Project only date and team cells from the already hash-verified bytes."""
    stream = io.BytesIO(raw)
    try:
        header = next(csv.reader([stream.readline().decode("utf-8-sig")], strict=True))
        required = {"Date": "date", "HomeTeam": "home", "AwayTeam": "away"}
        if any(header.count(column) != 1 for column in required):
            raise ValueError("Missing or duplicate metadata columns")
        columns = {header.index(column): name for column, name in required.items()}
    except (ValueError, UnicodeError, StopIteration, csv.Error) as exc:
        raise SourceManifestError(f"Invalid historical CSV header: {path}") from exc
    rows = []
    for line, record in enumerate(stream, 2):
        if not record.strip():
            continue
        try:
            fields = project_metadata(record, columns)
            text = fields["date"].strip()
            for fmt in _DATE_FORMATS:
                try:
                    day = datetime.strptime(text, fmt).date()
                    break
                except ValueError:
                    continue
            else:
                raise ValueError("Unsupported date")
            home, away = fields["home"], fields["away"]
            if not home.strip() or not away.strip():
                raise ValueError("Empty team label")
        except (ValueError, KeyError, UnicodeError, MetadataError) as exc:
            raise SourceManifestError(f"Invalid historical metadata: {path}:{line}") from None
        rows.append((day.isoformat(), home, away))

    dates = [day for day, _, _ in rows]
    if (not dates or len(dates) != entry["row_count"] or min(dates) != entry["min_date"]
            or max(dates) != entry["max_date"]):
        raise SourceManifestError(f"Historical row count or date range differs from manifest: {path}")
    return tuple(rows)


def load_source_manifest(root: Path, manifest_path: Path, expected_manifest_sha256: str) -> SourceManifest:
    """Verify the manifest, then all inventoried CSV bytes, before parsing CSV metadata."""
    root = Path(os.path.abspath(root))
    manifest_path = Path(manifest_path)
    manifest_path = Path(os.path.abspath(root / manifest_path))
    raw_manifest = _held_bytes(manifest_path, _sha(expected_manifest_sha256))
    try:
        document = json.loads(raw_manifest, object_pairs_hook=_json_object)
        entries, authorities, dispositions = _manifest_entries(document)
    except SourceManifestError:
        raise
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise SourceManifestError("Invalid source manifest structure") from exc

    try:
        actual = _inventory(root)
    except OSError as exc:
        raise SourceManifestError("Cannot enumerate historical league CSVs") from exc
    missing, unexpected = entries.keys() - actual, actual - entries.keys()
    if missing or unexpected:
        raise SourceManifestError(f"Historical inventory mismatch. Missing: {sorted(missing)}; unlisted: {sorted(unexpected)}")
    snapshots = {path: _held_bytes(root / path, entry["sha256"]) for path, entry in entries.items()}
    verified = {path: _metadata(path, entries[path], raw) for path, raw in snapshots.items()}
    seen = {}
    allowed = {}
    for (league, season), entry in authorities.items():
        path = entry["path"]
        start, end = date(season, 7, 1).isoformat(), date(season + 1, 8, 31).isoformat()
        for key in verified[path]:
            if not start <= key[0] <= end:
                raise SourceManifestError(f"Authoritative row outside declared season {league}/{season}: {path} ({key[0]})")
            previous = seen.setdefault((league, key), path)
            if previous != path:
                raise SourceManifestError(f"Exact fixture key shared by authoritative files: {previous} and {path}")
        allowed[(league, season)] = Authority(path, entry["sha256"], verified[path])
    return SourceManifest(MappingProxyType(allowed), MappingProxyType(dispositions))
