import hashlib
import json
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

from src.data_foundation import source_manifest as sources


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def inventory(tmp_path):
    for league in sources.LEAGUES:
        (tmp_path / "data" / "historical" / league).mkdir(parents=True)
    document = {
        "schema_version": 1,
        "season_scope": {league: {"first": 2024, "last": 2024} for league in sources.LEAGUES},
        "files": [],
        "known_gaps": [],
        "unverified_seasons": [
            {"league": league, "declared_season_start": 2024, "status": sources.UNVERIFIED_PROVIDER}
            for league in sources.LEAGUES if league != "PL"
        ],
    }

    def add(name="PL_2425.csv", season=2024, role=sources.AUTHORITATIVE,
            raw=b"Date,HomeTeam,AwayTeam,FTHG\n2024-09-01,A,B,1\n",
            first="2024-09-01", last="2024-09-01", rows=1):
        path = tmp_path / "data" / "historical" / "PL" / name
        path.write_bytes(raw)
        entry = {
            "league": "PL", "path": path.relative_to(tmp_path).as_posix(), "sha256": digest(raw),
            "filename_season": season, "declared_season_start": season, "role": role,
            "row_count": rows, "min_date": first, "max_date": last,
        }
        document["files"].append(entry)
        return path, entry

    def load(expected=None):
        pin = json.dumps(document).encode()
        (tmp_path / "manifest.json").write_bytes(pin)
        return sources.load_source_manifest(tmp_path, Path("manifest.json"), expected or digest(pin))

    return SimpleNamespace(root=tmp_path, doc=document, add=add, load=load)


def test_projection_accepts_bom_quotes_dates_and_never_decodes_poison(inventory):
    raw = (b'\xef\xbb\xbfDate,HomeTeam,AwayTeam,FTHG,Odds\n'
           b'01/09/24,"A, ""Club""",B,\xff,"\xff,\xfe"\n'
           b'\n02/09/2024,C,D,\xff,\xfe\n2024-09-03,E,F,\xff,\xfe\n')
    inventory.add(raw=raw, last="2024-09-03", rows=3)
    result = inventory.load()
    assert result.authoritative("PL", 2024).fixtures == (
        ("2024-09-01", 'A, "Club"', "B"),
        ("2024-09-02", "C", "D"),
        ("2024-09-03", "E", "F"),
    )
    with pytest.raises(sources.SourceManifestError, match="UNVERIFIED_PROVIDER"):
        result.authoritative("BL1", 2024)


def test_expected_manifest_digest_is_required(inventory):
    inventory.add()
    with pytest.raises(TypeError):
        sources.load_source_manifest(inventory.root, Path("manifest.json"))
    with pytest.raises(sources.SourceManifestError, match="SHA-256 mismatch"):
        inventory.load("0" * 64)


@pytest.mark.parametrize("change", ["missing", "unlisted", "changed"])
def test_inventory_and_file_hash_fail_before_csv_parsing(inventory, monkeypatch, change):
    inventory.add()
    changed, _ = inventory.add("snapshot.csv", role=sources.NON_AUTHORITATIVE_SNAPSHOT,
                              raw=b"Date,HomeTeam,AwayTeam\n2024-09-01,C,D\n")
    if change == "missing":
        changed.unlink()
    elif change == "unlisted":
        changed.with_name("extra.CSV").write_bytes(b"unlisted")
    else:
        changed.write_bytes(b"corrupt")

    def unexpected_parse(*args):
        pytest.fail("CSV parse started before every historical hash was verified")

    monkeypatch.setattr(sources, "_metadata", unexpected_parse)
    with pytest.raises(sources.SourceManifestError, match="inventory mismatch|SHA-256 mismatch"):
        inventory.load()


def test_identical_files_require_a_declared_copy(inventory):
    _, authority = inventory.add()
    path, duplicate = inventory.add("copy.csv", role=sources.NON_AUTHORITATIVE_SNAPSHOT)
    with pytest.raises(sources.SourceManifestError, match="Equal hashes"):
        inventory.load()
    duplicate.update(role=sources.SUPERSEDED_IDENTICAL, identical_to=authority["sha256"])
    assert inventory.load().authoritative("PL", 2024).path == authority["path"]
    duplicate["identical_to"] = "0" * 64
    with pytest.raises(sources.SourceManifestError, match="identical_to"):
        inventory.load()
    path.write_bytes(b"changed")
    duplicate["identical_to"] = authority["sha256"]
    with pytest.raises(sources.SourceManifestError, match="SHA-256 mismatch"):
        inventory.load()


def test_copy_must_reference_an_authoritative_entry(inventory):
    inventory.add()
    _, copy = inventory.add("copy.csv", role=sources.SUPERSEDED_IDENTICAL,
                            raw=b"Date,HomeTeam,AwayTeam\n2024-09-01,C,D\n")
    copy["identical_to"] = copy["sha256"]
    with pytest.raises(sources.SourceManifestError, match="Equal hashes"):
        inventory.load()


def test_another_authority_for_same_season_is_refused(inventory):
    inventory.add()
    inventory.add("other.csv", raw=b"Date,HomeTeam,AwayTeam\n2024-09-01,C,D\n")
    with pytest.raises(sources.SourceManifestError, match="Multiple dispositions"):
        inventory.load()


def test_a_snapshot_does_not_dispose_of_its_season(inventory):
    inventory.add(role=sources.NON_AUTHORITATIVE_SNAPSHOT)
    with pytest.raises(sources.SourceManifestError, match="Missing league-season dispositions"):
        inventory.load()
    inventory.doc["known_gaps"].append(
        {"league": "PL", "declared_season_start": 2024, "status": sources.MISSING}
    )
    with pytest.raises(sources.SourceManifestError, match="MISSING"):
        inventory.load().authoritative("PL", 2024)


def test_shared_fixture_across_declared_seasons_is_refused(inventory):
    inventory.doc["season_scope"]["PL"]["last"] = 2025
    inventory.add(raw=b"Date,HomeTeam,AwayTeam\n2025-07-20,A,B\n",
                  first="2025-07-20", last="2025-07-20")
    inventory.add("PL_2526.csv", season=2025, raw=b"Date,HomeTeam,AwayTeam\n20/07/2025,A,B\n",
                  first="2025-07-20", last="2025-07-20")
    with pytest.raises(sources.SourceManifestError, match="Exact fixture key shared"):
        inventory.load()


def test_explicit_season_preserves_spill_and_rejects_mislabel(inventory):
    _, entry = inventory.add(raw=b"Date,HomeTeam,AwayTeam\n02/06/2025,A,B\n2025-08-31,C,D\n",
                             first="2025-06-02", last="2025-08-31", rows=2)
    entry["filename_season"] = 2025
    assert len(inventory.load().authoritative("PL", 2024).fixtures) == 2
    entry["declared_season_start"] = 2025
    inventory.doc["season_scope"]["PL"] = {"first": 2025, "last": 2025}
    with pytest.raises(sources.SourceManifestError, match="outside declared season"):
        inventory.load()


def test_reparse_point_fails_even_if_the_csv_hash_would_match(inventory, monkeypatch):
    path, _ = inventory.add()
    lstat = Path.lstat

    def flagged(candidate):
        if candidate == path:
            return SimpleNamespace(st_mode=stat.S_IFREG, st_file_attributes=stat.FILE_ATTRIBUTE_REPARSE_POINT)
        return lstat(candidate)

    monkeypatch.setattr(Path, "lstat", flagged)
    with pytest.raises(sources.SourceManifestError, match="reparse point"):
        inventory.load()


def test_symlink_in_ancestor_is_refused(inventory, monkeypatch):
    path, _ = inventory.add()
    lstat = Path.lstat

    def flagged(candidate):
        if candidate == path.parent:
            return SimpleNamespace(st_mode=stat.S_IFLNK)
        return lstat(candidate)

    monkeypatch.setattr(Path, "lstat", flagged)
    with pytest.raises(sources.SourceManifestError, match="Symlink"):
        inventory.load()
