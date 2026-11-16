# Parked debt: this helper is not wired into the restored importer.
# Its season-disjointness gate failed on real inputs in all five leagues.
# See historical-coverage.json in the NEXT-38a-review-split-01 evidence packet.
"""Preflight source coverage before any legacy historical import writes."""
from datetime import datetime
from pathlib import Path
import csv
from .audit import project_metadata


class SourceCoverageError(ValueError):
    pass


def require_disjoint_seasons(paths, league):
    owners = {}
    coverage = {}
    for path in sorted(map(Path, paths)):
        seasons = set()
        with path.open("rb") as stream:
            header_bytes = stream.readline()
            if not header_bytes:
                raise SourceCoverageError(f"Empty historical source: {path}")
            header = next(csv.reader([header_bytes.decode("utf-8-sig")]))
            if header.count("Date") != 1:
                raise SourceCoverageError(f"Date column missing/duplicated: {path}")
            index = header.index("Date")
            for line, raw in enumerate(stream, 2):
                if not raw.strip():
                    continue
                value = project_metadata(raw, {index: "date"}).get("date", "")
                parsed = None
                for fmt in ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d"):
                    try:
                        parsed = datetime.strptime(value.strip(), fmt).date()
                        break
                    except ValueError:
                        continue
                if parsed is None:
                    raise SourceCoverageError(f"Unparseable historical date: {path}:{line}")
                seasons.add(parsed.year if parsed.month >= 6 else parsed.year-1)
        if not seasons:
            raise SourceCoverageError(f"No dated rows in historical source: {path}")
        coverage[str(path)] = sorted(seasons)
        for season in seasons:
            key = (league, season)
            if key in owners:
                raise SourceCoverageError(f"Two files cover {league}/{season}: {owners[key]} and {path}")
            owners[key] = str(path)
    if not coverage:
        raise SourceCoverageError(f"No historical sources for {league}")
    return coverage

