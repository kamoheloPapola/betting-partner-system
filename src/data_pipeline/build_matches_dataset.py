"""
Canonical Dataset Builder — Phase 1.

Reads raw Football-Data.co.uk CSV files directly from data/historical/,
strips all odds/bookmaker columns, normalises team names, and produces
a single clean matches.csv with a coverage metadata file.

Schema:
    match_hash, match_date, season, league,
    home_team, away_team, 
    home_goals, away_goals, home_goals_ht, away_goals_ht,
    home_shots, away_shots, home_shots_on_target, away_shots_on_target,
    home_corners, away_corners, home_fouls, away_fouls,
    home_yellow_cards, away_yellow_cards, home_red_cards, away_red_cards,
    home_cards, away_cards, home_total_shots, away_total_shots, referee

No odds. No bookmaker columns. No derived fields.
Feature engineering produces everything else from this source.

Usage:
    python -m src.cli.main build-canonical
"""
import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd

from src.config import DATA_DIR

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────
SCHEMA_VERSION = "2.0.0"

# Raw Div code → standard league code
DIV_MAP: Dict[str, str] = {
    "E0": "PL",
    "D1": "BL1",
    "F1": "FL1",
    "I1": "SA",
    "SP1": "PD",
}

VALID_LEAGUES: Set[str] = set(DIV_MAP.values())

# Source directories to scan
SOURCE_DIRS = [
    DATA_DIR / "historical",
    DATA_DIR / "raw",
]

# Columns to keep from raw CSVs → renamed
COL_MAP: Dict[str, str] = {
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG":     "home_goals",
    "FTAG":     "away_goals",
    "HTHG":     "home_goals_ht",
    "HTAG":     "away_goals_ht",
    "HS":       "home_shots",
    "AS":       "away_shots",
    "HST":      "home_shots_on_target",
    "AST":      "away_shots_on_target",
    "HC":       "home_corners",
    "AC":       "away_corners",
    "HF":       "home_fouls",
    "AF":       "away_fouls",
    "HY":       "home_yellow_cards",
    "AY":       "away_yellow_cards",
    "HR":       "home_red_cards",
    "AR":       "away_red_cards",
    "Referee":  "referee",
}

# Required raw columns — file is rejected without these
REQUIRED_RAW = {"HomeTeam", "AwayTeam", "FTHG", "FTAG", "Date"}

# Odds / bookmaker column patterns to DROP
ODDS_PATTERNS = re.compile(
    r"^(B365|BW|IW|PS|WH|VC|Bb|Avg|Max|closing|open|"
    r"BF|PSCH|PSCD|PSCA|LB|SB|GB|BS|SJ|"
    r"1X|AH|PCAH|MaxCAH|AvgCAH|BFECAH|"
    r"Over|Under|BbOU|BbMx|BbAv|BbAH)",
    re.IGNORECASE,
)

# Nullable signal columns (fill with NaN if absent — never drop rows)
NULLABLE_SIGNALS = [
    "home_goals_ht", "away_goals_ht",
    "home_shots", "away_shots",
    "home_shots_on_target", "away_shots_on_target",
    "home_corners", "away_corners",
    "home_fouls", "away_fouls",
    "home_yellow_cards", "away_yellow_cards",
    "home_red_cards", "away_red_cards",
    "referee",
]

# Output paths
OUTPUT_DIR = DATA_DIR / "processed"


class CanonicalDatasetBuilder:
    """
    Discovers, validates, normalises, and merges all raw CSV files into
    a single clean matches.csv with no odds and a consistent schema.
    """

    def __init__(
        self,
        source_dirs: Optional[List[Path]] = None,
        output_dir: Optional[Path] = None,
    ) -> None:
        self.source_dirs = source_dirs or SOURCE_DIRS
        self.output_dir = output_dir or OUTPUT_DIR

    # ==================================================================
    # Public API
    # ==================================================================
    def build(self) -> Dict[str, Any]:
        """
        Execute the full canonical build pipeline.

        Returns:
            Summary dict with counts, leagues, seasons, and output paths.
        """
        logger.info("═══ Canonical Dataset Build — START ═══")

        # Step 1: Discover
        csv_files = self._discover_files()
        if not csv_files:
            return {"status": "empty", "message": "No raw CSV files found"}

        # Step 2: Parse & normalise
        frames: List[pd.DataFrame] = []
        skipped: List[str] = []
        column_coverage: Dict[str, Dict[str, bool]] = {}

        for path, league_dir in csv_files:
            df, coverage = self._parse_and_normalise(path, league_dir)
            if df is not None:
                frames.append(df)
                column_coverage[path.name] = coverage
            else:
                skipped.append(path.name)

        if not frames:
            return {
                "status": "error",
                "message": "All files failed validation",
                "skipped": skipped,
            }

        # Step 3: Concatenate, deduplicate, sort
        combined = pd.concat(frames, ignore_index=True)

        pre_dedup = len(combined)
        combined = combined.drop_duplicates(subset=["match_hash"], keep="first")
        dupes_removed = pre_dedup - len(combined)
        if dupes_removed:
            logger.info("Removed %d duplicate matches (by match_hash)", dupes_removed)

        # Sort chronologically within league
        combined = combined.sort_values(
            ["league", "season", "match_date"]
        ).reset_index(drop=True)

        # Assert no future data
        today = pd.Timestamp.now().normalize()
        future_mask = combined["match_date"] > today
        future_count = future_mask.sum()
        if future_count:
            logger.warning("Dropping %d future-dated matches (> %s)", future_count, today.date())
            combined = combined[~future_mask].reset_index(drop=True)

        # Step 4: Schema validation
        self._validate_schema(combined)

        # Step 5: Write output
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Output 1: matches.csv
        matches_path = self.output_dir / "matches.csv"
        combined.to_csv(matches_path, index=False)
        logger.info("Wrote %d matches to %s", len(combined), matches_path)

        # Output 2: matches_hash.json (Freeze Hash)
        h = hashlib.sha256()
        with open(matches_path, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        
        matches_hash_data = {
            "sha256": h.hexdigest(),
            "rows": len(combined),
            "columns": len(combined.columns),
            "feature_count": len(combined.columns),
            "min_date": str(combined["match_date"].min().date()),
            "max_date": str(combined["match_date"].max().date()),
            "leagues": sorted(combined["league"].unique().tolist()),
            "seasons": sorted(combined["season"].unique().tolist()),
            "missing_values_total": int(combined.isna().sum().sum()),
        }
        hash_path = self.output_dir / "matches_hash.json"
        with open(hash_path, "w", encoding="utf-8") as f:
            json.dump(matches_hash_data, f, indent=2)

        # Output 3: data_profile.json (Data Drift Profiling)
        total_matches = len(combined)
        hm_wins = (combined["home_goals"] > combined["away_goals"]).sum()
        draws = (combined["home_goals"] == combined["away_goals"]).sum()
        aw_wins = (combined["home_goals"] < combined["away_goals"]).sum()

        profile_data = {
            "avg_goals_per_match": float((combined["home_goals"].sum() + combined["away_goals"].sum()) / total_matches),
            "avg_corners_per_match": float((combined["home_corners"].sum() + combined["away_corners"].sum()) / total_matches) if "home_corners" in combined.columns else None,
            "avg_cards_per_match": float((combined["home_cards"].sum() + combined["away_cards"].sum()) / total_matches) if "home_cards" in combined.columns else None,
            "avg_shots_per_match": float((combined["home_total_shots"].sum() + combined["away_total_shots"].sum()) / total_matches) if "home_total_shots" in combined.columns else None,
            "home_win_rate": float(hm_wins / total_matches),
            "draw_rate": float(draws / total_matches),
            "away_win_rate": float(aw_wins / total_matches),
        }
        profile_path = self.output_dir / "data_profile.json"
        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump(profile_data, f, indent=2)

        # Verify zero odds columns
        odds_leak = [c for c in combined.columns if ODDS_PATTERNS.match(c)]
        if odds_leak:
            logger.error("ODDS LEAK DETECTED: %s", odds_leak)

        # Output 4: matches_coverage.json (Coverage metadata)
        coverage = self._build_coverage(combined, csv_files, skipped, dupes_removed)
        coverage_path = self.output_dir / "matches_coverage.json"
        with open(coverage_path, "w", encoding="utf-8") as f:
            json.dump(coverage, f, indent=2, default=str)

        logger.info("═══ Canonical Dataset Build — DONE ═══")
        return {
            "status": "success",
            "total_matches": len(combined),
            "files_processed": len(csv_files) - len(skipped),
            "files_skipped": skipped,
            "duplicates_removed": dupes_removed,
            "leagues": sorted(combined["league"].unique().tolist()),
            "matches_path": str(matches_path),
            "coverage_path": str(coverage_path),
        }

    # ==================================================================
    # Step 1 — Discover raw files
    # ==================================================================
    def _discover_files(self) -> List[tuple]:
        """
        Walk source directories for CSV files.

        Returns list of (Path, league_dir_name) tuples.
        """
        files = []
        for base in self.source_dirs:
            if not base.exists():
                continue
            # Walk league subdirectories
            for league_dir in sorted(base.iterdir()):
                if not league_dir.is_dir():
                    continue
                for csv_file in sorted(league_dir.glob("*.csv")):
                    files.append((csv_file, league_dir.name))

        logger.info("Discovered %d raw CSV files across %d source dirs", len(files), len(self.source_dirs))
        return files

    # ==================================================================
    # Step 2 — Parse and normalise each file
    # ==================================================================
    def _parse_and_normalise(
        self, path: Path, league_dir: str
    ) -> tuple:
        """
        Parse a single raw CSV and normalise to canonical schema.

        Returns:
            (DataFrame or None, coverage dict)
        """
        coverage: Dict[str, bool] = {}

        try:
            # Read with BOM handling
            df = pd.read_csv(path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(path, encoding="unicode_escape")
            except Exception as e:
                logger.warning("Failed to read %s: %s", path.name, e)
                return None, coverage

        # Fix BOM in column names
        df.columns = [c.strip().replace("\ufeff", "") for c in df.columns]

        # Check required columns
        missing = REQUIRED_RAW - set(df.columns)
        if missing:
            logger.warning("Skipping %s — missing required: %s", path.name, missing)
            return None, coverage

        # ── Detect league from Div column or directory name ──
        league = self._detect_league(df, league_dir, path)
        if league is None:
            logger.warning("Skipping %s — unmapped league", path.name)
            return None, coverage

        # ── Drop ALL odds/bookmaker columns ──
        odds_cols = [c for c in df.columns if ODDS_PATTERNS.match(c)]
        # Also drop non-signal columns we don't need
        extra_drop = {"Div", "FTR", "HTR", "Time", "Attendance"}
        drop_cols = set(odds_cols) | (extra_drop & set(df.columns))
        df = df.drop(columns=list(drop_cols), errors="ignore")

        # ── Rename signal columns ──
        rename = {k: v for k, v in COL_MAP.items() if k in df.columns}
        df = df.rename(columns=rename)

        # ── Parse date ──
        df["match_date"] = pd.to_datetime(
            df["Date"], dayfirst=True, errors="coerce"
        )
        bad_dates = df["match_date"].isna().sum()
        if bad_dates:
            logger.warning("%s: %d rows with unparseable dates", path.name, bad_dates)
            df = df.dropna(subset=["match_date"])
        if df.empty:
            return None, coverage
        df = df.drop(columns=["Date"], errors="ignore")

        # ── Season detection from filename or dates ──
        df["season"] = self._detect_season(path, df["match_date"])

        # ── League ──
        df["league"] = league

        # ── Normalise team names ──
        df["home_team"] = df["home_team"].str.strip().str.upper()
        df["away_team"] = df["away_team"].str.strip().str.upper()

        int_cols = [
            "home_goals", "away_goals",
            "home_goals_ht", "away_goals_ht",
            "home_shots", "away_shots",
            "home_shots_on_target", "away_shots_on_target",
            "home_corners", "away_corners",
            "home_fouls", "away_fouls",
            "home_yellow_cards", "away_yellow_cards",
            "home_red_cards", "away_red_cards",
        ]
        for col in int_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                coverage[col] = True
            else:
                df[col] = pd.NA
                coverage[col] = False

        # Drop rows missing required goals
        df = df.dropna(subset=["home_goals", "away_goals"])
        if df.empty:
            return None, coverage

        # Cast goals to int
        df["home_goals"] = df["home_goals"].astype(int)
        df["away_goals"] = df["away_goals"].astype(int)

        # Referee
        if "referee" not in df.columns:
            df["referee"] = pd.NA
            coverage["referee"] = False
        else:
            df["referee"] = df["referee"].astype(str).str.strip()
            df.loc[df["referee"].isin(["nan", ""]), "referee"] = pd.NA
            coverage["referee"] = True

        # ── Compute derived discipline and shot metrics ──
        for side in ["home", "away"]:
            # cards = yellow + red
            df[f"{side}_cards"] = df[[f"{side}_yellow_cards", f"{side}_red_cards"]].sum(axis=1, min_count=1)
            coverage[f"{side}_cards"] = coverage.get(f"{side}_yellow_cards", False) or coverage.get(f"{side}_red_cards", False)

            # total_shots = shots
            if f"{side}_shots" in df.columns:
                df[f"{side}_total_shots"] = df[f"{side}_shots"]
                coverage[f"{side}_total_shots"] = coverage.get(f"{side}_shots", False)

        # ── Compute match_hash ──
        df["match_hash"] = df.apply(
            lambda r: hashlib.sha256(
                f"{r['league']}|{r['season']}|{r['match_date'].date()}|{r['home_team']}|{r['away_team']}".encode()
            ).hexdigest()[:16],
            axis=1,
        )

        # Keep only target schema columns
        schema_cols = [
            "match_hash", "match_date", "season", "league",
            "home_team", "away_team",
            "home_goals", "away_goals", "home_goals_ht", "away_goals_ht",
            "home_shots", "away_shots", "home_shots_on_target", "away_shots_on_target",
            "home_corners", "away_corners", "home_fouls", "away_fouls",
            "home_yellow_cards", "away_yellow_cards", "home_red_cards", "away_red_cards",
            "home_cards", "away_cards",
            "home_total_shots", "away_total_shots",
            "referee"
        ]
        df = df[[c for c in schema_cols if c in df.columns]]

        # Ensure all schema columns exist
        for c in schema_cols:
            if c not in df.columns:
                df[c] = pd.NA

        logger.info(
            "Parsed %s: %d matches [%s, %s]",
            path.name, len(df), league, df["season"].iloc[0] if not df.empty else "?",
        )
        return df, coverage

    # ==================================================================
    # Step 4 — Schema validation
    # ==================================================================
    def _validate_schema(self, df: pd.DataFrame) -> None:
        """Assert schema invariants."""
        required = ["league", "season", "match_date", "home_team", "away_team",
                     "home_goals", "away_goals"]
        missing = [c for c in required if c not in df.columns]
        assert not missing, f"Missing required columns: {missing}"

        assert (df["home_goals"] >= 0).all(), "Negative home_goals found"
        assert (df["away_goals"] >= 0).all(), "Negative away_goals found"

        invalid_leagues = set(df["league"].unique()) - VALID_LEAGUES
        assert not invalid_leagues, f"Invalid leagues: {invalid_leagues}"

        assert df["match_hash"].nunique() == len(df), "Duplicate match_hash after dedup"

        # Log coverage summary
        for league in sorted(df["league"].unique()):
            lg_df = df[df["league"] == league]
            seasons = sorted(lg_df["season"].unique())
            logger.info(
                "[Validate] %s: %d matches, seasons %s, dates %s → %s",
                league, len(lg_df), seasons,
                lg_df["match_date"].min().date(),
                lg_df["match_date"].max().date(),
            )

        # Null rate log
        for col in NULLABLE_SIGNALS:
            if col in df.columns:
                null_rate = df[col].isna().mean()
                if null_rate > 0:
                    logger.info("[Validate] %s null rate: %.1f%%", col, null_rate * 100)

    # ==================================================================
    # Helpers
    # ==================================================================
    def _detect_league(
        self, df: pd.DataFrame, league_dir: str, path: Path
    ) -> Optional[str]:
        """Detect league code from Div column, filename, or directory."""
        # Try Div column first
        if "Div" in df.columns:
            raw_code = str(df["Div"].dropna().iloc[0]).strip() if not df["Div"].dropna().empty else None
            if raw_code and raw_code in DIV_MAP:
                return DIV_MAP[raw_code]

        # Try filename prefix
        stem = path.stem.upper()
        for raw, std in DIV_MAP.items():
            if stem.startswith(raw.upper()):
                return std

        # Try directory name (already a standard code)
        if league_dir.upper() in VALID_LEAGUES:
            return league_dir.upper()

        # Try directory-to-league mapping
        dir_map = {
            "PL": "PL", "BL1": "BL1", "FL1": "FL1", "SA": "SA", "PD": "PD",
        }
        return dir_map.get(league_dir.upper())

    def _detect_season(
        self, path: Path, dates: pd.Series
    ) -> str:
        """
        Detect season label from filename or date range.

        Returns format: '2023-24'
        """
        stem = path.stem

        # Try extracting from filename patterns like 'E0_2425' or 'PL_season_2019_2020'
        # Pattern: 4-digit pairs like '2425'
        m = re.search(r"(\d{2})(\d{2})(?:_|$)", stem)
        if m:
            y1 = int("20" + m.group(1))
            y2 = int("20" + m.group(2))
            if 2010 <= y1 <= 2030 and y2 == y1 + 1:
                return f"{y1}-{m.group(2)}"

        # Pattern: full years like '2019_2020'
        m = re.search(r"(\d{4})[_\-](\d{4})", stem)
        if m:
            y1 = int(m.group(1))
            y2_short = m.group(2)[2:]
            return f"{y1}-{y2_short}"

        # Fallback: derive from dates
        if not dates.dropna().empty:
            min_date = dates.min()
            max_date = dates.max()
            # Football seasons cross year boundary (August → May)
            if min_date.month >= 7:
                start_year = min_date.year
            else:
                start_year = min_date.year - 1
            end_short = str(start_year + 1)[2:]
            return f"{start_year}-{end_short}"

        return "unknown"

    def _build_coverage(
        self,
        df: pd.DataFrame,
        source_files: List[tuple],
        skipped: List[str],
        dupes_removed: int,
    ) -> Dict[str, Any]:
        """Build coverage metadata."""
        league_stats: Dict[str, Any] = {}
        for league, grp in df.groupby("league"):
            seasons = sorted(grp["season"].dropna().unique().tolist())
            null_rates = {}
            for col in NULLABLE_SIGNALS:
                if col in grp.columns:
                    rate = round(float(grp[col].isna().mean()), 4)
                    if rate > 0:
                        null_rates[col] = rate

            league_stats[str(league)] = {
                "match_count": len(grp),
                "seasons": seasons,
                "date_range": {
                    "earliest": str(grp["match_date"].min().date()),
                    "latest": str(grp["match_date"].max().date()),
                },
                "null_rates": null_rates,
            }

        return {
            "schema_version": SCHEMA_VERSION,
            "build_timestamp": datetime.now(timezone.utc).isoformat(),
            "total_matches": len(df),
            "total_leagues": df["league"].nunique(),
            "duplicates_removed": dupes_removed,
            "source_files_processed": len(source_files) - len(skipped),
            "source_files_skipped": skipped,
            "leagues": league_stats,
            "columns": sorted(df.columns.tolist()),
        }
