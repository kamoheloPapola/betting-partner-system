"""
Prediction Result Resolver.

Maps predictions to actual match outcomes for evaluation. The
AuthoritativeResolver class reads predictions and labeled results,
resolves WON/LOST/VOID states, and produces prediction_outcomes.csv.

Features:
- Hash-based match linking
- Market alias normalization
- Incremental processing with deduplication
"""

import pandas as pd
import numpy as np
import logging
import os
import sys
import re
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import ClassVar, List, Dict, Any, Set, Optional, Iterator, TypeAlias
from sqlalchemy import select
from sqlalchemy.orm import Session

# Add project root relative to this file
try:
    from src.config import DATA_DIR
    from src.utils.naming import generate_match_fingerprint
    from src.core.exceptions import DataValidationError
    from src.db.connection import database_is_configured, get_engine
    from src.db.models import ResolvedPrediction
except ImportError:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.config import DATA_DIR
    from src.utils.naming import generate_match_fingerprint
    from src.core.exceptions import DataValidationError
    from src.db.connection import database_is_configured, get_engine
    from src.db.models import ResolvedPrediction

logger = logging.getLogger(__name__)

LabeledResultsMap: TypeAlias = Dict[str, Dict[str, Any]]

@dataclass
class ResolutionStats:
    """Track resolution metrics."""
    processed_files: int = 0
    predictions_read: int = 0
    resolved_count: int = 0
    unmatched_predictions: int = 0
    duplicate_ids: int = 0
    invalid_probs: int = 0
    failed_hashes: int = 0
    validation_errors: int = 0
    won: int = 0
    lost: int = 0
    void: int = 0

    @property
    def skipped_count(self) -> int:
        return self.duplicate_ids + self.invalid_probs + self.failed_hashes

    def log_summary(self):
        """Log a summary of the resolution process."""
        resolved_pct = (self.resolved_count / max(self.predictions_read, 1)) * 100
        logger.info(
            f"Resolution Summary: "
            f"Files={self.processed_files}, "
            f"Read={self.predictions_read}, "
            f"Resolved={self.resolved_count} ({resolved_pct:.1f}%) "
            f"(W={self.won}, L={self.lost}, V={self.void}), "
            f"Unmatched={self.unmatched_predictions}, "
            f"Skipped={self.skipped_count} "
            f"(Dupes={self.duplicate_ids}, BadProb={self.invalid_probs}, BadHash={self.failed_hashes}), "
            f"ValErrors={self.validation_errors}"
        )

@dataclass
class AuthoritativeResolver:
    """
    Decoupled resolver that maps predictions to labeled results.
    Produces data/eval/prediction_outcomes.csv with states WON, LOST, VOID.
    """
    
    # Class-level defaults
    DEFAULT_LABELED_PATH: ClassVar[Path] = DATA_DIR / "results" / "labeled" / "results_labeled.csv"
    DEFAULT_OUTCOMES_PATH: ClassVar[Path] = DATA_DIR / "eval" / "prediction_outcomes.csv"
    DEFAULT_PRED_DIR: ClassVar[Path] = DATA_DIR / "predictions"
    MAX_FILE_SIZE_MB: ClassVar[int] = 500
    OUTCOME_COLUMNS: ClassVar[List[str]] = [
        "prediction_id",
        "match_hash",
        "league",
        "kickoff_date",
        "market",
        "probability",
        "outcome",
        "resolved_at",
    ]
    
    MARKET_ALIASES: ClassVar[Dict[str, str]] = {
        # 1x2
        'HOME_WIN': 'home_win',
        'AWAY_WIN': 'away_win',
        'home': 'home_win',
        'away': 'away_win',

        # Goals / team goals
        'HOME_TG_U1.5': 'home_goals_under_1_5',
        'AWAY_TG_U1.5': 'away_goals_under_1_5',
        'home_under_1_5': 'home_goals_under_1_5',
        'away_under_1_5': 'away_goals_under_1_5',
        'over_2_5': 'goals_over_2_5',
        'under_2_5': 'goals_under_2_5',
        'btts': 'btts_yes',
        'btts_no': 'btts_no',

        # Total corners — main markets
        'CORNERS_U11.5': 'corners_under_11_5',
        'corners_u11.5': 'corners_under_11_5',
        'corn_u11': 'corners_under_11_5',
        'corners_under_11_5': 'corners_under_11_5',
        'corn_o75': 'corners_over_7_5',
        'corners_over_7_5': 'corners_over_7_5',

        # Home corners lines
        'corn_home_u25': 'home_corners_under_2_5',
        'corn_home_o25': 'home_corners_over_2_5',
        'corn_home_u35': 'home_corners_under_3_5',
        'corn_home_o35': 'home_corners_over_3_5',
        'corn_home_u45': 'home_corners_under_4_5',
        'corn_home_o45': 'home_corners_over_4_5',
        'corn_home_u55': 'home_corners_under_5_5',
        'corn_home_o55': 'home_corners_over_5_5',
        'corn_home_u65': 'home_corners_under_6_5',
        'corn_home_o65': 'home_corners_over_6_5',

        # Away corners lines
        'corn_away_u25': 'away_corners_under_2_5',
        'corn_away_o25': 'away_corners_over_2_5',
        'corn_away_u35': 'away_corners_under_3_5',
        'corn_away_o35': 'away_corners_over_3_5',
        'corn_away_u45': 'away_corners_under_4_5',
        'corn_away_o45': 'away_corners_over_4_5',
        'corn_away_u55': 'away_corners_under_5_5',
        'corn_away_o55': 'away_corners_over_5_5',
        'corn_away_u65': 'away_corners_under_6_5',
        'corn_away_o65': 'away_corners_over_6_5',

        # Cards
        'cards_u4.5': 'cards_under_4_5',
        'card_u45': 'cards_under_4_5',
        'card_u55': 'cards_under_5_5',
        'CARDS_U4.5': 'cards_under_4_5',
        'cards_o2.5': 'cards_over_2_5',
        'cards_under_5_5': 'cards_under_5_5',

        # Double chance
        'HOME_DC': 'home_dc',
        'AWAY_DC': 'away_dc',
    }
    VALID_LEAGUES: ClassVar[Set[str]] = {"PL", "BL1", "PD", "SA", "FL1", "UCL"}

    # Instance attributes (allows dependency injection)
    labeled_path: Optional[Path] = None
    outcomes_path: Optional[Path] = None
    pred_dir: Optional[Path] = None
    dry_run: bool = False
    
    def __post_init__(self):
        self.labeled_path = self.labeled_path or self.DEFAULT_LABELED_PATH
        self.outcomes_path = self.outcomes_path or self.DEFAULT_OUTCOMES_PATH
        self.pred_dir = self.pred_dir or self.DEFAULT_PRED_DIR

    def validate(self) -> ResolutionStats:
        """Validate predictions without saving outcomes (convenience wrapper)."""
        original_dry_run = self.dry_run
        self.dry_run = True
        try:
            return self.resolve_all()
        finally:
            self.dry_run = original_dry_run

    def load_outcomes(self, include_void: bool = False) -> pd.DataFrame:
        """Load normalized outcome records for downstream diagnostics."""
        outcomes = self._load_outcomes_frame()
        if outcomes.empty:
            return outcomes

        outcomes = self._repair_legacy_outcomes_schema(outcomes)

        required = {"league", "market", "probability", "outcome"}
        missing = required - set(outcomes.columns)
        if missing:
            raise DataValidationError(
                f"Outcome file missing required columns: {missing}",
                context={"path": str(self.outcomes_path), "available": list(outcomes.columns)},
            )

        if "kickoff_date" in outcomes.columns:
            outcomes["kickoff_date"] = pd.to_datetime(
                outcomes["kickoff_date"], errors="coerce", utc=True
            )
        if "resolved_at" in outcomes.columns:
            outcomes["resolved_at"] = pd.to_datetime(
                outcomes["resolved_at"], errors="coerce", utc=True
            )

        outcomes["probability"] = pd.to_numeric(outcomes["probability"], errors="coerce")
        outcomes = outcomes[outcomes["probability"].between(0, 1, inclusive="both")]

        # Normalise legacy numeric outcomes (0/1) to string schema (WON/LOST)
        outcome_col = outcomes["outcome"]
        if pd.api.types.is_numeric_dtype(outcome_col):
            num = pd.to_numeric(outcome_col, errors="coerce")
            outcomes["outcome"] = num.map({1.0: "WON", 0.0: "LOST"}).fillna(
                outcome_col.astype(str)
            )

        if not include_void:
            outcomes = outcomes[outcomes["outcome"].isin(["WON", "LOST"])]

        outcomes["market"] = outcomes["market"].astype(str).str.upper()
        outcomes["outcome"] = outcomes["outcome"].map({"WON": 1, "LOST": 0})
        outcomes = outcomes.dropna(subset=["league", "market", "probability", "outcome"]).copy()
        outcomes["outcome"] = outcomes["outcome"].astype(int)
        return outcomes

    def resolve_all(self) -> ResolutionStats:
        """Main orchestration method."""
        try:
            labeled_df = self._load_labeled_results()
            # Convert to map for amortized O(1) lookup: Key=match_hash
            labeled_map: LabeledResultsMap = labeled_df.to_dict('index')
            
            processed_ids = self._load_existing_outcomes()
            stats = ResolutionStats()
            
            outcomes = []
            
            for pred_file in self._iter_prediction_files():
                stats.processed_files += 1
                try:
                    file_outcomes = self._process_prediction_file(
                        pred_file, labeled_map, processed_ids, stats
                    )
                    outcomes.extend(file_outcomes)
                except (pd.errors.ParserError, KeyError, ValueError) as e:
                    logger.error(
                        f"Failed to process prediction file",
                        extra={"file": pred_file.name, "error": str(e), "error_type": type(e).__name__}
                    )
                except Exception as e:
                    logger.critical(
                        f"Unexpected error in prediction processing",
                        extra={"file": pred_file.name, "error": str(e)}
                    )
                    raise
            
            self._save_outcomes(outcomes)
            stats.log_summary()
            
            if stats.validation_errors > 0:
                logger.warning(f"{stats.validation_errors} validation errors occurred during resolution")
                
            return stats
            
        except Exception as e:
            logger.error(f"Resolution failed: {e}")
            raise

    def _load_labeled_results(self) -> pd.DataFrame:
        """Load and validate labeled results."""
        if not self.labeled_path.exists():
            raise DataValidationError(
                "Labeled results not found. Run 'ingest-results' first.",
                context={"expected_path": str(self.labeled_path)}
            )
        
        df = pd.read_csv(self.labeled_path)
        
        # Validate required columns
        required = ['match_hash', 'kickoff_date_utc']
        missing = set(required) - set(df.columns)
        if missing:
            raise DataValidationError(
                f"Labeled results missing required columns: {missing}",
                context={"available": list(df.columns)}
            )
        
        return df.set_index('match_hash')

    def _load_existing_outcomes(self) -> Set[str]:
        """Load previously processed prediction IDs."""
        if database_is_configured():
            db_prediction_ids = self._load_existing_outcome_ids_from_db()
            if db_prediction_ids:
                return db_prediction_ids

        if not self.outcomes_path.exists():
            return set()
        
        try:
            existing = pd.read_csv(self.outcomes_path)
            if 'prediction_id' in existing.columns:
                return set(existing['prediction_id'].unique())
            return set()
        except pd.errors.EmptyDataError:
            return set()

    def _load_outcomes_frame(self) -> pd.DataFrame:
        if database_is_configured():
            db_outcomes = self._load_outcomes_from_db()
            if db_outcomes is not None:
                return db_outcomes

        if not self.outcomes_path.exists():
            return pd.DataFrame(columns=self.OUTCOME_COLUMNS)

        try:
            return pd.read_csv(self.outcomes_path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame(columns=self.OUTCOME_COLUMNS)

    def _load_outcomes_from_db(self) -> Optional[pd.DataFrame]:
        try:
            with Session(get_engine()) as session:
                rows = session.execute(
                    select(ResolvedPrediction).order_by(ResolvedPrediction.resolved_at.asc())
                ).scalars().all()
        except Exception as exc:
            logger.warning("Failed to load resolved predictions from database: %s", exc)
            return None

        if not rows:
            return None

        return pd.DataFrame(
            [
                {
                    "prediction_id": row.prediction_id,
                    "match_hash": row.match_hash,
                    "league": row.league,
                    "kickoff_date": row.kickoff_date.isoformat() if row.kickoff_date else None,
                    "market": row.market,
                    "probability": row.probability,
                    "outcome": row.outcome,
                    "resolved_at": row.resolved_at.isoformat() if row.resolved_at else None,
                }
                for row in rows
            ],
            columns=self.OUTCOME_COLUMNS,
        )

    def _load_existing_outcome_ids_from_db(self) -> Set[str]:
        try:
            with Session(get_engine()) as session:
                rows = session.execute(select(ResolvedPrediction.prediction_id)).all()
        except Exception as exc:
            logger.warning("Failed to load resolved prediction ids from database: %s", exc)
            return set()

        return {str(row[0]) for row in rows if row and row[0]}

    def _iter_prediction_files(self) -> Iterator[Path]:
        """Yield prediction files to process."""
        if not self.pred_dir.exists():
            logger.warning(f"Prediction directory not found: {self.pred_dir}")
            return
        
        files = list(self.pred_dir.glob("*.csv"))
        csv_files = [f for f in files if "evaluation" not in f.name]
        
        if not csv_files:
            logger.info(f"No prediction files found in {self.pred_dir}")
            return

        logger.info(f"Found {len(csv_files)} prediction file(s) to process")
        
        for pred_file in csv_files:
            try:
                pred_file.resolve().relative_to(self.pred_dir.resolve())
            except ValueError:
                logger.warning(f"Skipping file outside pred directory: {pred_file}")
                continue
                
            yield pred_file

    def _normalize_prediction_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names across different prediction formats."""
        df = df.copy()
        
        col_map = {
            'match_id': 'match_hash',
            'date': 'prediction_date',
            'timestamp': 'prediction_date',
            'probability': 'predicted_probability',
            'goal_confidence': 'predicted_probability',
            'goal_pick': 'market'
        }
        
        for source, target in col_map.items():
            if target not in df.columns and source in df.columns:
                df.rename(columns={source: target}, inplace=True)
                
        return df
    
    def _validate_probability(self, prob: Any) -> bool:
        """Validate probability is a valid numeric value between 0 and 1."""
        try:
            # Convert to float if possible (handles numeric strings)
            prob_val = float(prob) if not pd.isna(prob) else None
            return prob_val is not None and 0 <= prob_val <= 1
        except (ValueError, TypeError):
            return False

    def _validate_match_hash(self, h: str) -> bool:
        """Validate match hash format (16 hex chars)."""
        return bool(re.match(r'^[a-f0-9]{16}$', h, re.IGNORECASE))

    def _process_prediction_file(
        self,
        pred_file: Path,
        labeled_map: LabeledResultsMap,
        processed_ids: Set[str],
        stats: ResolutionStats
    ) -> List[Dict[str, Any]]:
        """
        Process a single prediction file and resolve results.
        """
        # Size risk mitigation
        try:
            file_size_mb = pred_file.stat().st_size / (1024 * 1024)
            if file_size_mb > self.MAX_FILE_SIZE_MB:
                logger.warning(f"Skipping large file {pred_file.name} ({file_size_mb:.1f}MB > {self.MAX_FILE_SIZE_MB}MB)")
                return []
        except OSError as e:
            logger.warning(f"Could not stat file {pred_file.name}: {e}")
            return []

        preds = pd.read_csv(pred_file)
        preds = self._normalize_prediction_columns(preds)
        
        # Validate critical columns after normalization
        required = ['match_hash', 'prediction_date', 'predicted_probability', 'market']
        missing = [c for c in required if c not in preds.columns]
        
        if missing:
            logger.warning(
                f"Skipping prediction file: missing critical columns",
                extra={"file": pred_file.name, "missing": missing}
            )
            return []

        # Ensure date formats
        preds['prediction_date'] = pd.to_datetime(preds['prediction_date'], errors='coerce')
        
        if 'kickoff_date' in preds.columns:
            preds['kickoff_date'] = pd.to_datetime(preds['kickoff_date'], errors='coerce')
        else:
            logger.debug(f"No kickoff_date column in {pred_file.name}, using prediction_date")
            preds['kickoff_date'] = preds['prediction_date']

        file_outcomes = []
        
        for _, row in preds.iterrows():
            stats.predictions_read += 1
            
            # Check probability validation
            prob = row.get('predicted_probability')
            if not self._validate_probability(prob):
                logger.warning(f"Invalid probability {prob} in {pred_file.name}. Skipping.")
                stats.invalid_probs += 1
                continue

            # Normalize Hash
            h = self._normalize_match_hash(row)
            if not h:
                stats.failed_hashes += 1
                continue

            market = row['market']
            pred_id = f"{h}_{market}"
            
            if pred_id in processed_ids:
                stats.duplicate_ids += 1
                continue
            
            processed_ids.add(pred_id)
            
            if h in labeled_map:
                try:
                    outcome = self._generate_outcome_record(row, labeled_map[h], h, pred_file.name)
                    outcome['prediction_id'] = pred_id
                    
                    if outcome['outcome'] == 'WON': stats.won += 1
                    elif outcome['outcome'] == 'LOST': stats.lost += 1
                    else: stats.void += 1
                    
                    file_outcomes.append(outcome)
                    stats.resolved_count += 1
                    
                except DataValidationError as e:
                    logger.error(f"Validation error for {h}: {e}")
                    stats.validation_errors += 1
                    continue
            else:
                stats.unmatched_predictions += 1
        
        return file_outcomes

    def _normalize_match_hash(self, row: pd.Series) -> Optional[str]:
        """Normalize match hash, regenerating if necessary."""
        h = str(row.get('match_hash', ''))
        
        if self._validate_match_hash(h):
            return h
        
        # Attempt regeneration
        try:
            required_fields = ['league', 'prediction_date', 'home_team', 'away_team']
            if not all(field in row for field in required_fields):
                return None
            
            if row[required_fields].isna().any():
                return None

            date_obj = pd.to_datetime(row['prediction_date'])
            regenerated = generate_match_fingerprint(
                row['league'], 
                date_obj, 
                row['home_team'], 
                row['away_team']
            )
            return regenerated
            
        except Exception as e:
            logger.debug(f"Hash regeneration failed: {type(e).__name__} - {e}")
            return None

    def _load_result_date(self, result_row: Dict[str, Any], match_hash: str) -> pd.Timestamp:
        """Safely parse result date and ensure UTC."""
        date_val = result_row.get('kickoff_date_utc')
        
        if date_val is None:
            raise DataValidationError(
                "Missing kickoff_date_utc in labeled results",
                context={"match_hash": match_hash}
            )

        ts = pd.to_datetime(date_val)
        if ts.tz is None:
            ts = ts.tz_localize('UTC')
        else:
            ts = ts.tz_convert('UTC')
        return ts

    def _generate_outcome_record(
        self,
        pred_row: pd.Series,
        result_row: Dict[str, Any],
        match_hash: str,
        filename: str = "unknown"
    ) -> Dict[str, Any]:
        """Create outcome record for a single prediction."""
        
        # Date Safety Check with UTC normalization
        res_date = self._load_result_date(result_row, match_hash)
        
        # pred_row date needs similar treatment
        pred_date = pd.to_datetime(pred_row['kickoff_date'])
        
        if pd.isna(pred_date):
             raise DataValidationError(
                f"Invalid kickoff_date for prediction",
                context={"match_hash": match_hash, "file": filename}
            )

        if pred_date.tz is None:
             pred_date = pred_date.tz_localize('UTC')
        else:
             pred_date = pred_date.tz_convert('UTC')
        
        if res_date < pred_date:
             raise DataValidationError(
                f"Result date precedes prediction date (time travel detected)",
                context={
                    "match_hash": match_hash,
                    "result_date": res_date.isoformat(),
                    "prediction_date": pred_date.isoformat(),
                    "file": filename
                }
            )

        market = pred_row['market']
        lookup_market = self.MARKET_ALIASES.get(market, market)
        
        val = result_row.get(lookup_market)
        
        if val is not None and not pd.isna(val):
            state = "WON" if int(val) == 1 else "LOST"
        else:
            logger.warning(f"Market {market} not found in labeled results for {match_hash}. VOID.")
            state = "VOID"

        return {
            'match_hash': match_hash,
            'league': pred_row.get('league'),
            'kickoff_date': res_date.isoformat(),
            'market': market,
            'probability': pred_row['predicted_probability'],
            'outcome': state,
            'resolved_at': datetime.now(timezone.utc).isoformat()
        }

    def _repair_legacy_outcomes_schema(self, outcomes: pd.DataFrame) -> pd.DataFrame:
        """Repair rows appended with a historical column-order bug."""
        if outcomes.empty:
            return outcomes

        required = set(self.OUTCOME_COLUMNS)
        if not required.issubset(outcomes.columns):
            return outcomes

        repaired = outcomes.copy()
        shifted_rows = (
            repaired["probability"].astype(str).isin({"WON", "LOST", "VOID"})
            & pd.to_numeric(repaired["market"], errors="coerce").notna()
        )
        if not shifted_rows.any():
            return repaired

        original = repaired.loc[shifted_rows].copy()
        repaired.loc[shifted_rows, "prediction_id"] = original["resolved_at"]
        repaired.loc[shifted_rows, "match_hash"] = original["prediction_id"]
        repaired.loc[shifted_rows, "league"] = original["match_hash"]
        repaired.loc[shifted_rows, "kickoff_date"] = original["league"]
        repaired.loc[shifted_rows, "market"] = original["kickoff_date"]
        repaired.loc[shifted_rows, "probability"] = original["market"]
        repaired.loc[shifted_rows, "outcome"] = original["probability"]
        repaired.loc[shifted_rows, "resolved_at"] = original["outcome"]
        return repaired

    def _save_outcomes(self, outcomes: List[Dict[str, Any]], batch_size: int = 50000) -> None:
        """Append outcomes to CSV in batches."""
        if self.dry_run:
            logger.info(f"DRY RUN: Would save {len(outcomes)} outcomes")
            return

        if not outcomes:
            logger.info("No new predictions to resolve.")
            return
        
        self.outcomes_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Batch save to manage memory and partial writes
        total_saved = 0
        tmp_path = self.outcomes_path.with_suffix(".tmp")
        if tmp_path.exists():
            tmp_path.unlink()
        if self.outcomes_path.exists():
            tmp_path.write_bytes(self.outcomes_path.read_bytes())
        
        for i in range(0, len(outcomes), batch_size):
            batch = outcomes[i:i+batch_size]
            normalized_batch = []
            for row in batch:
                try:
                    normalized_batch.append(self._normalize_outcome_row(row))
                except ValueError as exc:
                    logger.warning(
                        "Skipping invalid outcome row (match_hash=%s): %s",
                        row.get("match_hash"),
                        exc,
                    )
            new_df = pd.DataFrame(normalized_batch, columns=self.OUTCOME_COLUMNS)
            
            # Determine mode appropriately: append always unless it's the very first write to a new file
            # If default_outcomes_path exists, process is purely additive.
            # If it didn't exist, the first batch (i=0) creates it (w), subsequent append (a).
            
            if i == 0:
                 mode = 'a' if self.outcomes_path.exists() else 'w'
                 header = not self.outcomes_path.exists()
            else:
                 mode = 'a'
                 header = False
             
            new_df.to_csv(tmp_path, mode=mode, header=header, index=False)
            self._save_outcomes_batch_to_db(normalized_batch)
            total_saved += len(batch)
        if tmp_path.exists():
            if self.outcomes_path.exists() and (tmp_path.stat().st_size == 0):
                tmp_path.unlink()
            else:
                os.replace(tmp_path, self.outcomes_path)
        
        logger.info(
            f"Resolved predictions",
            extra={"count": total_saved, "output_path": str(self.outcomes_path)}
        )

    @staticmethod
    def _parse_datetime(value: Any) -> Optional[datetime]:
        if value in {None, ""}:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, pd.Timestamp):
            return value.to_pydatetime()
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except Exception:
            parsed = pd.to_datetime(value, errors="coerce", utc=True)
            if pd.isna(parsed):
                return None
            return parsed.to_pydatetime()

    def _normalize_outcome_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        missing = [column for column in self.OUTCOME_COLUMNS if column not in row]
        extras = [column for column in row.keys() if column not in self.OUTCOME_COLUMNS]
        if missing or extras:
            raise ValueError(
                f"Outcome row schema mismatch: missing={missing} extras={extras} row_keys={list(row.keys())}"
            )

        normalized: Dict[str, Any] = {column: row[column] for column in self.OUTCOME_COLUMNS}

        normalized["prediction_id"] = str(normalized["prediction_id"]).strip()
        normalized["match_hash"] = str(normalized["match_hash"]).strip()

        league = str(normalized["league"]).strip().upper()
        if league not in self.VALID_LEAGUES:
            raise ValueError(f"Unknown league code in outcome row: {league}")
        normalized["league"] = league

        kickoff_date = self._parse_datetime(normalized["kickoff_date"])
        if kickoff_date is None:
            raise ValueError(f"Invalid kickoff_date in outcome row: {normalized['kickoff_date']}")
        if kickoff_date.tzinfo is None:
            kickoff_date = kickoff_date.replace(tzinfo=timezone.utc)
        normalized["kickoff_date"] = kickoff_date.astimezone(timezone.utc).isoformat()

        normalized["market"] = str(normalized["market"]).strip().lower()

        try:
            probability = float(normalized["probability"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid probability in outcome row: {normalized['probability']}") from exc
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"Probability out of bounds in outcome row: {probability}")
        normalized["probability"] = probability

        normalized["outcome"] = str(normalized["outcome"]).strip().upper()

        resolved_at = self._parse_datetime(normalized["resolved_at"])
        if resolved_at is None:
            raise ValueError(f"Invalid resolved_at in outcome row: {normalized['resolved_at']}")
        if resolved_at.tzinfo is None:
            resolved_at = resolved_at.replace(tzinfo=timezone.utc)
        normalized["resolved_at"] = resolved_at.astimezone(timezone.utc).isoformat()

        if list(normalized.keys()) != self.OUTCOME_COLUMNS:
            raise ValueError(
                f"Outcome row column order mismatch: expected={self.OUTCOME_COLUMNS} actual={list(normalized.keys())}"
            )

        return normalized

    def _save_outcomes_batch_to_db(self, outcomes: List[Dict[str, Any]]) -> None:
        if not outcomes or not database_is_configured():
            return

        try:
            with Session(get_engine()) as session:
                for outcome in outcomes:
                    session.merge(
                        ResolvedPrediction(
                            prediction_id=str(outcome.get("prediction_id")),
                            match_hash=str(outcome.get("match_hash")),
                            league=str(outcome.get("league")),
                            kickoff_date=self._parse_datetime(outcome.get("kickoff_date")),
                            market=str(outcome.get("market")) if outcome.get("market") is not None else None,
                            probability=float(outcome["probability"])
                            if outcome.get("probability") is not None
                            else None,
                            outcome=str(outcome.get("outcome")) if outcome.get("outcome") is not None else None,
                            resolved_at=self._parse_datetime(outcome.get("resolved_at")),
                        )
                    )
                session.commit()
        except Exception as exc:
            logger.warning("Failed to persist resolved predictions to database: %s", exc)
        
MARKET_ALIASES = AuthoritativeResolver.MARKET_ALIASES

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    resolver = AuthoritativeResolver()
    resolver.resolve_all()
