"""
Team-Level Corner/Card Offsets.

Provides team-specific deviation adjustments for corner and card predictions.
Uses Bayesian shrinkage toward league mean for stability.

Rules:
- Only apply offset if team has ≥200 matches
- Freeze offsets: Only update pre-season OR after ≥50 new matches
- Shrink aggressively toward league mean
"""
import logging
import unicodedata
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

from src.config import DATA_DIR, PROCESSED_DATA_DIR
from src.utils.naming import normalize_team_name

logger = logging.getLogger(__name__)

# Define public API
__all__ = ["TeamOffsets", "TeamOffsetManager"]

# --- CONFIGURATION ---
MIN_MATCHES_FOR_OFFSET = 15   # Lowered to 15 to include teams with ~half a season of data
MIN_NEW_MATCHES_FOR_UPDATE = 20  # Need ≥20 new matches to update
SHRINKAGE_FACTOR = 0.7  # How aggressively to shrink toward league mean

def _normalize_team_name(name: str, league: Optional[str] = None) -> str:
    """
    Normalize team name using the canonical normalization logic.
    """
    return normalize_team_name(name, league=league)


@dataclass
class TeamOffsets:
    """Offset data for a single team."""
    team_name: str
    league: str
    home_corner_bias: float = 0.0
    away_corner_bias: float = 0.0
    home_card_bias: float = 0.0
    away_card_bias: float = 0.0
    match_count: int = 0
    last_updated: Optional[datetime] = None
    frozen: bool = False
    
    def is_valid(self) -> bool:
        """Check if offset is valid for use."""
        return self.match_count >= MIN_MATCHES_FOR_OFFSET


class TeamOffsetManager:
    """
    Manages team-level offsets for corners and cards.
    
    Usage:
        manager = TeamOffsetManager()
        manager.compute_offsets("PL")
        
        # Get offset for a specific team
        offset = manager.get_offset("Manchester United", "PL")
        adjusted_lambda = base_lambda + offset.home_corner_bias
    """
    
    def __init__(self):
        self.offsets: Dict[Tuple[str, str], TeamOffsets] = {}
        self.offset_file = DATA_DIR / "models" / "team_offsets.csv"
        self._load_offsets()
    
    def _load_offsets(self) -> None:
        """Load cached offsets from disk."""
        if not self.offset_file.exists():
            logger.info("No cached team offsets found")
            return
        
        try:
            df = pd.read_csv(self.offset_file)
            for _, row in df.iterrows():
                l_code = str(row['league']).upper()
                # Action 8.4: Global Normalization using canonical utility
                t_name = _normalize_team_name(row['team_name'], league=l_code)
                key = (t_name, l_code)
                
                # De-duplicate: Keep the entry with highest match count for this normalized key
                match_count = int(row['match_count'])
                if key in self.offsets:
                    if match_count <= self.offsets[key].match_count:
                        continue
                
                self.offsets[key] = TeamOffsets(
                    team_name=t_name,
                    league=l_code,
                    home_corner_bias=row['home_corner_bias'],
                    away_corner_bias=row['away_corner_bias'],
                    home_card_bias=row.get('home_card_bias', 0.0),
                    away_card_bias=row.get('away_card_bias', 0.0),
                    match_count=match_count,
                    last_updated=pd.to_datetime(row['last_updated']) if pd.notna(row.get('last_updated')) else None,
                    frozen=bool(row.get('frozen', False))
                )
            logger.info(f"Loaded {len(self.offsets)} team offsets from cache")
        except Exception as e:
            logger.warning(f"Failed to load team offsets: {e}")
    
    def _save_offsets(self) -> None:
        """Persist offsets to disk."""
        self.offset_file.parent.mkdir(parents=True, exist_ok=True)
        
        rows = []
        for key, offset in self.offsets.items():
            rows.append({
                'team_name': offset.team_name,
                'league': offset.league,
                'home_corner_bias': offset.home_corner_bias,
                'away_corner_bias': offset.away_corner_bias,
                'home_card_bias': offset.home_card_bias,
                'away_card_bias': offset.away_card_bias,
                'match_count': offset.match_count,
                'last_updated': offset.last_updated.isoformat() if offset.last_updated else None,
                'frozen': offset.frozen
            })
        
        pd.DataFrame(rows).to_csv(self.offset_file, index=False)
        logger.info(f"Saved {len(rows)} team offsets to {self.offset_file}")
    
    def compute_offsets(self, league: str, force: bool = False) -> int:
        """
        Compute team offsets for a league from historical data.
        
        Args:
            league: League code
            force: If True, recompute even if frozen
            
        Returns:
            Number of teams with valid offsets
        """
        # Load historical data
        match_files = list((PROCESSED_DATA_DIR / "matches").glob(f"{league}_*.csv"))
        if not match_files:
            logger.warning(f"No match data found for {league}")
            return 0
        
        dfs = []
        for f in match_files:
            try:
                dfs.append(pd.read_csv(f, low_memory=False))
            except Exception as e:
                logger.warning(f"Failed to load {f}: {e}")
        
        if not dfs:
            return 0
        
        matches = pd.concat(dfs, ignore_index=True)
        
        # Calculate league means for shrinkage
        league_mean_home_corners = matches['home_corners'].mean() if 'home_corners' in matches.columns else 5.0
        league_mean_away_corners = matches['away_corners'].mean() if 'away_corners' in matches.columns else 4.5
        
        # Card means for shrinkage (using total_cards if available, otherwise compute)
        if 'home_total_cards' in matches.columns:
            league_mean_home_cards = matches['home_total_cards'].mean()
            league_mean_away_cards = matches['away_total_cards'].mean() if 'away_total_cards' in matches.columns else league_mean_home_cards
        elif 'total_cards' in matches.columns:
            league_mean_total_cards = matches['total_cards'].mean()
            league_mean_home_cards = league_mean_total_cards / 2
            league_mean_away_cards = league_mean_total_cards / 2
        else:
            league_mean_home_cards = 2.0  # Default fallback
            league_mean_away_cards = 2.0
        
        # Aggregate by team
        teams = set(matches['home_team'].unique()) | set(matches['away_team'].unique())
        valid_count = 0
        
        for team in teams:
            # Action 8.4: Normalize before key creation
            t_norm = _normalize_team_name(team, league=league)
            
            # Check freeze status
            key = (t_norm, league)
            existing = self.offsets.get(key)
            
            if existing and existing.frozen and not force:
                logger.debug(f"Skipping frozen offset for {team}")
                continue
            
            # Count matches
            home_matches = matches[matches['home_team'] == team]
            away_matches = matches[matches['away_team'] == team]
            total_matches = len(home_matches) + len(away_matches)
            
            # Check update criteria
            if existing and not force:
                new_matches = total_matches - existing.match_count
                if new_matches < MIN_NEW_MATCHES_FOR_UPDATE:
                    logger.debug(f"Skipping {team}: only {new_matches} new matches")
                    continue
            
            # Calculate raw CORNER deviations
            if 'home_corners' in matches.columns and len(home_matches) > 0:
                raw_home_corner = home_matches['home_corners'].mean() - league_mean_home_corners
            else:
                raw_home_corner = 0.0
            
            if 'away_corners' in matches.columns and len(away_matches) > 0:
                raw_away_corner = away_matches['away_corners'].mean() - league_mean_away_corners
            else:
                raw_away_corner = 0.0
            
            # Calculate raw CARD deviations
            raw_home_card = 0.0
            raw_away_card = 0.0
            
            if 'home_total_cards' in matches.columns and len(home_matches) > 0:
                raw_home_card = home_matches['home_total_cards'].mean() - league_mean_home_cards
            elif 'total_cards' in matches.columns and len(home_matches) > 0:
                # Use proportion of total cards for home games
                raw_home_card = (home_matches['total_cards'].mean() / 2) - league_mean_home_cards
            
            if 'away_total_cards' in matches.columns and len(away_matches) > 0:
                raw_away_card = away_matches['away_total_cards'].mean() - league_mean_away_cards
            elif 'total_cards' in matches.columns and len(away_matches) > 0:
                raw_away_card = (away_matches['total_cards'].mean() / 2) - league_mean_away_cards
            
            # Apply Bayesian shrinkage
            # Shrink more aggressively for teams with fewer matches
            reliability = min(total_matches / 200, 1.0)  # Full reliability at 200 matches (lowered from 500)
            effective_shrinkage = SHRINKAGE_FACTOR * reliability
            
            shrunk_home_corner = raw_home_corner * effective_shrinkage
            shrunk_away_corner = raw_away_corner * effective_shrinkage
            shrunk_home_card = raw_home_card * effective_shrinkage
            shrunk_away_card = raw_away_card * effective_shrinkage
            
            # Store offset
            self.offsets[key] = TeamOffsets(
                team_name=team,
                league=league,
                home_corner_bias=shrunk_home_corner,
                away_corner_bias=shrunk_away_corner,
                home_card_bias=shrunk_home_card,
                away_card_bias=shrunk_away_card,
                match_count=total_matches,
                last_updated=datetime.now(),
                frozen=False
            )
            
            if total_matches >= MIN_MATCHES_FOR_OFFSET:
                valid_count += 1
                logger.debug(f"Computed offset for {team}: corners h={shrunk_home_corner:.3f}/a={shrunk_away_corner:.3f}, cards h={shrunk_home_card:.3f}/a={shrunk_away_card:.3f}")
        
        # Persist
        self._save_offsets()
        logger.info(f"Computed offsets for {league}: {valid_count} teams with valid offsets")
        
        return valid_count
    
    def get_offset(self, team: str, league: str) -> Optional[TeamOffsets]:
        """
        Get offset for a team, if valid.
        
        """
        # Action 8.4: Case-Insensitive Lookup with Normalization
        # GUARDRAIL: Verify caller is passing normalized names (or re-normalize defensively)
        normalized_team = _normalize_team_name(team, league=league)
        if team.upper() != normalized_team:
            logger.debug(f"TeamOffsetManager.get_offset received un-normalized name: '{team}' -> '{normalized_team}'")
            
        key = (normalized_team, str(league).upper())
        offset = self.offsets.get(key)
        
        if offset and offset.is_valid():
            return offset
        
        return None
    
    def freeze_all(self, league: str) -> None:
        """Freeze all offsets for a league (mid-season protection)."""
        frozen_count = 0
        for key, offset in self.offsets.items():
            if offset.league == league:
                offset.frozen = True
                frozen_count += 1
        
        self._save_offsets()
        logger.info(f"Froze {frozen_count} team offsets for {league}")
    
    def unfreeze_all(self, league: str) -> None:
        """Unfreeze all offsets for a league (pre-season)."""
        for key, offset in self.offsets.items():
            if offset.league == league:
                offset.frozen = False
        
        self._save_offsets()
        logger.info(f"Unfroze all team offsets for {league}")
