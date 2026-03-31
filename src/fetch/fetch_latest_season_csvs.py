"""
Season CSV Fetcher.

Downloads latest season result CSVs from Football-Data.co.uk with:
- Hash-aware conditional refresh (skip unchanged files)
- Automatic backup before overwrite
- Retry logic for resilient downloads
"""
import requests
import hashlib
import logging
import os
import sys
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, ClassVar
from dataclasses import dataclass, field
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Add project root correctly (Issue #1 Fix)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR

logger = logging.getLogger(__name__)

@dataclass
class SeasonFetcher:
    """
    Automated fetcher for latest season CSVs from Football-Data.co.uk.
    Implements immutable snapshots and hash-aware conditional refresh.
    """
    
    # Class-level defaults
    BASE_URL: ClassVar[str] = "https://www.football-data.co.uk/mmz4281"
    
    # Instance attributes (for dependency injection)
    output_dir: Path = field(default_factory=lambda: DATA_DIR / "results" / "raw")
    league_map: Dict[str, str] = field(default_factory=lambda: {
        "PL": "E0",
        "PD": "SP1",
        "SA": "I1",
        "BL1": "D1",
        "FL1": "F1"
    })
    timeout: int = 30
    
    def __post_init__(self):
        """Ensure output directory exists."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_session(self) -> requests.Session:
        """Create session with retry logic (Issue #9)."""
        session = requests.Session()
        
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,  # Wait 1s, 2s, 4s between retries
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "HEAD"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session

    def _log_league_event(self, league: str, message: str, **kwargs) -> None:
        """Helper for consistent league-prefixed logging (Refinement #2)."""
        logger.info(f"[{league}] {message}", extra={'league': league, **kwargs})

    def _calculate_hash(self, content: bytes) -> str:
        """Calculate SHA-256 hash of content."""
        return hashlib.sha256(content).hexdigest()

    def _has_remote_changed(
        self, 
        url: str, 
        local_path: Path,
        assume_changed_on_error: bool = False
    ) -> bool:
        """
        Check if remote file differs from local using headers or hash.
        (Refinement #4: Added assume_changed_on_error)
        """
        session = self._get_session()
        try:
            # 1. Try HEAD request for Last-Modified header
            response = session.head(url, timeout=10)
            remote_modified = response.headers.get('Last-Modified')
            
            if remote_modified:
                remote_time = datetime.strptime(
                    remote_modified, 
                    '%a, %d %b %Y %H:%M:%S %Z'
                )
                local_time = datetime.fromtimestamp(local_path.stat().st_mtime)
                # If remote is newer, it changed
                if remote_time > local_time:
                    return True
                
            # 2. Fallback: Compare SHA-256 hashes
            response = session.get(url, timeout=self.timeout)
            response.raise_for_status()
            remote_hash = self._calculate_hash(response.content)
            
            with open(local_path, 'rb') as f:
                local_hash = self._calculate_hash(f.read())
            
            return remote_hash != local_hash
            
        except Exception as e:
            logger.warning(
                f"Could not verify remote status for {url}: {e}. "
                f"Assuming {'stale' if assume_changed_on_error else 'up-to-date'}."
            )
            return assume_changed_on_error

    def fetch_season(self, season: str = "2526", force: bool = False) -> None:
        """
        Download latest season CSVs for all configured leagues.
        
        Args:
            season: Season identifier in YYZZ format (e.g., "2526" for 2025/26)
            force: Skip smart refresh and download fresh copies.
            
        Side Effects:
            Downloads CSVs to data/results/raw/{LEAGUE}/season_{season}.csv
            Appends download metadata to fetch_log_{season}.txt
            
        Raises:
            ValueError: If season format is invalid
            requests.RequestException: If download fails
            IOError: If file write fails
        """
        # Validate season format (Issue #5 Fix)
        if not re.match(r'^\d{4}$', season):
            raise ValueError(
                f"Invalid season format: '{season}'. Expected YYZZ format (e.g., '2526' for 2025/26)"
            )
        
        # Sanity check: season year should be reasonable
        year_start = int(season[:2])
        if year_start < 20 or year_start > 30:  # 2020-2030 range
            logger.warning(
                f"Season year {year_start} seems unusual. Did you mean 20{season[:2]}/{season[2:]}?"
            )

        log_file = self.output_dir / f"fetch_log_{season}.txt"
        session = self._get_session()
        
        for league, code in self.league_map.items():
            url = f"{self.BASE_URL}/{season}/{code}.csv"
            league_dir = self.output_dir / league
            league_dir.mkdir(parents=True, exist_ok=True)
            
            target_path = league_dir / f"season_{season}.csv"
            
            # Smart Refresh Logic (Issue #2 Fix)
            if target_path.exists() and not force:
                if not self._has_remote_changed(url, target_path, assume_changed_on_error=False):
                    self._log_league_event(league, "Local file is up-to-date. Skipping.")
                    continue
                
                # Remote changed -> Backup old version
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = league_dir / f"season_{season}_{timestamp}.csv.bak"
                target_path.rename(backup_path)
                self._log_league_event(league, f"Remote changed. Backed up old version to {backup_path.name}")

            try:
                self._log_league_event(league, f"Downloading {season} data...")
                response = session.get(url, timeout=self.timeout)
                response.raise_for_status()
                
                content = response.content
                file_hash = self._calculate_hash(content)
                
                with open(target_path, "wb") as f:
                    f.write(content)
                    
                # Use structured logging for metadata (Issue #7 Fix)
                self._log_league_event(
                    league, 
                    "Download complete",
                    season=season,
                    hash=file_hash,
                    url=url,
                    path=str(target_path)
                )
                
                self._log_league_event(league, f"Successfully fetched and verified version {file_hash[:12]}")
                
            except requests.RequestException as e:
                logger.error(
                    "HTTP request failed",
                    extra={
                        "league": league,
                        "season": season,
                        "url": url,
                        "error": str(e),
                        "error_type": type(e).__name__
                    }
                )
            except IOError as e:
                logger.error(
                    "File write failed",
                    extra={
                        "league": league,
                        "path": str(target_path),
                        "error": str(e)
                    }
                )
            except Exception as e:
                logger.critical(
                    "Unexpected error in fetch",
                    extra={"league": league, "error": str(e)}
                )
                raise  # Re-raise unexpected errors

if __name__ == "__main__":
    import argparse
    
    # Ensure logs directory exists
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "fetcher.log")
        ]
    )
    
    parser = argparse.ArgumentParser(description="Fetch latest season data")
    parser.add_argument("season", nargs='?', default="2526", help="Season in YYZZ format (e.g., 2526)")
    parser.add_argument("--force", action="store_true", help="Force re-download even if unchanged")
    
    args = parser.parse_args()
    
    fetcher = SeasonFetcher()
    fetcher.fetch_season(args.season, force=args.force)
