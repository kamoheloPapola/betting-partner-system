from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
from rich.console import Console
from rich.table import Table
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.config import LOGS_DIR
from src.core.container import ServiceContainer
from src.db.models import ResolvedPrediction
from src.ml.calibration import calculate_ece

DEFAULT_DAYS = 30
DEFAULT_LOG_FILE = LOGS_DIR / "predictions.log"
ECE_TOLERANCE = 0.005
LIVE_MARKET_ALIASES = {
    "home_win": "home_win",
    "home": "home_win",
    "draw": "draw",
    "x": "draw",
    "away_win": "away_win",
    "away": "away_win",
    "over_2_5": "over_2_5",
    "goals_over_2_5": "over_2_5",
    "over_2_5_goals": "over_2_5",
    "btts_yes": "btts_yes",
}
SHADOW_FIELD_TO_MARKET = {
    "home": "home_win",
    "draw": "draw",
    "away": "away_win",
    "over_2_5": "over_2_5",
    "btts_yes": "btts_yes",
}
RL_SHADOW_PATTERN = re.compile(
    r"\[RL-SHADOW\]\s+"
    r"match_id=(?P<match_id>\S+)\s+"
    r"context=(?P<context>\S+)\s+"
    r"weights=\{.*?\}\s+"
    r"home=(?P<home>\d+(?:\.\d+)?)\s+"
    r"draw=(?P<draw>\d+(?:\.\d+)?)\s+"
    r"away=(?P<away>\d+(?:\.\d+)?)\s+"
    r"over_2_5=(?P<over_2_5>\d+(?:\.\d+)?)\s+"
    r"btts_yes=(?P<btts_yes>\d+(?:\.\d+)?)\s+"
    r"applied=(?P<applied>\S+)"
)


@dataclass(frozen=True)
class ShadowComparisonRow:
    bucket: str
    live_ece: float
    shadow_ece: float
    delta: float
    verdict: str


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare live and RL shadow calibration over recent resolved predictions.")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS, help="Number of lookback days to compare (default: 30)")
    parser.add_argument(
        "--log-file",
        type=Path,
        default=DEFAULT_LOG_FILE,
        help="Prediction log file containing [RL-SHADOW] entries",
    )
    return parser.parse_args(argv)


def normalize_live_market(market: str) -> Optional[str]:
    return LIVE_MARKET_ALIASES.get(str(market or "").strip().lower())


def compute_ece_from_frame(frame: pd.DataFrame, probability_column: str) -> float:
    if frame.empty:
        return 0.0
    probabilities = frame[probability_column].to_numpy(dtype=float)
    outcomes = frame["actual_outcome"].to_numpy(dtype=float)
    return float(calculate_ece(outcomes, probabilities, n_bins=10))


def verdict_for_bucket(live_ece: float, shadow_ece: float) -> str:
    if shadow_ece < (live_ece - ECE_TOLERANCE):
        return "PROMOTE"
    if abs(shadow_ece - live_ece) <= ECE_TOLERANCE:
        return "HOLD"
    return "REVERT"


def determine_exit_code(rows: list[ShadowComparisonRow]) -> int:
    verdicts = {row.verdict for row in rows}
    if "REVERT" in verdicts:
        return 2
    if "PROMOTE" in verdicts:
        return 0
    return 1


def load_live_rows(days: int, container: Optional[ServiceContainer] = None) -> pd.DataFrame:
    active_container = container or ServiceContainer.get_instance()
    with Session(active_container.engine) as session:
        records = session.execute(
            select(ResolvedPrediction).where(
                ResolvedPrediction.outcome.in_(("WON", "LOST"))
            )
        ).scalars().all()

    if not records:
        return pd.DataFrame(
            columns=["match_hash", "league", "market", "live_probability", "actual_outcome", "resolved_at"]
        )

    frame = pd.DataFrame(
        [
            {
                "match_hash": str(record.match_hash),
                "league": str(record.league).upper(),
                "market": record.market,
                "live_probability": record.probability,
                "actual_outcome": 1 if record.outcome == "WON" else 0,
                "resolved_at": record.resolved_at.isoformat() if record.resolved_at else None,
            }
            for record in records
        ]
    )
    frame["resolved_at"] = pd.to_datetime(frame["resolved_at"], utc=True, errors="coerce")
    cutoff = pd.Timestamp(datetime.now(timezone.utc) - timedelta(days=days))
    frame = frame[frame["resolved_at"] >= cutoff]
    frame["market"] = frame["market"].map(normalize_live_market)
    frame["live_probability"] = pd.to_numeric(frame["live_probability"], errors="coerce")
    frame["actual_outcome"] = pd.to_numeric(frame["actual_outcome"], errors="coerce")
    frame = frame.dropna(subset=["match_hash", "league", "market", "live_probability", "actual_outcome"])
    frame = frame[frame["live_probability"].between(0.0, 1.0, inclusive="both")]
    return frame.reset_index(drop=True)


def parse_shadow_log(log_file: Path) -> pd.DataFrame:
    if not log_file.exists():
        return pd.DataFrame(columns=["match_hash", "league", "market", "shadow_probability"])

    rows: list[dict[str, object]] = []
    with open(log_file, encoding="utf-8") as handle:
        for line in handle:
            match = RL_SHADOW_PATTERN.search(line)
            if match is None:
                continue

            context = match.group("context")
            league = context.split(":", 1)[0].upper()
            match_hash = match.group("match_id")
            for field, market in SHADOW_FIELD_TO_MARKET.items():
                rows.append(
                    {
                        "match_hash": match_hash,
                        "league": league,
                        "market": market,
                        "shadow_probability": float(match.group(field)),
                    }
                )

    if not rows:
        return pd.DataFrame(columns=["match_hash", "league", "market", "shadow_probability"])

    frame = pd.DataFrame(rows)
    frame = frame.drop_duplicates(subset=["match_hash", "league", "market"], keep="last")
    return frame.reset_index(drop=True)


def build_comparison_rows(
    *,
    days: int = DEFAULT_DAYS,
    log_file: Path = DEFAULT_LOG_FILE,
    container: Optional[ServiceContainer] = None,
) -> list[ShadowComparisonRow]:
    live_rows = load_live_rows(days, container=container)
    shadow_rows = parse_shadow_log(log_file)
    if live_rows.empty or shadow_rows.empty:
        return []

    merged = live_rows.merge(
        shadow_rows,
        on=["match_hash", "league", "market"],
        how="inner",
    )
    if merged.empty:
        return []

    rows: list[ShadowComparisonRow] = []
    for (league, market), group in merged.groupby(["league", "market"]):
        live_ece = compute_ece_from_frame(group, "live_probability")
        shadow_ece = compute_ece_from_frame(group, "shadow_probability")
        rows.append(
            ShadowComparisonRow(
                bucket=f"{league}:{market}",
                live_ece=live_ece,
                shadow_ece=shadow_ece,
                delta=live_ece - shadow_ece,
                verdict=verdict_for_bucket(live_ece, shadow_ece),
            )
        )

    return sorted(rows, key=lambda row: row.bucket)


def render_table(rows: list[ShadowComparisonRow], console: Optional[Console] = None) -> None:
    active_console = console or Console()
    table = Table(title="RL Shadow Comparison")
    table.add_column("Bucket")
    table.add_column("Live ECE", justify="right")
    table.add_column("Shadow ECE", justify="right")
    table.add_column("Delta", justify="right")
    table.add_column("Verdict")

    for row in rows:
        table.add_row(
            row.bucket,
            f"{row.live_ece:.4f}",
            f"{row.shadow_ece:.4f}",
            f"{row.delta:+.4f}",
            row.verdict,
        )

    active_console.print(table)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    rows = build_comparison_rows(days=int(args.days), log_file=Path(args.log_file))
    console = Console()
    if not rows:
        console.print("No comparable live/shadow buckets found.")
        return 1

    render_table(rows, console=console)
    return determine_exit_code(rows)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
