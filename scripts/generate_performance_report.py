from __future__ import annotations

import csv
import json
import sys
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

OUTCOMES_CSV = Path("data/eval/prediction_outcomes.csv")
REPORT_JSON = Path("data/eval/performance_report.json")
VALID_OUTCOMES = {"WON", "LOST", "VOID"}


def _safe_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _win_rate(wins: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(wins / total, 4)


def _roi_pct(total_profit: float, total_bets: int) -> float:
    if total_bets <= 0:
        return 0.0
    return round((total_profit / total_bets) * 100.0, 2)


def _bin_index(probability: float) -> int | None:
    if not 0.0 <= probability <= 1.0:
        return None
    if probability == 1.0:
        return 9
    return int(probability * 10)


def generate_report() -> dict:
    if not OUTCOMES_CSV.exists():
        raise FileNotFoundError(f"Prediction outcomes file not found: {OUTCOMES_CSV}")

    market_counts: dict[str, dict[str, int | float]] = defaultdict(
        lambda: {"total": 0, "wins": 0, "profit": 0.0}
    )
    league_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "wins": 0}
    )
    calibration_bins: dict[int, dict[str, int]] = defaultdict(
        lambda: {"count": 0, "wins": 0}
    )

    total_bets = 0
    total_wins = 0
    total_voids = 0
    total_profit = 0.0

    with OUTCOMES_CSV.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            outcome = str(row.get("outcome", "")).strip().upper()
            if outcome not in VALID_OUTCOMES:
                continue

            if outcome == "VOID":
                total_voids += 1
                continue

            market = str(row.get("market", "") or "").strip() or "UNKNOWN"
            league = str(row.get("league", "") or "").strip() or "UNKNOWN"

            total_bets += 1
            market_counts[market]["total"] += 1
            league_counts[league]["total"] += 1

            if outcome == "WON":
                total_wins += 1
                market_counts[market]["wins"] += 1
                league_counts[league]["wins"] += 1

            probability = _safe_float(row.get("probability"))
            if probability is not None and probability > 0.0:
                profit = (1.0 / probability) - 1.0 if outcome == "WON" else -1.0
                total_profit += profit
                market_counts[market]["profit"] += profit

            if probability is None:
                continue

            bin_idx = _bin_index(probability)
            if bin_idx is None:
                continue

            calibration_bins[bin_idx]["count"] += 1
            if outcome == "WON":
                calibration_bins[bin_idx]["wins"] += 1

    by_market = {}
    for market in sorted(market_counts):
        total = int(market_counts[market]["total"])
        wins = int(market_counts[market]["wins"])
        profit = float(market_counts[market]["profit"])
        by_market[market] = {
            "total": total,
            "wins": wins,
            "win_rate": _win_rate(wins, total),
            "roi_pct": _roi_pct(profit, total),
        }

    by_league = {}
    for league in sorted(league_counts):
        total = int(league_counts[league]["total"])
        wins = int(league_counts[league]["wins"])
        by_league[league] = {
            "total": total,
            "wins": wins,
            "win_rate": _win_rate(wins, total),
        }

    calibration = []
    for bin_idx in range(10):
        count = calibration_bins[bin_idx]["count"]
        if count < 10:
            continue
        lower = bin_idx / 10
        upper = 1.0 if bin_idx == 9 else (bin_idx + 1) / 10
        midpoint = round((lower + upper) / 2, 4)
        actual_win_rate = round(calibration_bins[bin_idx]["wins"] / count, 4)
        calibration.append(
            {
                "bin_label": f"{lower:.1f}-{upper:.1f}",
                "count": count,
                "actual_win_rate": actual_win_rate,
                "midpoint": midpoint,
            }
        )

    generated_at = datetime.now(timezone.utc).isoformat()
    summary = {
        "wins": total_wins,
        "overall_win_rate": _win_rate(total_wins, total_bets),
        "total_bets": total_bets,
        "total_voids": total_voids,
        "roi_pct": _roi_pct(total_profit, total_bets),
        "generated_at": generated_at,
    }

    report = {
        "summary": summary,
        "by_market": by_market,
        "by_league": by_league,
        "calibration": calibration,
    }

    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=== PERFORMANCE REPORT ===")
    print(
        f"Overall: {total_wins} W / {total_bets} bets "
        f"({summary['overall_win_rate']:.1%}) | ROI: {summary['roi_pct']:.2f}%"
    )
    print(f"Generated: {generated_at}")
    print()

    print("--- By Market ---")
    for market, stats in by_market.items():
        print(
            f"{market:<20} {stats['total']:>5} bets  "
            f"{stats['win_rate']:>6.1%} win rate  ROI: {stats['roi_pct']:>+5.1f}%"
        )
    print()

    print("--- By League ---")
    for league, stats in by_league.items():
        print(
            f"{league:<6} {stats['total']:>5} bets  "
            f"{stats['win_rate']:>6.1%} win rate"
        )
    print()

    print("--- Calibration (bins >= 10 bets) ---")
    for bucket in calibration:
        print(
            f"{bucket['bin_label']:<12} {bucket['count']:>4} bets  "
            f"actual: {bucket['actual_win_rate']:.1%}  expected: {bucket['midpoint']:.1%}"
        )

    return report


if __name__ == "__main__":
    try:
        generate_report()
        raise SystemExit(0)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
