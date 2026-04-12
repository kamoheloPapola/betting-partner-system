from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.markets import MARKET_DEFINITIONS


LABELED_PATH = PROJECT_ROOT / "data" / "results" / "labeled" / "results_labeled.csv"
MASTER_PATH = PROJECT_ROOT / "data" / "results" / "normalized" / "results_master.csv"

LEGACY_MAP = {
    "over_2_5": "goals_over_2_5",
    "over_2_5_goals": "goals_over_2_5",
    "under_2_5": "goals_under_2_5",
    "under_2_5_goals": "goals_under_2_5",
    "over_9_5_corners": "corners_over_9_5",
    "total_corners_over_9_5": "corners_over_9_5",
    "under_11_5_corners": "corners_under_11_5",
    "total_corners_under_11_5": "corners_under_11_5",
    "home_under_1_5": "home_goals_under_1_5",
    "away_under_1_5": "away_goals_under_1_5",
    "total_cards_over_3_5": "cards_over_3_5",
    "total_cards_under_4_5": "cards_under_4_5",
}


def _canonical_market_columns() -> list[str]:
    columns: list[str] = []
    for markets in MARKET_DEFINITIONS.values():
        for market_name in markets:
            if market_name != "dependencies":
                columns.append(market_name)
    for canonical in LEGACY_MAP.values():
        if canonical not in columns:
            columns.append(canonical)
    return columns


def _fill_from_legacy(labeled_df: pd.DataFrame) -> int:
    filled = 0
    for legacy, canonical in LEGACY_MAP.items():
        if canonical not in labeled_df.columns:
            labeled_df[canonical] = np.nan
        if legacy not in labeled_df.columns:
            continue

        mask = labeled_df[canonical].isna() & labeled_df[legacy].notna()
        filled += int(mask.sum())
        labeled_df.loc[mask, canonical] = labeled_df.loc[mask, legacy]
    return filled


def _compute_market_labels(master_df: pd.DataFrame) -> pd.DataFrame:
    computed = pd.DataFrame({"match_hash": master_df["match_hash"]})

    for markets in MARKET_DEFINITIONS.values():
        deps = markets.get("dependencies", [])
        missing_deps = [dep for dep in deps if dep not in master_df.columns]

        for market_name, condition_fn in markets.items():
            if market_name == "dependencies":
                continue
            if missing_deps:
                computed[market_name] = np.nan
                continue

            bool_series = condition_fn(master_df)
            result = bool_series.astype(float)
            if deps:
                missing_mask = master_df[deps].isna().any(axis=1)
                result[missing_mask] = np.nan
            computed[market_name] = result

    return computed


def _fill_by_recompute(labeled_df: pd.DataFrame, master_df: pd.DataFrame) -> int:
    if "match_hash" not in labeled_df.columns:
        raise RuntimeError("Labeled results missing required column: match_hash")
    if "match_hash" not in master_df.columns:
        raise RuntimeError("Master results missing required column: match_hash")

    master_unique = master_df.drop_duplicates(subset=["match_hash"], keep="last")
    joined = labeled_df[["match_hash"]].merge(master_unique, on="match_hash", how="left")
    computed = _compute_market_labels(joined)

    filled = 0
    for market_name in _canonical_market_columns():
        if market_name not in labeled_df.columns:
            labeled_df[market_name] = np.nan
        if market_name not in computed.columns:
            continue

        missing_mask = labeled_df[market_name].isna()
        fillable_mask = missing_mask & computed[market_name].notna()
        filled += int(fillable_mask.sum())
        labeled_df.loc[fillable_mask, market_name] = computed.loc[fillable_mask, market_name]
    return filled


def _write_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)


def _print_summary(
    *,
    legacy_filled: int,
    recompute_filled: int,
    labeled_df: pd.DataFrame,
) -> None:
    print(f"Filled from legacy columns: {legacy_filled}")
    print(f"Filled by recompute: {recompute_filled}")
    print("Remaining null counts:")
    null_counts = labeled_df.isna().sum()
    remaining = null_counts[null_counts > 0]
    if remaining.empty:
        print("  none")
    else:
        for column, count in remaining.items():
            print(f"  {column}: {int(count)}")
    print(f"Total rows: {len(labeled_df)}")
    print(f"Total columns after drop: {len(labeled_df.columns)}")
    print(f"Saved: {LABELED_PATH}")


def main() -> int:
    labeled_df = pd.read_csv(LABELED_PATH)
    master_df = pd.read_csv(MASTER_PATH)

    legacy_filled = _fill_from_legacy(labeled_df)
    recompute_filled = _fill_by_recompute(labeled_df, master_df)
    labeled_df = labeled_df.drop(columns=list(LEGACY_MAP.keys()), errors="ignore")

    _write_atomic(labeled_df, LABELED_PATH)
    _print_summary(
        legacy_filled=legacy_filled,
        recompute_filled=recompute_filled,
        labeled_df=labeled_df,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
