"""
One-time backfill: re-resolve VOID outcomes using updated MARKET_ALIASES.

Safe to run multiple times — only touches rows with outcome == 'VOID'
where the market now has a known alias mapping to a labeled column.
Writes atomically (temp file → rename).
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.resolve_results import AuthoritativeResolver

OUTCOMES_PATH = Path("data/eval/prediction_outcomes.csv")
LABELED_PATH = Path("data/results/labeled/results_labeled.csv")


def main() -> None:
    print("Loading outcomes...")
    outcomes = pd.read_csv(OUTCOMES_PATH)
    total = len(outcomes)
    void_mask = outcomes["outcome"] == "VOID"
    void_count = void_mask.sum()
    print(f"  Total rows  : {total}")
    print(f"  VOID rows   : {void_count}")

    print("\nLoading labeled results...")
    labeled = pd.read_csv(LABELED_PATH).set_index("match_hash")
    print(f"  Labeled rows: {len(labeled)}")

    aliases = AuthoritativeResolver.MARKET_ALIASES

    updated = 0
    skipped_no_alias = 0
    skipped_no_hash = 0
    skipped_no_column = 0

    print("\nBackfilling VOID rows...")
    for idx in outcomes.index[void_mask]:
        row = outcomes.loc[idx]
        market = str(row["market"])
        match_hash = str(row.get("match_hash", ""))

        # Skip if market has no alias
        lookup_market = aliases.get(market)
        if lookup_market is None:
            skipped_no_alias += 1
            continue

        # Skip if match_hash not in labeled
        if match_hash not in labeled.index:
            skipped_no_hash += 1
            continue

        result_row = labeled.loc[match_hash]

        # Handle duplicate index (multiple rows for same hash)
        if isinstance(result_row, pd.DataFrame):
            result_row = result_row.iloc[0]

        val = result_row.get(lookup_market)

        if val is None or pd.isna(val):
            skipped_no_column += 1
            continue

        new_outcome = "WON" if int(val) == 1 else "LOST"
        outcomes.at[idx, "outcome"] = new_outcome
        outcomes.at[idx, "resolved_at"] = datetime.now(timezone.utc).isoformat()
        updated += 1

    print(f"\nResults:")
    print(f"  Updated (VOID -> WON/LOST) : {updated}")
    print(f"  Skipped (no alias)         : {skipped_no_alias}")
    print(f"  Skipped (hash not in labeled): {skipped_no_hash}")
    print(f"  Skipped (column missing)   : {skipped_no_column}")
    print(f"  Remaining VOID             : {void_count - updated}")

    if updated == 0:
        print("\nNo rows updated. Exiting without writing.")
        return

    # Atomic write
    tmp = OUTCOMES_PATH.with_suffix(".tmp")
    outcomes.to_csv(tmp, index=False)
    tmp.replace(OUTCOMES_PATH)
    print(f"\nAtomically wrote {len(outcomes)} rows to {OUTCOMES_PATH}")

    # Final summary
    final = pd.read_csv(OUTCOMES_PATH)
    won = (final["outcome"] == "WON").sum()
    lost = (final["outcome"] == "LOST").sum()
    void = (final["outcome"] == "VOID").sum()
    print(f"\nFinal state:")
    print(f"  WON        : {won}")
    print(f"  LOST       : {lost}")
    print(f"  VOID       : {void}")
    print(f"  Settled    : {won + lost}")
    wr = won / (won + lost) * 100 if (won + lost) > 0 else 0
    print(f"  Win rate   : {wr:.1f}%")


if __name__ == "__main__":
    main()
