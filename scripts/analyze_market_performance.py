"""Analyze market performance from reconstructed history."""
import json
from collections import defaultdict
from pathlib import Path

history_file = Path("data/slips/ff_reconstructed_history.jsonl")
market_stats = defaultdict(lambda: {"wins": 0, "losses": 0})

with open(history_file) as f:
    for line in f:
        sel = json.loads(line)
        market = sel["market"]
        outcome = sel["outcome"]
        if outcome == "WIN":
            market_stats[market]["wins"] += 1
        elif outcome == "LOSS":
            market_stats[market]["losses"] += 1

print("=" * 55)
print("MARKET PERFORMANCE ANALYSIS (Jan 4-11, 2026)")
print("=" * 55)
print()

# Sort by win rate
sorted_markets = sorted(
    market_stats.items(),
    key=lambda x: x[1]["wins"] / (x[1]["wins"] + x[1]["losses"]) if (x[1]["wins"] + x[1]["losses"]) > 0 else 0,
    reverse=True
)

print(f"{'Market':<20} {'Wins':<6} {'Losses':<6} {'Total':<6} {'Win Rate':<10}")
print("-" * 55)

for market, stats in sorted_markets:
    total = stats["wins"] + stats["losses"]
    win_rate = stats["wins"] / total * 100 if total > 0 else 0
    print(f"{market:<20} {stats['wins']:<6} {stats['losses']:<6} {total:<6} {win_rate:.1f}%")

print()
print("BEST PERFORMING MARKET:", sorted_markets[0][0] if sorted_markets else "N/A")
