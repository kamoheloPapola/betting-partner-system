"""Test show-predictions directly without typer"""
import sys
sys.path.insert(0, '.')

from rich.console import Console
console = Console(force_terminal=True)

from src.core.container import ServiceContainer
from src.cli.utils import filter_matches_by_date, resolve_league_code
from src.core.validators import validate_match_dataframe

print("Starting prediction test...")

# 1. Data Loading
df = ServiceContainer.get_instance().pipeline.run()
validate_match_dataframe(df, context="show_predictions")
print(f"Loaded {len(df)} total matches")

df_target = filter_matches_by_date(df, 'yesterday', show_all=False, user_timezone='LOCAL')
print(f"Filtered to {len(df_target)} matches for yesterday")

if df_target.empty:
    console.print("[yellow][!] No matches found for filter: yesterday[/yellow]")
else:
    # Show first few matches
    console.print(f"[bold green]Found {len(df_target)} matches for yesterday:[/bold green]")
    for idx, row in df_target.head(5).iterrows():
        console.print(f"  • {row['home_team']} vs {row['away_team']} ({row['league']})")
    
    # Now try running predictions
    from src.cli.commands.prediction import _run_predict_loop, _render_output, _prepare_bets
    from src.strategies.selection_gate import SelectionGate
    
    print("\nRunning prediction loop...")
    try:
        results = _run_predict_loop(df_target)
        print(f"Generated {len(results)} predictions")
        
        if results:
            print("First prediction:")
            print(f"  Match: {results[0]['match']}")
            print(f"  Card U5.5: {results[0].get('card_u55', 'N/A')}")
            
            # Try rendering
            gated, stats = SelectionGate().process(_prepare_bets(results))
            _render_output(results, gated, stats, console)
    except Exception as e:
        print(f"Error in prediction: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
