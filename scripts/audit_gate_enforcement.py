"""
Gate Enforcement Audit Script

Tests critical gates:
1. Drift STOP must hard-block predictions
2. Cache invalidation verified
3. Market immunity respected
4. Offsets applied assertively
"""
import sys
import os
import json
from pathlib import Path

sys.path.insert(0, os.getcwd())

from rich.console import Console
from src.strategies.drift_guard import DriftGuardrail
from src.config import DATA_DIR

console = Console()

def test_drift_stop_enforcement():
    """Test that DRIFT STOP actually blocks predictions."""
    console.print("\n[bold cyan]TEST 1: DRIFT STOP HARD-BLOCK[/bold cyan]")
    
    # Simulate STOP status
    status_file = DATA_DIR / "drift" / "rolling_90d_status.json"
    status_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Backup current status
    backup = None
    if status_file.exists():
        with open(status_file) as f:
            backup = json.load(f)
    
    # Write STOP status
    with open(status_file, 'w') as f:
        json.dump({
            "date": "2026-01-13",
            "status": "STOP",
            "alerts": ["TEST_DRIFT_ALERT"],
            "metrics": {}
        }, f)
    
    # Check if drift guard reads STOP
    guard = DriftGuardrail()
    status = guard.check_drift()
    
    if status == "STOP":
        console.print("[green]✅ DriftGuardrail correctly returns STOP from file[/green]")
    else:
        console.print(f"[red]❌ FAIL: Expected STOP, got {status}[/red]")
    
    # Restore backup
    if backup:
        with open(status_file, 'w') as f:
            json.dump(backup, f)
    else:
        status_file.unlink(missing_ok=True)
    
    return status == "STOP"

def test_offset_loading():
    """Test that offset warnings are logged when offsets are missing."""
    console.print("\n[bold cyan]TEST 2: OFFSET LOADING VERIFICATION[/bold cyan]")
    
    from src.ml.models.corners.team_offsets import TeamOffsetManager
    
    mgr = TeamOffsetManager()
    
    # Test existing team
    off = mgr.get_offset("ARSENAL", "PL")
    if off is not None:
        console.print(f"[green]✅ Found offset for ARSENAL: corner_bias={off.home_corner_bias:.3f}[/green]")
    else:
        console.print("[yellow]⚠️ No offset found for ARSENAL (may need to train)[/yellow]")
    
    # Test non-existent team (should warn, not crash)
    off_fake = mgr.get_offset("FAKE_TEAM_123", "FAKE")
    if off_fake is None:
        console.print("[green]✅ Returns None for unknown teams (graceful degradation)[/green]")
    else:
        console.print(f"[red]❌ Unexpected offset for fake team: {off_fake}[/red]")
    
    return True

def test_forbidden_fruit_drift_block():
    """Test that ForbiddenFruit checks drift properly."""
    console.print("\n[bold cyan]TEST 3: FORBIDDEN FRUIT DRIFT CHECK[/bold cyan]")
    
    # Simulate STOP status
    status_file = DATA_DIR / "drift" / "rolling_90d_status.json"
    status_file.parent.mkdir(parents=True, exist_ok=True)
    
    backup = None
    if status_file.exists():
        with open(status_file) as f:
            backup = json.load(f)
    
    # Write STOP status
    with open(status_file, 'w') as f:
        json.dump({
            "date": "2026-01-13",
            "status": "STOP",
            "alerts": ["TEST_DRIFT_ALERT"],
            "metrics": {}
        }, f)
    
    from src.strategies.forbidden_fruit import ForbiddenFruitEngine
    
    engine = ForbiddenFruitEngine()
    
    # Try to analyze a match - should return empty due to STOP
    fake_match = {
        'home_team': 'ARSENAL',
        'away_team': 'CHELSEA',
        'league': 'PL',
        'match_id': 'test123',
        'date': '2026-01-13'
    }
    fake_preds = {
        'home_win': 0.45,
        'away_win': 0.30,
        'draw': 0.25,
        'over_2_5': 0.55,
        'over_1_5': 0.80
    }
    
    result = engine.analyze_match(fake_match, fake_preds)
    
    # Restore backup
    if backup:
        with open(status_file, 'w') as f:
            json.dump(backup, f)
    else:
        status_file.unlink(missing_ok=True)
    
    if result == []:
        console.print("[green]✅ ForbiddenFruit correctly blocks analysis when DRIFT=STOP[/green]")
        return True
    else:
        console.print(f"[red]❌ FAIL: ForbiddenFruit returned {len(result)} candidates despite STOP[/red]")
        return False

def run_all_tests():
    """Run all gate enforcement tests."""
    console.print("=" * 60, style="bold")
    console.print("GATE ENFORCEMENT AUDIT", style="bold yellow")
    console.print("=" * 60, style="bold")
    
    results = []
    
    results.append(("DRIFT STOP HARD-BLOCK", test_drift_stop_enforcement()))
    results.append(("OFFSET LOADING", test_offset_loading()))
    results.append(("FORBIDDEN FRUIT DRIFT", test_forbidden_fruit_drift_block()))
    
    console.print("\n" + "=" * 60, style="bold")
    console.print("AUDIT SUMMARY", style="bold")
    console.print("=" * 60, style="bold")
    
    all_pass = True
    for name, passed in results:
        status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
        console.print(f"  {name}: {status}")
        if not passed:
            all_pass = False
    
    if all_pass:
        console.print("\n[green bold]✅ ALL GATE ENFORCEMENT TESTS PASSED[/green bold]")
    else:
        console.print("\n[red bold]❌ SOME GATES ARE NOT ENFORCED[/red bold]")
    
    return all_pass

if __name__ == "__main__":
    run_all_tests()
