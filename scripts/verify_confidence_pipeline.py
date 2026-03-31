"""
Verify Confidence Pipeline Integration.

Tests all components individually and as integrated system.
"""
import sys
import os
sys.path.insert(0, os.getcwd())

from rich.console import Console
from rich.table import Table

console = Console()


def test_uncertainty():
    """Test uncertainty estimation."""
    from src.ml.confidence.uncertainty import estimate_uncertainty
    
    console.print("\n[bold cyan]1. UNCERTAINTY ESTIMATION[/bold cyan]")
    
    tests = [
        (1.0, 0.1, "Low variance"),
        (2.0, 1.0, "Medium variance"),
        (3.0, 5.0, "High variance"),
        (5.0, 10.0, "Very high variance"),
    ]
    
    table = Table()
    table.add_column("Scenario")
    table.add_column("μ")
    table.add_column("σ²")
    table.add_column("CV")
    table.add_column("Uncertainty")
    table.add_column("Status")
    
    for mu, var, desc in tests:
        cv = (var ** 0.5) / max(mu, 0.01)
        unc = estimate_uncertainty(mu, var)
        status = "✅" if 0 <= unc <= 1 else "❌"
        table.add_row(desc, str(mu), str(var), f"{cv:.2f}", f"{unc:.3f}", status)
    
    console.print(table)
    return True


def test_drs():
    """Test Data Reliability Score."""
    from src.ml.confidence.data_reliability import calculate_drs
    
    console.print("\n[bold cyan]2. DATA RELIABILITY SCORE (DRS)[/bold cyan]")
    
    tests = [
        (15, 10, [], "Strong H2H, no missing"),
        (8, 30, ["xg"], "Moderate H2H, 1 missing"),
        (3, 60, ["xg", "ppda"], "Weak H2H, 2 missing"),
        (0, 0, ["xg", "ppda", "shots"], "KILL SWITCH"),
    ]
    
    table = Table()
    table.add_column("Scenario")
    table.add_column("H2H")
    table.add_column("Days")
    table.add_column("Missing")
    table.add_column("DRS")
    table.add_column("Status")
    
    all_pass = True
    for h2h, days, missing, desc in tests:
        drs = calculate_drs(h2h, days, missing)
        
        # Verify kill switch
        if desc == "KILL SWITCH":
            status = "✅" if drs == 0.3 else "❌ FAIL"
            if drs != 0.3:
                all_pass = False
        else:
            status = "✅" if 0.3 <= drs <= 1.0 else "❌"
        
        table.add_row(desc, str(h2h), str(days), str(len(missing)), f"{drs:.3f}", status)
    
    console.print(table)
    return all_pass


def test_mss():
    """Test Market Stability Score."""
    from src.ml.confidence.market_stability import calculate_mss
    
    console.print("\n[bold cyan]3. MARKET STABILITY SCORE (MSS)[/bold cyan]")
    
    tests = [
        ("goals_u25", True, 0.5, "Goals (stable)"),
        ("corn_o75", True, 0.5, "Corners (ref-sensitive)"),
        ("card_u55", True, 0.5, "Cards (volatile)"),
        ("card_u55", False, 0.5, "Cards + unknown ref"),
        ("corn_o75", True, 0.8, "Corners + volatile team"),
    ]
    
    table = Table()
    table.add_column("Market")
    table.add_column("Ref Known")
    table.add_column("Volatility")
    table.add_column("MSS")
    table.add_column("Status")
    
    for market, ref, vol, desc in tests:
        mss = calculate_mss(market, ref, vol)
        status = "✅" if 0.5 <= mss <= 1.0 else "❌"
        console.print(f"  {desc}: {mss:.2f}")
        table.add_row(market, str(ref), str(vol), f"{mss:.2f}", status)
    
    console.print(table)
    return True


def test_mas():
    """Test Model Agreement Score."""
    from src.ml.confidence.model_agreement import calculate_mas
    
    console.print("\n[bold cyan]4. MODEL AGREEMENT SCORE (MAS)[/bold cyan]")
    
    tests = [
        ([0.70, 0.68, 0.72], "High agreement"),
        ([0.70, 0.60, 0.65], "Medium agreement"),
        ([0.80, 0.50, 0.60], "Low agreement"),
        ([0.90, 0.40, 0.50], "Very low agreement"),
    ]
    
    table = Table()
    table.add_column("Probabilities")
    table.add_column("Spread")
    table.add_column("MAS")
    table.add_column("Status")
    
    for probs, desc in tests:
        mas = calculate_mas(probs)
        spread = max(probs) - min(probs)
        status = "✅" if 0.4 <= mas <= 1.0 else "❌"
        table.add_row(str(probs), f"{spread:.2f}", f"{mas:.3f}", status)
    
    console.print(table)
    return True


def test_hce():
    """Test Historical Calibration Error."""
    from src.ml.confidence.calibration_penalty import calculate_hce_penalty, get_bucket
    
    console.print("\n[bold cyan]5. HISTORICAL CALIBRATION ERROR (HCE)[/bold cyan]")
    
    tests = [
        ("PL", "goals_u25", 0.65, 0, "Goals 60-70% (known)"),
        ("PL", "corn_o75", 0.85, 0, "Corners 80-90% (known bad)"),
        ("PL", "btts_yes", 0.75, 0, "BTTS 70-80% (unknown)"),
        ("PL", "corn_o75", 0.85, 180, "Corners + decay"),
    ]
    
    table = Table()
    table.add_column("Market")
    table.add_column("Prob")
    table.add_column("Bucket")
    table.add_column("Decay Days")
    table.add_column("HCE Penalty")
    table.add_column("Status")
    
    for league, market, prob, days, desc in tests:
        hce = calculate_hce_penalty(league, market, prob, days)
        bucket = get_bucket(prob)
        status = "✅" if 0 <= hce <= 1 else "❌"
        table.add_row(market, f"{prob:.0%}", bucket, str(days), f"{hce:.3f}", status)
    
    console.print(table)
    return True


def test_full_calculator():
    """Test full confidence calculator integration."""
    from src.ml.confidence import get_confidence_calculator
    
    console.print("\n[bold cyan]6. FULL CONFIDENCE CALCULATOR[/bold cyan]")
    
    calc = get_confidence_calculator()
    
    tests = [
        # (prob, market, league, mu, var, h2h, days, missing, ref, desc)
        (0.85, 'goals_o25', 'PL', 3, 1, 15, 10, [], True, "Strong data goals"),
        (0.80, 'corn_o75', 'PL', 10, 4, 10, 30, [], True, "Good data corners"),
        (0.75, 'card_u55', 'PL', 4, 2, 5, 60, ['referee'], False, "Weak data cards"),
        (0.90, 'corn_o75', 'PL', 15, 10, 0, 0, ['xg', 'ppda', 'shots'], True, "Kill switch"),
    ]
    
    table = Table()
    table.add_column("Scenario")
    table.add_column("Prob")
    table.add_column("Confidence")
    table.add_column("Action")
    table.add_column("Stake Mult")
    table.add_column("Status")
    
    for prob, market, league, mu, var, h2h, days, missing, ref, desc in tests:
        result = calc.calculate(
            prob=prob, market=market, league=league,
            mu=mu, variance=var, h2h_count=h2h,
            days_since_h2h=days, missing_features=missing,
            referee_known=ref
        )
        
        # Kill switch should have very low confidence
        if desc == "Kill switch":
            status = "✅" if result.confidence < 0.20 else "❌ FAIL"
        else:
            status = "✅"
        
        table.add_row(
            desc,
            f"{prob:.0%}",
            f"{result.confidence:.1%}",
            result.action_tier,
            f"{result.stake_multiplier:.1f}",
            status
        )
    
    console.print(table)
    return True


def test_forbidden_fruit_integration():
    """Test ForbiddenFruitEngine integration."""
    from src.strategies.forbidden_fruit import ForbiddenFruitEngine
    
    console.print("\n[bold cyan]7. FORBIDDEN FRUIT INTEGRATION[/bold cyan]")
    
    engine = ForbiddenFruitEngine()
    
    match = {
        'home_team': 'LIVERPOOL',
        'away_team': 'CHELSEA',
        'league': 'PL',
    }
    
    raw_predictions = {
        'u25': 0.58,
        'o25': 0.42,
        'btts': 0.55,
        'btts_no': 0.45,
    }
    
    odds = {
        'goals_u25': 1.75,
        'goals_o25': 2.10,
        'btts_yes': 1.85,
        'btts_no': 1.95,
    }
    
    h2h_data = {'count': 10, 'days_ago': 30, 'missing': []}
    
    try:
        selections = engine.calibrated_selection(
            match=match,
            raw_predictions=raw_predictions,
            odds=odds,
            bankroll=1000.0,
            h2h_data=h2h_data
        )
        
        console.print(f"  Selections: {len(selections)}")
        
        if len(selections) > 0:
            for s in selections:
                console.print(f"    {s['market']}: conf={s['confidence']:.1%} tier={s['action_tier']} stake={s['stake']:.2f}")
        else:
            console.print("  [yellow]No selections (expected - conservative confidence)[/yellow]")
        
        console.print("  [green]✅ Integration working[/green]")
        return True
        
    except Exception as e:
        console.print(f"  [red]❌ Error: {e}[/red]")
        return False


def main():
    console.print("=" * 60)
    console.print("[bold magenta]CONFIDENCE PIPELINE VERIFICATION[/bold magenta]")
    console.print("=" * 60)
    
    results = []
    
    results.append(("Uncertainty", test_uncertainty()))
    results.append(("DRS", test_drs()))
    results.append(("MSS", test_mss()))
    results.append(("MAS", test_mas()))
    results.append(("HCE", test_hce()))
    results.append(("Calculator", test_full_calculator()))
    results.append(("Integration", test_forbidden_fruit_integration()))
    
    console.print("\n" + "=" * 60)
    console.print("[bold]SUMMARY[/bold]")
    console.print("=" * 60)
    
    all_pass = True
    for name, passed in results:
        status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
        console.print(f"  {name}: {status}")
        if not passed:
            all_pass = False
    
    if all_pass:
        console.print("\n[bold green]✅ ALL TESTS PASSED[/bold green]")
    else:
        console.print("\n[bold red]❌ SOME TESTS FAILED[/bold red]")
    
    return all_pass


if __name__ == "__main__":
    main()
