"""
Diagnostic Check: Verifies model correctness after retraining.
Run: python scripts/diagnostic_check.py
"""
from src.ml.models.corners.team_offsets import TeamOffsetManager
from src.ml.registry import ModelRegistry
from src.config.model_state import is_locked, get_model_state

print("=== DIAGNOSTIC CHECK ===")
print()

# 1. Check Lock Status
print(f"1. Lock Status: {get_model_state()}")
print()

# 2. Check Offset Manager (Case-Insensitive)
mgr = TeamOffsetManager()
udinese_offset = mgr.get_offset("Udinese", "SA")
UDINESE_offset = mgr.get_offset("UDINESE", "SA")
print("2. Offset Lookup Test:")
print(f"   - mgr.get_offset('Udinese', 'SA') = {udinese_offset}")
print(f"   - mgr.get_offset('UDINESE', 'SA') = {UDINESE_offset}")
print(f"   - Match: {udinese_offset == UDINESE_offset}")
print()

# 3. Check Model Registry
registry = ModelRegistry()
home_model = registry.get_production_model_for_league("SA", "poisson_home_base")
print("3. Model Registry Check (SA Home Goals):")
if home_model:
    print(f"   - Model: {home_model.get('name')}")
    print(f"   - Version: {home_model.get('version')}")
    print(f"   - Calibration: {home_model.get('metrics', {}).get('calibration_score', 'N/A')}")
    print(f"   - Trained At: {home_model.get('trained_at', 'N/A')}")
else:
    print("   - ERROR: No production model found!")
print()

all_passed = udinese_offset == UDINESE_offset and home_model is not None
print("=== ALL CHECKS PASSED ===" if all_passed else "=== CHECKS FAILED ===")
