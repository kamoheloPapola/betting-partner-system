"""
Strategy Domain.

Core decision engines and safety guardrails for prediction filtering and slip construction.
"""
from src.strategies.css_math import calculate_css, count_correlated_pairs
from src.strategies.edge_engine import EdgeEngine
from src.strategies.forbidden_fruit import ForbiddenFruitEngine, ForbiddenFruitEvaluator
from src.strategies.selection_gate import SelectionGate
from src.strategies.drift_guard import DriftGuardrail
from src.strategies.slip_builder import ForbiddenFruitSlipBuilder

__all__ = [
    # Engines
    "ForbiddenFruitEngine",
    "ForbiddenFruitEvaluator",
    "ForbiddenFruitSlipBuilder",
    "EdgeEngine",
    "SelectionGate",
    
    # Safety
    "DriftGuardrail",
    
    # Utilities
    "calculate_css",
    "count_correlated_pairs"
]
