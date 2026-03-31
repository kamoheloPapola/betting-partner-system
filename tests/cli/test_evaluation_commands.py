"""
Tests for CLI Evaluation Commands.

Unit tests for model health checks, date validation, and Wilson CI calculation.
"""
from unittest.mock import Mock

import pytest

from src.cli.commands.evaluation import (
    ModelHealthStatus,
    _calculate_wilson_ci,
    _check_model_health,
    _validate_date_format,
)


def test_check_model_health_all_ok():
    """Verify health check with all models present."""
    mock_registry = Mock()
    
    # Mock all models as present and sufficient
    mock_registry.get_production_model_for_league.return_value = {
        'league': 'PL',
        'train_size': 1500,
        'test_size': 500
    }
    
    status = _check_model_health(
        mock_registry, 
        'PL', 
        {'goals': 2000, 'corners': 1500, 'cards': 1000}
    )
    
    assert status.league == 'PL'
    assert status.health == "[green]HEALTHY[/green]"
    assert len(status.flags) == 0

def test_check_model_health_global_fallback():
    """Verify detection of global fallback."""
    mock_registry = Mock()
    
    # Mock goals using global
    mock_registry.get_production_model_for_league.side_effect = [
        {'league': 'Global', 'train_size': 5000, 'test_size': 1000},  # Goals
        {'league': 'PL', 'train_size': 1500, 'test_size': 500},       # Corners
        {'league': 'PL', 'train_size': 1000, 'test_size': 200}        # Cards
    ]
    
    status = _check_model_health(mock_registry, 'PL', {'goals': 2000, 'corners': 1500, 'cards': 1000})
    
    assert "Global-Goals" in status.flags
    assert status.health == "[yellow]PARTIAL[/yellow]"

def test_validate_date_format_valid():
    """Verify valid date passes."""
    result = _validate_date_format("20250107")
    assert result == "20250107"

def test_validate_date_format_invalid():
    """Verify invalid format raises error."""
    with pytest.raises(ValueError, match="Invalid date format"):
        _validate_date_format("2025-01-07")  # Wrong format

def test_calculate_wilson_ci():
    """Verify confidence interval calculation."""
    lower, upper = _calculate_wilson_ci(0.65, 100)
    
    # 65% accuracy with 100 samples should have ~±10% margin
    assert 0.55 < lower < 0.60
    assert 0.70 < upper < 0.75
