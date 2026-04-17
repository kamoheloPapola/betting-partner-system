from unittest.mock import patch

import pandas as pd
import pytest

from src.cli.commands.maintenance import run_mpig_gate
from src.config.strategies import STRATEGY_REQUIREMENTS, Strategy
from src.core.exceptions import DataValidationError, ModelNotFoundError


def test_run_mpig_gate_delegates_to_verify_system():
    """
    run_mpig_gate must call verify_system with serialized match records,
    required markets, and strategy, and must return None on success.

    Asserts on the arguments passed to verify_system, not just that it was called.
    """
    df = pd.DataFrame([{"home_team": "A", "away_team": "B"}])

    with patch("src.cli.commands.maintenance.verify_system") as mock_verify:
        result = run_mpig_gate(df, Strategy.FORBIDDEN_FRUIT)

    assert result is None, "run_mpig_gate must return None when verify_system succeeds"
    mock_verify.assert_called_once_with(
        df.to_dict("records"),
        STRATEGY_REQUIREMENTS[Strategy.FORBIDDEN_FRUIT],
        Strategy.FORBIDDEN_FRUIT,
    )


def test_run_mpig_gate_wraps_integrity_failures():
    df = pd.DataFrame([{"home_team": "A", "away_team": "B"}])

    with patch(
        "src.cli.commands.maintenance.verify_system",
        side_effect=ModelNotFoundError("missing model"),
    ):
        with pytest.raises(DataValidationError, match="MPIG check failed"):
            run_mpig_gate(df, Strategy.FORBIDDEN_FRUIT)
