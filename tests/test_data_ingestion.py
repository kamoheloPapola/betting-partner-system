from unittest.mock import patch

import pandas as pd
import pytest

from src.cli.commands.maintenance import run_mpig_gate
from src.config.strategies import Strategy
from src.core.exceptions import DataValidationError, ModelNotFoundError


def test_run_mpig_gate_delegates_to_verify_system():
    df = pd.DataFrame([{"home_team": "A", "away_team": "B"}])

    with patch("src.cli.commands.maintenance.verify_system") as mock_verify:
        assert run_mpig_gate(df, Strategy.FORBIDDEN_FRUIT) is None
        mock_verify.assert_called_once()


def test_run_mpig_gate_wraps_integrity_failures():
    df = pd.DataFrame([{"home_team": "A", "away_team": "B"}])

    with patch(
        "src.cli.commands.maintenance.verify_system",
        side_effect=ModelNotFoundError("missing model"),
    ):
        with pytest.raises(DataValidationError, match="MPIG check failed"):
            run_mpig_gate(df, Strategy.FORBIDDEN_FRUIT)
