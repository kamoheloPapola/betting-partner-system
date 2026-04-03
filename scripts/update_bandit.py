from __future__ import annotations

import logging

from src.simulation.rl_bandit import ContextualBandit, DEFAULT_STATE_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> int:
    bandit = ContextualBandit(
        state_path=DEFAULT_STATE_PATH,
        auto_load=True,
        auto_bootstrap=False,
    )
    ece_by_context = bandit.refresh_from_resolved_predictions()
    bandit.save(DEFAULT_STATE_PATH)
    logger.info(
        "Updated RL bandit state for %s context bucket(s) at %s",
        len(ece_by_context),
        DEFAULT_STATE_PATH,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
