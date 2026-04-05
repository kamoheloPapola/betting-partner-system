from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import DATA_DIR
from src.simulation.rl_bandit import DEFAULT_STATE_PATH

LIVE_FLAG_PATH = DATA_DIR / "rl_bandit_live.flag"


def main() -> int:
    with open(DEFAULT_STATE_PATH, encoding="utf-8") as handle:
        json.load(handle)

    LIVE_FLAG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LIVE_FLAG_PATH.write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")
    print("Bandit promoted to live. Flag written at data/rl_bandit_live.flag")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
