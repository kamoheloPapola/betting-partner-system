"""Generate or verify .env.example from the canonical environment schema."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.env_contract import env_example_path, render_env_example


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    destination = env_example_path()
    rendered = render_env_example()

    if args.check:
        if not destination.exists() or destination.read_text(encoding="utf-8") != rendered:
            print(f"{destination} is out of date; run scripts/generate_env_example.py")
            return 1
        print(f"{destination} matches ENV_SCHEMA")
        return 0

    destination.write_text(rendered, encoding="utf-8")
    print(f"Generated {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
