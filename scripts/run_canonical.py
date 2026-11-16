"""Canonical job launcher: absolute project import root; no credentials or dotenv."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.canonical.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
