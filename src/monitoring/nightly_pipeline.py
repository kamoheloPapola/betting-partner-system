from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT_DIR / "scripts"

if not SCRIPTS_DIR.is_dir():
    raise RuntimeError(
        f"scripts/ directory not found at {SCRIPTS_DIR}. "
        "Check ROOT_DIR resolution in nightly_pipeline.py"
    )

__all__ = ["ROOT_DIR", "SCRIPTS_DIR"]
