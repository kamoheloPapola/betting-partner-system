"""
CLI Application - Single Entry Point.

Initializes the Typer app, registers all commands, and applies
platform-specific fixes (e.g., Windows UTF-8 encoding).

Usage:
    python -m src.cli show-predictions --date today
"""
import logging
import os
import sys
import io
from enum import Enum
from pathlib import Path

import typer

# Define public API
__all__ = ["app", "Verbosity", "main"]

logger = logging.getLogger(__name__)


class Verbosity(str, Enum):
    """Output verbosity levels for CLI commands."""
    QUIET = "quiet"      # Tables + Slip only
    VERBOSE = "verbose"  # + CSS scores, gates, warnings
    DEBUG = "debug"      # Full pipeline logs


# --- TYPER APP ---
from src.cli.base import app


from rich.logging import RichHandler
from rich.console import Console

# --- GLOBAL CONSOLE ---
# Standard console used across the CLI for consistent formatting
console = Console()

@app.callback()
def callback(
    verbosity: Verbosity = typer.Option(
        Verbosity.QUIET,
        "--verbosity",
        "-V",
        case_sensitive=False,
        help="Output verbosity level: quiet, verbose, debug"
    )
) -> None:
    """
    Global CLI configuration.
    
    Controls logging verbosity for all commands.
    """
    # Configure logging based on verbosity
    level = logging.WARNING
    if verbosity == Verbosity.DEBUG:
        level = logging.DEBUG
    elif verbosity == Verbosity.VERBOSE:
        level = logging.INFO
        
    logging.getLogger().setLevel(level)


# --- WINDOWS ENCODING FIX ---
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8')
        logger.debug("Applied Windows UTF-8 encoding fix (reconfigure)")
    except (AttributeError, io.UnsupportedOperation):
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
            logger.debug("Applied Windows UTF-8 encoding fix (TextIOWrapper fallback)")
        except Exception:
            logger.warning("Could not apply Windows encoding fix")


# --- LOGGING CONFIGURATION ---
DEFAULT_LOG_FILE = Path(__file__).parent.parent.parent / "prediction_system.log"
LOG_FILE = Path(os.getenv("PREDICTION_SYSTEM_LOG_FILE", str(DEFAULT_LOG_FILE)))
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s', # RichHandler handles the rest
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        RichHandler(
            console=console, 
            rich_tracebacks=True,
            show_path=False, # cleaner look
            markup=True
        )
    ]
)


# --- REGISTER COMMANDS ---
# Import command modules to register them with the Typer app
import src.cli.commands.training
import src.cli.commands.prediction
import src.cli.commands.data
import src.cli.commands.evaluation
import src.cli.commands.maintenance

logger.debug("CLI commands registered successfully")


def main() -> None:
    """Run the CLI application with graceful error handling."""
    try:
        app()
    except KeyboardInterrupt:
        print("\n[!] Operation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        logger.exception("Fatal error in CLI")
        print(f"\n[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
