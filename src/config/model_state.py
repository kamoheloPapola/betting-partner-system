"""
Model State Management.

Tracks the frozen state of the modeling pipeline.
Any change after freezing must bump the version.

Usage:
    from src.config.model_state import MODEL_STATE, is_locked, require_unlocked
    
    require_unlocked()  # Raises if locked
    # ... proceed with training/tuning
"""
from typing import Final
from pathlib import Path

# Define public API
__all__ = [
    "MODEL_STATE", 
    "NEXT_LOCK_VERSION", 
    "is_locked", 
    "require_unlocked",
    "freeze",
    "get_state_file"
]

# --- STATE PERSISTENCE ---
_STATE_FILE = Path(__file__).parent.parent.parent / "data" / ".model_state"

# --- VERSION CONSTANTS ---
NEXT_LOCK_VERSION: Final[str] = "LOCKED_v14.0"
_UNLOCKED: Final[str] = "UNLOCKED"


def get_state_file() -> Path:
    """Return path to state file."""
    return _STATE_FILE


def _read_state() -> str:
    """Read current state from disk."""
    if _STATE_FILE.exists():
        return _STATE_FILE.read_text().strip()
    return _UNLOCKED


def _write_state(state: str) -> None:
    """Write state to disk."""
    _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _STATE_FILE.write_text(state)


# Dynamic property - reads from disk each time
@property
def MODEL_STATE() -> str:
    """Current model state (reads from disk)."""
    return _read_state()


# For backwards compatibility, expose as function call
def get_model_state() -> str:
    """Get current model state."""
    return _read_state()


# Make MODEL_STATE a simple variable for import compatibility
MODEL_STATE: str = _read_state()


def is_locked() -> bool:
    """Check if model pipeline is locked (starts with LOCKED_)."""
    state = _read_state()
    return state.startswith("LOCKED_")


def is_frozen() -> bool:
    """Alias for is_locked() for backwards compatibility."""
    return is_locked()


def require_unlocked(operation: str = "This operation") -> None:
    """
    Raise error if system is locked.
    
    Args:
        operation: Description of blocked operation for error message.
        
    Raises:
        RuntimeError: If MODEL_STATE starts with LOCKED_
    """
    if is_locked():
        state = _read_state()
        raise RuntimeError(
            f"{operation} is blocked. Model pipeline is LOCKED ({state}). "
            "To make changes, you must explicitly unlock the system first."
        )


def _get_git_commit() -> str:
    """Get current git commit hash."""
    import subprocess
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _get_operator() -> str:
    """Get operator identity (username or 'local')."""
    import os
    return os.environ.get("USER", os.environ.get("USERNAME", "local"))


def _log_freeze_event(action: str, version: str) -> None:
    """
    Log freeze/unlock event to audit file.
    
    Logs: timestamp, git commit, operator, action, version
    """
    import json
    from datetime import datetime
    
    log_file = _STATE_FILE.parent / ".model_state_audit.jsonl"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    entry = {
        "timestamp": datetime.now().isoformat(),
        "git_commit": _get_git_commit(),
        "operator": _get_operator(),
        "action": action,
        "version": version
    }
    
    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")


def freeze() -> None:
    """
    Freeze the model pipeline.
    
    Sets MODEL_STATE to NEXT_LOCK_VERSION.
    All training/tuning/offset operations will fail after this.
    
    Logs: timestamp, git commit hash, operator identity.
    """
    _write_state(NEXT_LOCK_VERSION)
    _log_freeze_event("FREEZE", NEXT_LOCK_VERSION)
    
    # Update module-level variable
    global MODEL_STATE
    MODEL_STATE = NEXT_LOCK_VERSION


def unlock() -> None:
    """
    Unlock the model pipeline.
    
    WARNING: This should only be done intentionally.
    Logs: timestamp, git commit hash, operator identity.
    """
    _write_state(_UNLOCKED)
    _log_freeze_event("UNLOCK", _UNLOCKED)
    
    global MODEL_STATE
    MODEL_STATE = _UNLOCKED


def get_freeze_audit_log() -> list:
    """
    Read freeze/unlock audit log.
    
    Returns list of audit entries.
    """
    import json
    
    log_file = _STATE_FILE.parent / ".model_state_audit.jsonl"
    if not log_file.exists():
        return []
    
    entries = []
    for line in log_file.read_text().strip().split("\n"):
        if line:
            entries.append(json.loads(line))
    return entries

