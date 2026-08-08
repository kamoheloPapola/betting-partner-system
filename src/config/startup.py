"""Container startup preflight: environment first, model state second."""

from __future__ import annotations

import argparse
import logging
import os
from typing import Mapping, Optional, Sequence

from src.config.env_contract import (
    API_PROCESS,
    KNOWN_PROCESSES,
    SCHEDULER_PROCESS,
    EnvironmentContractError,
    validate_environment,
)
from src.monitoring.alert_pipeline import dispatch_alert

logger = logging.getLogger(__name__)


def _check_model_state(process: str, environ: Mapping[str, str]) -> None:
    # Deliberately import only after environment validation succeeds.
    from src.config.model_state import ModelStateError, get_model_state, require_unlocked

    if process == API_PROCESS:
        state = get_model_state()
        skip = str(environ.get("SKIP_MODEL_LOCK_CHECK", "")).strip().lower() == "true"
        if skip:
            logger.warning(
                "startup_model_state_bypassed process=api state=%s",
                state,
            )
            return
        if not state.startswith("LOCKED_"):
            raise ModelStateError(
                f"API startup requires a locked model state; current state is {state}."
            )
        return

    if process == SCHEDULER_PROCESS:
        require_unlocked("Nightly scheduler startup")


def run_preflight(
    process: str,
    environ: Optional[Mapping[str, str]] = None,
) -> None:
    values = os.environ if environ is None else environ
    validate_environment(process, values)
    logger.info("startup_environment_valid process=%s", process)
    _check_model_state(process, values)
    logger.info("startup_model_state_valid process=%s", process)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate process startup prerequisites.")
    parser.add_argument("process", choices=sorted(KNOWN_PROCESSES))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        run_preflight(args.process)
    except EnvironmentContractError as exc:
        logger.error(
            "startup_environment_invalid process=%s missing=%s",
            exc.process,
            ",".join(exc.missing),
        )
        return 2
    except Exception as exc:
        from src.config.model_state import ModelStateError

        if not isinstance(exc, ModelStateError):
            raise
        logger.error(
            "startup_model_state_blocked process=%s error_type=%s detail=%s",
            args.process,
            type(exc).__name__,
            exc,
        )
        if args.process == SCHEDULER_PROCESS:
            logger.error(
                "nightly_pipeline_blocked reason=model_state error_type=%s detail=%s",
                type(exc).__name__,
                exc,
            )
            dispatch_alert(
                source="nightly_pipeline_blocked",
                severity="CRITICAL",
                message="Nightly pipeline blocked by model state",
                context={
                    "process": args.process,
                    "error_type": type(exc).__name__,
                    "detail": str(exc),
                    "exit_code": 3,
                },
            )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
