from __future__ import annotations

import logging
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StepResult:
    name: str
    command: tuple[str, ...]
    returncode: int
    successful: bool
    completed: bool


def _run_step(
    name: str,
    command: Sequence[str],
    *,
    success_codes: set[int],
) -> StepResult:
    result = subprocess.run(
        list(command),
        check=False,
        cwd=str(ROOT_DIR),
    )
    returncode = int(result.returncode)
    successful = returncode in success_codes
    if not successful:
        logger.warning("WARNING: %s returned code %s", name, returncode)
    return StepResult(
        name=name,
        command=tuple(str(part) for part in command),
        returncode=returncode,
        successful=successful,
        completed=True,
    )


def _print_summary(
    *,
    started_at: datetime,
    step_results: list[StepResult],
    promotion_happened: bool,
) -> None:
    print("\n" + "=" * 60)
    print("NIGHTLY PIPELINE SUMMARY")
    print("=" * 60)
    print(f"Date: {started_at.isoformat()}")
    print("Steps completed:")
    for step in step_results:
        if step.successful:
            status = "OK" if step.returncode == 0 else f"OK (rc={step.returncode})"
        else:
            status = f"WARN (rc={step.returncode})"
        print(f"- {step.name}: {status}")
    print(f"Promotion happened: {'YES' if promotion_happened else 'NO'}")
    print("=" * 60)


def main() -> int:
    started_at = datetime.now(timezone.utc)
    step_results: list[StepResult] = []
    promotion_happened = False

    train_command = (
        sys.executable,
        str(SCRIPTS_DIR / "train_batch.py"),
        "--leagues",
        "PL",
        "BL1",
        "FL1",
        "SA",
        "PD",
    )
    update_command = (
        sys.executable,
        str(SCRIPTS_DIR / "update_bandit.py"),
    )
    shadow_command = (
        sys.executable,
        str(SCRIPTS_DIR / "shadow_comparison.py"),
        "--days",
        "30",
    )
    promote_command = (
        sys.executable,
        str(SCRIPTS_DIR / "promote_bandit.py"),
    )

    step_results.append(_run_step("train_batch", train_command, success_codes={0}))
    step_results.append(_run_step("update_bandit", update_command, success_codes={0}))
    shadow_result = _run_step("shadow_comparison", shadow_command, success_codes={0, 1, 2})
    step_results.append(shadow_result)

    if shadow_result.returncode == 0:
        promote_result = _run_step("promote_bandit", promote_command, success_codes={0})
        step_results.append(promote_result)
        promotion_happened = promote_result.successful

    any_failed = any(not step.successful for step in step_results)
    _print_summary(
        started_at=started_at,
        step_results=step_results,
        promotion_happened=promotion_happened,
    )
    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
