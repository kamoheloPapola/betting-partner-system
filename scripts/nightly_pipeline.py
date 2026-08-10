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

if not SCRIPTS_DIR.is_dir():
    raise RuntimeError(
        f"scripts/ directory not found at {SCRIPTS_DIR}. "
        "Check ROOT_DIR resolution in nightly_pipeline.py"
    )

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


def _publish_processed_data_step(*, alerter, publisher=None) -> StepResult:
    """Publish processed data as a fatal, explicitly-alerted delivery step."""
    if publisher is None:
        from src.data.release_client import publish_processed_data_release

        publisher = publish_processed_data_release

    try:
        uploaded_asset = publisher()
    except Exception as exc:
        error_detail = f"{type(exc).__name__}: {exc}"
        logger.exception("Processed-data Release publication failed: %s", error_detail)
        alerter.send_alert(
            f"Nightly processed-data publication failed: {error_detail}",
            context={
                "step": "publish_processed_data",
                "error": error_detail,
            },
            severity="CRITICAL",
        )
        return StepResult(
            name="publish_processed_data",
            command=("github-release", "data-nightly", "processed-data.tar.gz"),
            returncode=1,
            successful=False,
            completed=True,
        )

    logger.info(
        "Processed data published to data-nightly (%s)",
        uploaded_asset.get("name", "processed-data.tar.gz"),
    )
    return StepResult(
        name="publish_processed_data",
        command=("github-release", "data-nightly", "processed-data.tar.gz"),
        returncode=0,
        successful=True,
        completed=True,
    )


def main() -> int:
    from src.strategies.drift_guard import DriftGuardrail

    started_at = datetime.now(timezone.utc)
    step_results: list[StepResult] = []
    promotion_happened = False
    fetch_command = (
        sys.executable,
        str(SCRIPTS_DIR / "fetch_fresh_data.py"),
    )

    fetch_result = _run_step("fetch_fresh_data", fetch_command, success_codes={0})
    if not fetch_result.successful:
        logger.warning("Data fetch failed on first attempt - retrying in 60 seconds")
        import time as _time
        _time.sleep(60)
        fetch_result = _run_step("fetch_fresh_data", fetch_command, success_codes={0})
        if not fetch_result.successful:
            logger.warning("Data fetch failed after retry - continuing with existing CSVs")
        else:
            logger.info("Data fetch succeeded on retry")
    step_results.append(fetch_result)

    enrichment_command = (
        sys.executable,
        str(SCRIPTS_DIR / "fetch_stats_enrichment.py"),
    )
    enrichment_result = _run_step(
        "fetch_stats_enrichment",
        enrichment_command,
        success_codes={0},
    )
    step_results.append(enrichment_result)

    drift = DriftGuardrail()
    drift_status = drift.check_drift()
    logger.info("Nightly pipeline drift status: %s", drift_status)
    from src.monitoring.alerter import Alerter

    alerter = Alerter()

    if drift_status == "STOP":
        logger.error(
            "NIGHTLY ABORT: DriftGuard is STOP — skipping training and promotion."
        )
        alerter.send_alert(
            "Nightly pipeline drift status is STOP",
            context={"date": started_at.isoformat()},
            severity="CRITICAL",
        )
        # No completed run exists to deliver; keep the current Release instead
        # of republishing potentially stale local CSVs from this aborted run.
        _print_summary(
            started_at=started_at,
            step_results=[],
            promotion_happened=False,
        )
        return 2

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
    cleanup_command = (
        sys.executable,
        str(SCRIPTS_DIR / "cleanup_old_models.py"),
        "--keep-versions",
        "3",
        "--confirm",
    )
    prediction_leagues = ["PL", "BL1", "FL1", "SA", "PD"]
    resolve_command = (
        sys.executable,
        "-m",
        "src.cli",
        "resolve-predictions",
    )
    performance_report_command = (
        sys.executable,
        str(SCRIPTS_DIR / "generate_performance_report.py"),
    )

    train_result = _run_step("train_batch", train_command, success_codes={0})
    step_results.append(train_result)
    if train_result.successful:
        try:
            from src.monitoring.drift_orchestrator import get_drift_orchestrator

            orchestrator = get_drift_orchestrator()
            orchestrator.reload_baselines()
            logger.info("Drift baselines reloaded after training.")
        except Exception as exc:
            logger.warning("Could not reload drift baselines: %s", exc)

    step_results.append(_run_step("update_bandit", update_command, success_codes={0}))
    shadow_result = _run_step("shadow_comparison", shadow_command, success_codes={0, 1, 2})
    step_results.append(shadow_result)

    if shadow_result.returncode == 0:
        promote_result = _run_step("promote_bandit", promote_command, success_codes={0})
        step_results.append(promote_result)
        promotion_happened = promote_result.successful

    post_drift = DriftGuardrail()
    post_status = post_drift.check_drift()
    if post_status == "OK":
        logger.info("Post-training drift check: OK — system is clean.")
    else:
        logger.warning("Post-training drift check: %s — monitor closely.", post_status)
    step_results.append(
        _run_step("cleanup_old_models", cleanup_command, success_codes={0})
    )
    for league in prediction_leagues:
        show_predictions_command = (
            sys.executable,
            "-m",
            "src.cli",
            "show-predictions",
            "--league",
            league,
            "--all",
            "--date",
            "all",
        )
        show_result = _run_step(
            f"show_predictions_{league}",
            show_predictions_command,
            success_codes={0},
        )
        step_results.append(show_result)
        if not show_result.successful:
            logger.warning(
                "show-predictions failed for %s - continuing to next league",
                league,
            )
    step_results.append(
        _run_step("resolve_predictions", resolve_command, success_codes={0})
    )
    for league in prediction_leagues:
        drift_check_command = (
            sys.executable,
            "-m",
            "src.cli",
            "check-drift",
            "--league",
            league,
            "--lookback",
            "120",
        )
        drift_result = _run_step(
            f"check_drift_{league}",
            drift_check_command,
            success_codes={0},
        )
        step_results.append(drift_result)
    step_results.append(
        _run_step(
            "generate_performance_report",
            performance_report_command,
            success_codes={0},
        )
    )
    step_results.append(_publish_processed_data_step(alerter=alerter))
    # Data refresh and drift checks are non-fatal; keep them in the summary.
    NON_FATAL_STEPS = {
        "fetch_fresh_data",
        "fetch_stats_enrichment",
        "check_drift_PL",
        "check_drift_BL1",
        "check_drift_FL1",
        "check_drift_SA",
        "check_drift_PD",
    }
    any_failed = any(
        not step.successful for step in step_results
        if step.name not in NON_FATAL_STEPS
    )
    _print_summary(
        started_at=started_at,
        step_results=step_results,
        promotion_happened=promotion_happened,
    )
    performance_report_path = ROOT_DIR / "data" / "eval" / "performance_report.json"
    notification_lines = ["Leagues trained: PL, BL1, FL1, SA, PD"]
    performance_report: dict[str, object] = {}
    if performance_report_path.exists():
        try:
            import json

            performance_report = json.loads(
                performance_report_path.read_text(encoding="utf-8")
            )
        except Exception as exc:
            logger.warning(
                "Could not read performance report for completion notification: %s",
                exc,
            )

    summary = performance_report.get("summary", {}) if isinstance(performance_report, dict) else {}
    overall_win_rate = None
    overall_roi = None
    if isinstance(summary, dict):
        overall_win_rate = summary.get("overall_win_rate")
        overall_roi = summary.get("overall_roi", summary.get("roi_pct"))

    if isinstance(overall_win_rate, (int, float)):
        notification_lines.append(f"Win rate: {overall_win_rate:.1%}")
    if isinstance(overall_roi, (int, float)):
        notification_lines.append(f"ROI: {overall_roi:.2f}%")

    if alerter.config.ntfy_enabled and alerter.config.ntfy_topic:
        try:
            from urllib.error import URLError
            from urllib.request import Request, urlopen

            req = Request(
                f"https://ntfy.sh/{alerter.config.ntfy_topic}",
                data="\n".join(notification_lines).encode("utf-8"),
                headers={
                    "Title": "Betting Partner - Nightly Complete",
                    "Priority": "default",
                    "Tags": "white_check_mark",
                    "Content-Type": "text/plain; charset=utf-8",
                },
            )

            with urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    logger.info("ntfy completion notification sent successfully")
                else:
                    logger.error(
                        "ntfy completion notification failed with status %s",
                        resp.status,
                    )
        except URLError as exc:
            logger.error("ntfy completion notification failed: %s", exc)
        except Exception as exc:
            logger.error("Unexpected ntfy completion error: %s", exc)
    if any_failed:
        alerter.send_alert(
            "Nightly pipeline completed with failures",
            context={
                "failed_steps": [s.name for s in step_results if not s.successful],
                "date": started_at.isoformat(),
            },
            severity="CRITICAL",
        )
    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
