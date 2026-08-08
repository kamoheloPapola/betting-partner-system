from __future__ import annotations

from pathlib import Path

import yaml

from src.config import startup
from src.config.env_contract import SCHEDULER_PROCESS, required_names
from src.config.model_state import ModelStateLockedError


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _read(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def test_scheduler_image_matches_api_python_and_runs_canonical_pipeline():
    api_dockerfile = _read("Dockerfile")
    scheduler_dockerfile = _read("Dockerfile.scheduler")

    assert api_dockerfile.splitlines()[0] == "FROM python:3.11-slim"
    assert scheduler_dockerfile.splitlines()[0] == "FROM python:3.11-slim"
    assert "libgomp1" in scheduler_dockerfile
    assert (
        'CMD ["sh", "-c", "python -m src.config.startup scheduler '
        '&& python -m src.scheduler.nightly"]'
    ) in scheduler_dockerfile


def test_scheduler_compose_contract_has_one_profile_gated_trigger_target():
    compose = yaml.safe_load(_read("docker-compose.yml"))
    scheduler = compose["services"]["scheduler"]

    assert scheduler["profiles"] == ["scheduler"]
    assert scheduler["depends_on"]["api"]["condition"] == "service_healthy"
    assert scheduler["environment"]["MODELS_DIR"] == "/app/data/models"
    assert set(required_names(SCHEDULER_PROCESS)) <= set(scheduler["environment"])
    assert scheduler["volumes"] == ["./data:/app/data"]


def test_committed_cron_invokes_only_the_canonical_compose_wrapper():
    cron = _read("deploy/cron/betting-nightly")
    active_lines = [
        line
        for line in cron.splitlines()
        if line.strip() and not line.lstrip().startswith("#") and line[0].isdigit()
    ]

    assert active_lines == [
        "0 6 * * * root /bin/bash "
        "/opt/betting_partner_system/scripts/run_scheduler_container.sh >> "
        "/var/log/betting-nightly.log 2>&1"
    ]

    wrapper = _read("scripts/run_scheduler_container.sh")
    assert "flock -n 9" in wrapper
    assert "docker compose up -d --wait --wait-timeout 300 api" in wrapper
    assert "docker compose run --rm --no-deps scheduler" in wrapper
    for event in (
        "scheduler_trigger_started",
        "scheduler_trigger_completed",
        "scheduler_trigger_failed",
        "scheduler_trigger_overlap_skipped",
    ):
        assert event in wrapper


def test_obsolete_scheduler_entrypoints_are_audit_only():
    obsolete_active_paths = (
        "scripts/daily.sh",
        "scripts/run_nightly.bat",
        ".agent/workflows/daily-workflow.md",
    )
    for relative_path in obsolete_active_paths:
        assert not (PROJECT_ROOT / relative_path).exists()

    legacy_paths = (
        "legacy/scheduler/daily.sh",
        "legacy/scheduler/run_nightly.bat",
        "legacy/scheduler/daily-workflow.md",
        "legacy/scheduler/deployment-guide-08-schedule.md",
    )
    for relative_path in legacy_paths:
        contents = _read(relative_path)
        assert "LEGACY/UNUSED" in contents
        assert "superseded" in contents.lower()


def test_deployment_guide_documents_trigger_heartbeat_alert():
    guide = _read("docs/deployment-guide.md")
    normalized_guide = " ".join(guide.split())

    assert "only recurring trigger" in normalized_guide
    assert "scheduler_trigger_completed" in guide
    assert "nightly_heartbeat.json" in guide
    assert "NIGHTLY_HEARTBEAT_MAX_AGE_HOURS" in guide
    assert "dispatches a critical alert" in normalized_guide


def test_scheduler_startup_preserves_nightly_blocked_event_and_exit_3(
    monkeypatch,
    caplog,
):
    def _blocked(_process):
        raise ModelStateLockedError("locked")

    monkeypatch.setattr(startup, "run_preflight", _blocked)

    assert startup.main([SCHEDULER_PROCESS]) == 3
    assert "startup_model_state_blocked process=scheduler" in caplog.text
    assert "nightly_pipeline_blocked reason=model_state" in caplog.text
