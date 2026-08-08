from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.config import env_contract, startup
from src.monitoring import alerter as alerter_module


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _satisfied_environment(process: str) -> dict[str, str]:
    values = {name: "configured" for name in env_contract.required_names(process)}
    values["ARTIFACT_SIGNING_KEY"] = "x" * 32
    values["CLI_AUTH_TOKENS"] = '{"test-token":"admin"}'
    return values


REQUIRED_CASES = [
    (process, name)
    for process in sorted(env_contract.KNOWN_PROCESSES)
    for name in env_contract.required_names(process)
]


def test_schema_has_one_entry_per_environment_variable():
    names = [spec.name for spec in env_contract.ENV_SCHEMA]

    assert len(names) == 36
    assert len(names) == len(set(names))
    assert all(spec.required_for or spec.is_optional for spec in env_contract.ENV_SCHEMA)


@pytest.mark.parametrize(("process", "missing_name"), REQUIRED_CASES)
@pytest.mark.parametrize("missing_value", [None, "", "   "])
def test_each_required_variable_fails_with_its_specific_name(
    process, missing_name, missing_value
):
    values = _satisfied_environment(process)
    if missing_value is None:
        values.pop(missing_name, None)
    else:
        values[missing_name] = missing_value

    with pytest.raises(env_contract.EnvironmentContractError) as exc_info:
        env_contract.validate_environment(process, values)

    assert exc_info.value.missing == (missing_name,)
    assert missing_name in str(exc_info.value)


@pytest.mark.parametrize("process", sorted(env_contract.KNOWN_PROCESSES))
def test_full_required_schema_succeeds(process):
    env_contract.validate_environment(process, _satisfied_environment(process))


def test_preflight_validates_environment_before_reading_model_state(monkeypatch):
    calls = []
    monkeypatch.setattr(
        startup,
        "validate_environment",
        lambda process, values: calls.append(("environment", process)),
    )
    monkeypatch.setattr(
        startup,
        "_check_model_state",
        lambda process, values: calls.append(("model_state", process)),
    )

    startup.run_preflight(env_contract.API_PROCESS, {})

    assert calls == [
        ("environment", env_contract.API_PROCESS),
        ("model_state", env_contract.API_PROCESS),
    ]


def test_startup_main_returns_distinct_environment_exit(monkeypatch, caplog):
    error = env_contract.EnvironmentContractError(
        env_contract.API_PROCESS,
        ["CLI_AUTH_TOKENS"],
    )
    monkeypatch.setattr(startup, "run_preflight", lambda process: (_ for _ in ()).throw(error))

    exit_code = startup.main([env_contract.API_PROCESS])

    assert exit_code == 2
    assert "startup_environment_invalid" in caplog.text
    assert "CLI_AUTH_TOKENS" in caplog.text


def test_container_commands_put_preflight_before_network_work():
    api_command = next(
        line
        for line in (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8").splitlines()
        if line.startswith("CMD ")
    )
    scheduler_command = next(
        line
        for line in (PROJECT_ROOT / "Dockerfile.scheduler").read_text(encoding="utf-8").splitlines()
        if line.startswith("CMD ")
    )

    assert api_command.index("src.config.startup api") < api_command.index("download_models.sh")
    assert api_command.index("download_models.sh") < api_command.index("fetch_fresh_data.py")
    assert api_command.index("fetch_fresh_data.py") < api_command.index("uvicorn")
    assert scheduler_command.index("src.config.startup scheduler") < scheduler_command.index(
        "src.scheduler.nightly"
    )


def test_application_entrypoints_cannot_bypass_preflight():
    api_source = (PROJECT_ROOT / "src" / "api" / "main.py").read_text(encoding="utf-8")
    scheduler_source = (PROJECT_ROOT / "src" / "scheduler" / "nightly.py").read_text(
        encoding="utf-8"
    )

    assert api_source.index("startup_preflight_main([API_PROCESS])") < api_source.index(
        'init_sentry(component="prediction-api"'
    )
    main_body = scheduler_source[scheduler_source.index("def main(") :]
    assert main_body.index("startup_preflight_main") < main_body.index("run_nightly(")


def test_compose_healthcheck_requires_ready_nonempty_model_snapshot():
    compose = (PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert "condition: service_healthy" in compose
    assert "payload.get('status')=='ok'" in compose
    assert "payload.get('model_count',0)>0" in compose
    assert "missing_file" in compose


def test_env_example_is_generated_from_schema():
    assert env_contract.env_example_path().read_text(encoding="utf-8") == env_contract.render_env_example()
    assert "DATABASE_URL" not in env_contract.render_env_example()
    assert "POSTGRES_" not in env_contract.render_env_example()


def test_runtime_environment_reads_are_declared_in_schema():
    discovered = set()
    python_patterns = (
        re.compile(
            r"(?:os\.getenv|os\.environ\.get|get_required_env)\(\s*[\"']([A-Z][A-Z0-9_]*)[\"']"
        ),
        re.compile(r"os\.environ\[\s*[\"']([A-Z][A-Z0-9_]*)[\"']\s*\]"),
    )
    shell_pattern = re.compile(r"\$\{([A-Z][A-Z0-9_]*)[:-]")

    for root in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
        for path in root.rglob("*.py"):
            source = path.read_text(encoding="utf-8-sig")
            for pattern in python_patterns:
                discovered.update(pattern.findall(source))
    for path in (
        PROJECT_ROOT / "scripts" / "download_models.sh",
        PROJECT_ROOT / "docker-compose.yml",
    ):
        discovered.update(shell_pattern.findall(path.read_text(encoding="utf-8")))

    operating_system_values = {"USER", "USERNAME"}
    assert discovered - operating_system_values <= set(env_contract.SCHEMA_BY_NAME)


def test_legacy_environment_names_have_no_runtime_or_documentation_references():
    legacy_names = (
        "API_KEY_FOOTBALL",
        "FOOTBALL_DATA_ORG_KEY",
        "RAPIDAPI_KEY",
        "OPENWEATHERMAP_API_KEY",
    )
    paths = [
        *list((PROJECT_ROOT / "src").rglob("*.py")),
        *list((PROJECT_ROOT / "scripts").rglob("*.py")),
        *list((PROJECT_ROOT / "scripts").rglob("*.sh")),
        *list((PROJECT_ROOT / "docs").rglob("*.md")),
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "USER_GUIDE.md",
        PROJECT_ROOT / ".env.example",
    ]

    offenders = {
        str(path.relative_to(PROJECT_ROOT)): name
        for path in paths
        for name in legacy_names
        if name in path.read_text(encoding="utf-8-sig")
    }
    assert offenders == {}


def test_alert_channels_degrade_to_local_logging_when_unconfigured(
    monkeypatch, tmp_path
):
    for name in (
        "SLACK_WEBHOOK_URL",
        "NTFY_TOPIC",
        "ALERT_EMAIL_TO",
        "ALERT_EMAIL_FROM",
        "SMTP_HOST",
        "SMTP_PORT",
        "NIGHTLY_HEARTBEAT_MAX_AGE_HOURS",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        alerter_module,
        "ALERT_HISTORY_FILE",
        tmp_path / "alert-history.json",
    )
    warnings = []
    monkeypatch.setattr(
        alerter_module.logger,
        "warning",
        lambda message, *args: warnings.append(message % args if args else message),
    )

    sent = alerter_module.Alerter().send_alert("local fallback", severity="WARNING")

    assert sent is True
    assert any("[ALERT-WARNING]" in message for message in warnings)
