from pathlib import Path

import yaml

from src.config.env_contract import SCHEMA_BY_NAME


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _read(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def test_compose_contains_only_application_services_and_shared_data_bind():
    compose = yaml.safe_load(_read("docker-compose.yml"))

    assert set(compose["services"]) == {"api", "scheduler"}
    assert "volumes" not in compose
    assert compose["services"]["api"]["volumes"] == ["./data:/app/data"]
    assert compose["services"]["scheduler"]["volumes"] == ["./data:/app/data"]


def test_external_database_configuration_is_not_recognized():
    retired_names = {"DATABASE_URL", "POSTGRES_DB", "POSTGRES_USER", "POSTGRES_PASSWORD"}

    assert retired_names.isdisjoint(SCHEMA_BY_NAME)
    env_example = _read(".env.example")
    assert all(name not in env_example for name in retired_names)
    assert "psycopg" not in _read("requirements.txt").lower()
    assert "sqlalchemy" not in _read("requirements.txt").lower()


def test_manifest_resolution_and_drift_have_no_database_branches():
    for relative_path in (
        "src/ml/registry.py",
        "src/evaluation/resolve_results.py",
        "src/monitoring/drift_orchestrator.py",
    ):
        source = _read(relative_path)
        assert "database_is_configured" not in source
        assert "get_engine" not in source
        assert "src.db" not in source
        assert "sqlalchemy" not in source


def test_postgres_migration_is_audit_only():
    assert not (PROJECT_ROOT / "scripts" / "migrate_to_postgres.py").exists()
    legacy = _read("legacy/postgres/migrate_to_postgres.py")
    assert "LEGACY/UNUSED" in legacy
    assert "unsupported" in legacy.lower()
