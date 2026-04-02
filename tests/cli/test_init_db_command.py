import sqlite3

from typer.testing import CliRunner

from src.cli.app import app
from src.db import connection as connection_module


runner = CliRunner()


def test_init_db_creates_expected_tables_in_default_sqlite_backend(tmp_path, monkeypatch):
    target_db = tmp_path / "models" / "football_intelligence.db"
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(connection_module, "DEFAULT_SQLITE_DB_PATH", target_db)
    connection_module.get_engine.cache_clear()

    result = runner.invoke(app, ["init-db"])

    try:
        assert result.exit_code == 0
        assert target_db.exists()
        with sqlite3.connect(target_db) as conn:
            table_names = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }
        assert {"resolved_predictions", "model_manifest_entries", "drift_events"}.issubset(table_names)
    finally:
        connection_module.get_engine.cache_clear()
