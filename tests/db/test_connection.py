from src.db import connection as connection_module


def test_get_engine_defaults_to_sqlite_path(tmp_path, monkeypatch):
    target_db = tmp_path / "models" / "football_intelligence.db"
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(connection_module, "DEFAULT_SQLITE_DB_PATH", target_db)
    connection_module.get_engine.cache_clear()

    engine = connection_module.get_engine()
    try:
        assert engine.dialect.name == "sqlite"
        assert str(engine.url).endswith("/football_intelligence.db")
        assert target_db.parent.exists()
    finally:
        engine.dispose()
        connection_module.get_engine.cache_clear()


def test_get_engine_uses_database_url_when_present(monkeypatch):
    captured: dict[str, object] = {}

    class FakeEngine:
        def __init__(self, url: str):
            self.url = url

        def dispose(self) -> None:
            return None

    def fake_create_engine(url: str, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        return FakeEngine(url)

    monkeypatch.setenv("DATABASE_URL", "postgres://user:pass@localhost:5432/football")
    monkeypatch.setattr(connection_module, "create_engine", fake_create_engine)
    connection_module.get_engine.cache_clear()

    engine = connection_module.get_engine()
    try:
        assert engine.url == "postgresql+psycopg2://user:pass@localhost:5432/football"
        assert captured["url"] == "postgresql+psycopg2://user:pass@localhost:5432/football"
        assert captured["kwargs"] == {"connect_args": {}, "pool_pre_ping": True}
    finally:
        engine.dispose()
        connection_module.get_engine.cache_clear()
