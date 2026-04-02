from typer.testing import CliRunner

from src.cli.app import app


runner = CliRunner()


def test_sync_models_push_reads_env_and_calls_registry(monkeypatch):
    calls: list[tuple[str, str]] = []

    class FakeRegistry:
        def push_to_s3(self, bucket: str, prefix: str) -> int:
            calls.append((bucket, prefix))
            return 2

    monkeypatch.setenv("S3_BUCKET", "bucket-name")
    monkeypatch.setenv("S3_PREFIX", "models/prod")
    monkeypatch.setattr("src.ml.registry.ModelRegistry", FakeRegistry)

    result = runner.invoke(app, ["sync-models", "--push"])

    assert result.exit_code == 0
    assert calls == [("bucket-name", "models/prod")]
    assert "Uploaded 2 model artifacts" in result.stdout
