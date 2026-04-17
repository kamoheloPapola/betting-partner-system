import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_cli_registration_does_not_import_sqlalchemy(tmp_path):
    env = os.environ.copy()
    env["PREDICTION_SYSTEM_LOG_FILE"] = str(tmp_path / "prediction_system.log")
    script = (
        "import sys; "
        "import src.cli.app; "
        "print(any(name == 'sqlalchemy' or name.startswith('sqlalchemy.') for name in sys.modules))"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"
