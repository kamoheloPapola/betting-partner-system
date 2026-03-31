"""
CLI output verbosity tests.
"""

import os
import subprocess

import pytest


class TestCLIVerbosityLevels:
    """Tests for CLI verbosity control."""

    @pytest.fixture
    def run_cli(self):
        """Helper to run CLI commands with UTF-8-safe capture."""

        def _run(args: list[str], cwd: str = ".") -> tuple[str, str, int]:
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            result = subprocess.run(
                ["python", "-m", "src.cli"] + args,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=cwd,
                env=env,
            )
            return result.stdout or "", result.stderr or "", result.returncode

        return _run

    def test_quiet_mode_suppresses_pipeline_logs(self, run_cli):
        stdout, stderr, code = run_cli([
            "--verbosity", "quiet",
            "show-predictions", "--league", "PL", "--all",
        ])

        combined = stdout + stderr

        assert code == 0
        assert "Loaded PL_" not in combined
        assert "[CSS BREAKDOWN]" not in combined
        assert (
            "No matches found for filter: today" in combined
            or "Match" in combined
            or "DRIFT STOP" in combined
            or "No valid predictions generated." in combined
        )

    def test_verbose_mode_shows_info_logs(self, run_cli):
        stdout, stderr, code = run_cli([
            "--verbosity", "verbose",
            "show-predictions", "--league", "PL", "--all",
        ])

        combined = stdout + stderr

        assert code == 0
        assert "[Pipeline] Cache miss/expired for PL." in combined or "Slip Score:" in combined

    def test_debug_mode_shows_debug_logs(self, run_cli):
        stdout, stderr, code = run_cli([
            "--verbosity", "debug",
            "show-predictions", "--league", "PL", "--all",
        ])

        combined = stdout + stderr

        assert code == 0
        assert "DEBUG" in combined
        assert "Loaded PL_" in combined or "[CSS BREAKDOWN]" in combined

    def test_default_matches_quiet_log_visibility(self, run_cli):
        stdout_default, stderr_default, code_default = run_cli([
            "show-predictions", "--league", "PL", "--all",
        ])
        stdout_quiet, stderr_quiet, code_quiet = run_cli([
            "--verbosity", "quiet",
            "show-predictions", "--league", "PL", "--all",
        ])

        default_combined = stdout_default + stderr_default
        quiet_combined = stdout_quiet + stderr_quiet

        assert code_default == 0
        assert code_quiet == 0
        assert ("Loaded PL_" in default_combined) == ("Loaded PL_" in quiet_combined)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
