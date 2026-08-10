from __future__ import annotations

import ast

from scripts import nightly_pipeline


class RecordingAlerter:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def send_alert(self, message, context=None, severity="WARNING"):
        self.calls.append(
            {
                "message": message,
                "context": context,
                "severity": severity,
            }
        )
        return True


def test_publish_failure_alerts_critical_with_real_error() -> None:
    alerter = RecordingAlerter()

    def fail_publish():
        raise RuntimeError("GitHub upload returned HTTP 403")

    result = nightly_pipeline._publish_processed_data_step(
        alerter=alerter,
        publisher=fail_publish,
    )

    assert result.name == "publish_processed_data"
    assert result.successful is False
    assert result.returncode == 1
    assert alerter.calls == [
        {
            "message": (
                "Nightly processed-data publication failed: "
                "RuntimeError: GitHub upload returned HTTP 403"
            ),
            "context": {
                "step": "publish_processed_data",
                "error": "RuntimeError: GitHub upload returned HTTP 403",
            },
            "severity": "CRITICAL",
        }
    ]


def test_publish_success_is_a_completed_fatal_step() -> None:
    alerter = RecordingAlerter()

    result = nightly_pipeline._publish_processed_data_step(
        alerter=alerter,
        publisher=lambda: {"name": "processed-data.tar.gz"},
    )

    assert result.name == "publish_processed_data"
    assert result.successful is True
    assert result.returncode == 0
    assert alerter.calls == []


def test_publish_step_is_not_listed_as_non_fatal() -> None:
    source = nightly_pipeline.Path(nightly_pipeline.__file__).read_text(
        encoding="utf-8"
    )
    non_fatal_block = source.split("NON_FATAL_STEPS = {", 1)[1].split("}", 1)[0]

    assert '"publish_processed_data"' not in non_fatal_block


def test_main_has_exactly_one_publish_call_site() -> None:
    source = nightly_pipeline.Path(nightly_pipeline.__file__).read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    main_function = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "main"
    )
    publish_calls = [
        node
        for node in ast.walk(main_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_publish_processed_data_step"
    ]

    assert len(publish_calls) == 1
