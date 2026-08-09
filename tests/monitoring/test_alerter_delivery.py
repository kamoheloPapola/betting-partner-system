from __future__ import annotations

import logging
from urllib.error import URLError

from src.monitoring import alerter as alerter_module


class StubAlertConfig:
    def __init__(
        self,
        *,
        slack_webhook: str | None = None,
        ntfy_enabled: bool = False,
        ntfy_topic: str | None = None,
        email_enabled: bool = False,
    ) -> None:
        self.slack_webhook = slack_webhook
        self.ntfy_enabled = ntfy_enabled
        self.ntfy_topic = ntfy_topic
        self.email_enabled = email_enabled


def _alerter(monkeypatch, tmp_path, config: StubAlertConfig) -> alerter_module.Alerter:
    monkeypatch.setattr(
        alerter_module,
        "ALERT_HISTORY_FILE",
        tmp_path / "alert_history.json",
    )
    return alerter_module.Alerter(config=config)


def test_local_log_fallback_is_not_reported_as_delivery(
    monkeypatch,
    tmp_path,
    caplog,
) -> None:
    notifier = _alerter(monkeypatch, tmp_path, StubAlertConfig())

    with caplog.at_level(logging.WARNING, logger=alerter_module.__name__):
        sent = notifier.send_alert(
            "nightly pipeline failed",
            context={"source": "scheduler"},
            severity="CRITICAL",
        )

    assert sent is False
    assert notifier.history == {}
    assert not alerter_module.ALERT_HISTORY_FILE.exists()
    assert "No external alert channel delivered" in caplog.text
    assert "local log fallback only" in caplog.text


def test_failed_configured_channel_is_not_recorded_as_delivery(
    monkeypatch,
    tmp_path,
) -> None:
    notifier = _alerter(
        monkeypatch,
        tmp_path,
        StubAlertConfig(slack_webhook="https://alerts.invalid/example"),
    )
    monkeypatch.setattr(notifier, "_send_slack", lambda *_args: False)

    assert notifier.send_alert("nightly pipeline failed") is False
    assert notifier.history == {}
    assert not alerter_module.ALERT_HISTORY_FILE.exists()


def test_successful_external_delivery_is_recorded_for_deduplication(
    monkeypatch,
    tmp_path,
) -> None:
    notifier = _alerter(
        monkeypatch,
        tmp_path,
        StubAlertConfig(slack_webhook="https://alerts.invalid/example"),
    )
    calls: list[str] = []
    monkeypatch.setattr(
        notifier,
        "_send_slack",
        lambda message, _severity: calls.append(message) or True,
    )

    first = notifier.send_alert("nightly pipeline failed")
    second = notifier.send_alert("nightly pipeline failed")

    assert first is True
    assert second is False
    assert len(calls) == 1
    assert len(notifier.history) == 1
    assert alerter_module.ALERT_HISTORY_FILE.exists()


def test_partial_success_across_multiple_channels_is_delivery(
    monkeypatch,
    tmp_path,
) -> None:
    notifier = _alerter(
        monkeypatch,
        tmp_path,
        StubAlertConfig(
            slack_webhook="https://alerts.invalid/example",
            ntfy_enabled=True,
            ntfy_topic="pipeline-alerts",
        ),
    )
    calls: list[str] = []
    monkeypatch.setattr(
        notifier,
        "_send_slack",
        lambda *_args: calls.append("slack") or False,
    )
    monkeypatch.setattr(
        notifier,
        "_send_ntfy",
        lambda *_args: calls.append("ntfy") or True,
    )

    assert notifier.send_alert("nightly pipeline failed") is True
    assert calls == ["slack", "ntfy"]
    assert len(notifier.history) == 1
    assert alerter_module.ALERT_HISTORY_FILE.exists()


def test_throwing_channel_does_not_block_later_success(
    monkeypatch,
    tmp_path,
    caplog,
) -> None:
    notifier = _alerter(
        monkeypatch,
        tmp_path,
        StubAlertConfig(
            slack_webhook="https://alerts.invalid/example",
            ntfy_enabled=True,
            ntfy_topic="pipeline-alerts",
        ),
    )

    class SuccessfulResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def channel_transport(request, *, timeout):
        assert timeout == 10
        if request.full_url == "https://alerts.invalid/example":
            raise URLError("Slack transport unavailable")
        assert request.full_url == "https://ntfy.sh/pipeline-alerts"
        return SuccessfulResponse()

    monkeypatch.setattr(alerter_module, "urlopen", channel_transport)

    with caplog.at_level(logging.INFO, logger=alerter_module.__name__):
        sent = notifier.send_alert("nightly pipeline failed")

    assert sent is True
    assert "Slack webhook failed" in caplog.text
    assert "ntfy alert sent successfully" in caplog.text
    assert len(notifier.history) == 1
    assert alerter_module.ALERT_HISTORY_FILE.exists()
