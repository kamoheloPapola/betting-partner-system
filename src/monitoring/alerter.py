"""
Alert System with De-duplication.

Sends alerts via Slack and email for CRITICAL drift events.
Includes hash-based suppression with 24h cooldown.
"""
import os
import json
import hashlib
import logging
import smtplib
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from pathlib import Path
from email.mime.text import MIMEText
from urllib.request import Request, urlopen
from urllib.error import URLError

from src.config import DATA_DIR

logger = logging.getLogger(__name__)

# Define public API
__all__ = ["Alerter", "AlertConfig"]

# --- CONFIG ---
COOLDOWN_HOURS = 24
ALERT_HISTORY_FILE = DATA_DIR / "monitoring" / ".alert_history.json"


class AlertConfig:
    """Alert configuration from environment."""
    
    @property
    def slack_webhook(self) -> Optional[str]:
        return os.environ.get("SLACK_WEBHOOK_URL")

    @property
    def ntfy_topic(self) -> Optional[str]:
        return os.environ.get("NTFY_TOPIC")

    @property
    def ntfy_enabled(self) -> bool:
        return bool(self.ntfy_topic)
    
    @property
    def email_enabled(self) -> bool:
        return bool(os.environ.get("ALERT_EMAIL_TO"))
    
    @property
    def email_to(self) -> Optional[str]:
        return os.environ.get("ALERT_EMAIL_TO")
    
    @property
    def email_from(self) -> Optional[str]:
        return os.environ.get("ALERT_EMAIL_FROM", "betting-system@localhost")
    
    @property
    def smtp_host(self) -> str:
        return os.environ.get("SMTP_HOST", "localhost")
    
    @property
    def smtp_port(self) -> int:
        return int(os.environ.get("SMTP_PORT", "25"))


class Alerter:
    """
    Alert dispatcher with de-duplication.
    
    Features:
    - Slack webhook integration
    - Email fallback (SMTP)
    - Hash-based alert suppression
    - 24h cooldown per drift type
    
    Usage:
        alerter = Alerter()
        alerter.send_alert("CRITICAL: Calibration drift detected", {
            "league": "PL",
            "drift_type": "calibration_drift",
            "ece": 0.15
        })
    """
    
    def __init__(self, config: Optional[AlertConfig] = None):
        self.config = config or AlertConfig()
        self._load_history()
    
    def _load_history(self) -> None:
        """Load alert history for de-duplication."""
        self.history: Dict[str, datetime] = {}
        
        if ALERT_HISTORY_FILE.exists():
            try:
                data = json.loads(ALERT_HISTORY_FILE.read_text())
                for key, ts in data.items():
                    self.history[key] = datetime.fromisoformat(ts)
            except Exception as e:
                logger.warning(f"Failed to load alert history: {e}")
    
    def _save_history(self) -> None:
        """Persist alert history."""
        ALERT_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {k: v.isoformat() for k, v in self.history.items()}
        ALERT_HISTORY_FILE.write_text(json.dumps(data, indent=2))
    
    def _get_alert_hash(self, message: str, context: Dict[str, Any]) -> str:
        """Generate hash for alert de-duplication."""
        key = f"{message}:{json.dumps(context, sort_keys=True)}"
        return hashlib.md5(key.encode()).hexdigest()[:12]
    
    def _is_suppressed(self, alert_hash: str) -> bool:
        """Check if alert is within cooldown window."""
        if alert_hash not in self.history:
            return False
        
        last_sent = self.history[alert_hash]
        cooldown_end = last_sent + timedelta(hours=COOLDOWN_HOURS)
        
        if datetime.now() < cooldown_end:
            remaining = cooldown_end - datetime.now()
            logger.debug(f"Alert suppressed (cooldown: {remaining})")
            return True
        
        return False
    
    def _record_alert(self, alert_hash: str) -> None:
        """Record alert for de-duplication."""
        self.history[alert_hash] = datetime.now()
        self._save_history()
    
    def send_alert(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        severity: str = "WARNING",
        force: bool = False
    ) -> bool:
        """
        Send alert via configured channels.
        
        Args:
            message: Alert message
            context: Additional context (league, drift_type, etc.)
            severity: WARNING or CRITICAL
            force: Bypass de-duplication
            
        Returns:
            True if alert was sent, False if suppressed
        """
        context = context or {}
        alert_hash = self._get_alert_hash(message, context)
        
        # De-duplication check
        if not force and self._is_suppressed(alert_hash):
            logger.info(f"Alert suppressed (duplicate within {COOLDOWN_HOURS}h): {message[:50]}...")
            return False
        
        # Build formatted message
        formatted = self._format_message(message, context, severity)
        
        # Send via available channels
        sent = False
        
        if self.config.slack_webhook:
            sent = self._send_slack(formatted, severity) or sent

        if self.config.ntfy_enabled:
            sent = self._send_ntfy(formatted, severity) or sent
        
        if self.config.email_enabled and severity == "CRITICAL":
            sent = self._send_email(message, formatted) or sent
        
        if not sent:
            # Fallback: log to file
            logger.warning(f"[ALERT-{severity}] {formatted}")
            sent = True
        
        # Record for de-duplication
        if sent:
            self._record_alert(alert_hash)
        
        return sent
    
    def _format_message(
        self,
        message: str,
        context: Dict[str, Any],
        severity: str
    ) -> str:
        """Format alert message with context."""
        lines = [
            f"🚨 *[{severity}]* {message}",
            "",
            "*Details:*"
        ]
        
        for key, value in context.items():
            lines.append(f"• {key}: `{value}`")
        
        lines.extend([
            "",
            f"_Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_"
        ])
        
        return "\n".join(lines)
    
    def _send_slack(self, message: str, severity: str) -> bool:
        """Send alert via Slack webhook."""
        try:
            payload = {
                "text": message,
                "username": "Betting System Alert",
                "icon_emoji": ":rotating_light:" if severity == "CRITICAL" else ":warning:"
            }
            
            req = Request(
                self.config.slack_webhook,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"}
            )
            
            with urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    logger.info("Slack alert sent successfully")
                    return True
                    
        except URLError as e:
            logger.error(f"Slack webhook failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected Slack error: {e}")
        
        return False

    def _send_ntfy(self, message: str, severity: str) -> bool:
        """Send alert via ntfy."""
        try:
            topic = self.config.ntfy_topic
            if not topic:
                return False

            req = Request(
                f"https://ntfy.sh/{topic}",
                data=message.encode("utf-8"),
                headers={
                    "Title": f"Betting System {severity}",
                    "Priority": "high" if severity == "CRITICAL" else "default",
                    "Content-Type": "text/plain; charset=utf-8",
                },
            )

            with urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    logger.info("ntfy alert sent successfully")
                    return True

            logger.error("ntfy alert failed with status %s", resp.status)
        except URLError as e:
            logger.error(f"ntfy alert failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected ntfy error: {e}")

        return False
    
    def _send_email(self, subject: str, body: str) -> bool:
        """Send alert via email (SMTP)."""
        try:
            msg = MIMEText(body.replace("*", "").replace("`", ""))
            msg["Subject"] = f"[ALERT] {subject}"
            msg["From"] = self.config.email_from
            msg["To"] = self.config.email_to
            
            with smtplib.SMTP(
                self.config.smtp_host,
                self.config.smtp_port,
                timeout=10,
            ) as server:
                server.send_message(msg)
            
            logger.info(f"Email alert sent to {self.config.email_to}")
            return True
            
        except Exception as e:
            logger.error(f"Email alert failed: {e}")
            return False
    
    def clear_history(self) -> None:
        """Clear alert history (for testing)."""
        self.history = {}
        if ALERT_HISTORY_FILE.exists():
            ALERT_HISTORY_FILE.unlink()
