"""Best-effort HTTPS reporting of one canonical run's outcome."""
from http.client import HTTPSConnection
import os
from queue import Empty, Queue
import re
from socket import SHUT_RDWR
import ssl
from threading import Event, Thread
from urllib.parse import urlsplit

from .clock import canonical

ENV_VAR = "CANONICAL_HEARTBEAT_URL"
TIMEOUT = 10
WARNINGS = {"FAILED": "HEARTBEAT_FAILED", "NOT_CONFIGURED": "HEARTBEAT_NOT_CONFIGURED"}


def _endpoint():
    value = os.environ.get(ENV_VAR, "")
    if re.search(r"[^\x21-\x7e]|[\\#]|%(?![0-9a-fA-F]{2})", value):
        return None
    try:
        url = urlsplit(value)
        if (url.scheme == "https" and re.fullmatch(r"(?:[a-zA-Z0-9.-]+|\[[a-fA-F0-9:.]+\])(?::[0-9]+)?", url.netloc)
                and url.port != 0 and not url.hostname.startswith(".") and ".." not in url.hostname):
            return url
    except ValueError:
        pass
    return None


def _https_get(url, body, timeout):
    # Bound the whole attempt, including DNS, the TLS handshake and response headers.
    result = Queue(maxsize=1)
    cancelled = Event()
    connection = None

    def request():
        nonlocal connection
        status = None
        try:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            context.load_default_certs()
            connection = HTTPSConnection(url.hostname, url.port, timeout=timeout, context=context)
            try:
                connection.connect()
                if not cancelled.is_set():
                    target = url.path or "/"
                    if url.query:
                        target += "?" + url.query
                    connection.request("GET", target, body=body,
                                       headers={"Content-Type": "application/json", "Cache-Control": "no-store"})
                    with connection.getresponse() as response:
                        status = response.status
            finally:
                connection.close()
        except BaseException as exc:
            status = type(exc)  # Keep URL and server text in the worker.
        result.put(status)

    Thread(target=request, daemon=True, name="canonical-heartbeat").start()
    try:
        return result.get(timeout=timeout)
    except Empty:
        cancelled.set()
        sock = connection.sock if connection is not None else None
        if sock is not None:
            try:
                sock.shutdown(SHUT_RDWR)
            except OSError:
                pass
        return None


def send(run_id, state, withheld_leagues, *, blocked=False, transport=None):
    kind = "FAILURE"
    error_type = None
    try:
        if state == "SUCCESS" and not withheld_leagues and not blocked:
            kind = "SUCCESS"
        url = _endpoint()
        if url is None:
            return {"state": "DISABLED", "kind": kind}
        if kind == "FAILURE":
            url = url._replace(path=url.path.rstrip("/") + "/fail")
        body = canonical({"run_id": run_id, "state": state, "withheld_leagues": withheld_leagues}).encode("utf-8")
        get = _https_get if transport is None else transport
        for _ in range(2):
            try:
                status = get(url, body, TIMEOUT)
            except Exception as exc:
                status = type(exc)
            if isinstance(status, type) and issubclass(status, BaseException):
                error_type = status.__name__
                if not issubclass(status, Exception):
                    break
                continue
            if status is None:
                error_type = "TimeoutError"
                continue
            error_type = None
            if 200 <= status < 300:
                return {"state": "SENT", "kind": kind}
            if status != 429 and not 500 <= status < 600:
                break
    except BaseException as exc:
        error_type = type(exc).__name__
    delivery = {"state": "FAILED", "kind": kind}
    if error_type:
        delivery["error_type"] = error_type
    return delivery


class Attempt:
    """Share one outcome, including its transport retry, across the run and its CLI."""

    def __init__(self, transport=None):
        self.run_id = None
        self._transport = transport
        self._delivery = None

    def report(self, state, withheld_leagues=(), *, blocked=False):
        if self._delivery is None:
            self._delivery = send(self.run_id, state, withheld_leagues, blocked=blocked, transport=self._transport)
        return self._delivery


def status(receipt=None):
    if receipt:
        if receipt["state"] == "DISABLED":
            return {"state": "NOT_CONFIGURED", "kind": receipt["kind"]}
        return receipt
    return {"state": "NOT_EVALUATED" if _endpoint() else "NOT_CONFIGURED", "kind": None}
