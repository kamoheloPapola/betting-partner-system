"""D1-D4 offline fixtures only; no genuine credentials or provider evidence."""
from __future__ import annotations

import copy
import http.client
import io
import json
import logging
import os
from pathlib import Path
import secrets
import traceback
import unittest
from unittest.mock import patch
import uuid

import requests
from urllib3._collections import HTTPHeaderDict
from urllib3.response import HTTPResponse
from urllib3.connectionpool import HTTPSConnectionPool

from src.evaluation import bl1_capture_transport as transport
from src.evaluation import future_lineage_context as contract

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "tmp/BL1-transport-binding-implementation-20261024-01/remediation-01"


class TransportTests(unittest.TestCase):
    def setUp(self):
        base = Path(os.environ["BL1_TRANSPORT_TEST_ROOT"]).absolute()
        self.assertTrue(base.is_relative_to(EVIDENCE))
        self.base = base
        self.directories = []
        self.fresh()
        self.credential = secrets.token_urlsafe(32)
        self.calls = 0

    def tearDown(self):
        self.no_credentials_retained()

    def fresh(self):
        self.directory = contract.safe_path(EVIDENCE,
            (self.base / uuid.uuid4().hex).relative_to(EVIDENCE).as_posix())
        self.directory.mkdir(parents=True)
        self.directories.append(self.directory)

    def capture(self, body=b'{"offline_fixture":true}', headers=None, configure=None,
                stream=None, send_error=False, **changes):
        args = dict(credential=self.credential, connect_timeout_seconds=1.0,
                    read_timeout_seconds=2.0, max_response_bytes=200000)
        args.update(changes)
        def send(adapter, prepared, **kwargs):
            self.calls += 1
            self.assertEqual(adapter.max_retries.total, 0)
            self.assertEqual(kwargs, {"stream": True, "timeout": (
                args["connect_timeout_seconds"], args["read_timeout_seconds"]),
                "verify": True, "cert": None, "proxies": {}})
            self.assertTrue(prepared.headers.get("X-Auth-Token") == self.credential)
            self.assertEqual(set(prepared.headers), {"X-Auth-Token", "Accept-Encoding"})
            self.assertEqual(prepared.headers["Accept-Encoding"], "identity")
            self.assertEqual(prepared.method, "GET")
            self.assertEqual(prepared.url, transport.PREPARED_URL)
            if send_error:
                raise OSError(self.credential)  # Must never reach retained error text.
            raw = HTTPResponse(body=stream or io.BytesIO(body),
                headers=HTTPHeaderDict(headers if headers is not None else
                    [("Content-Length", str(len(body)))]), status=200,
                preload_content=False, decode_content=False, enforce_content_length=True)
            response = requests.Response()
            response.status_code = 200
            response.request = prepared
            response.url = transport.PREPARED_URL
            response.raw = raw
            response.headers = requests.structures.CaseInsensitiveDict(raw.headers)
            if configure:
                configure(response, prepared)
            return response
        with patch.object(requests.adapters.HTTPAdapter, "send", send):
            return transport.capture_transport(ROOT, self.directory, "sample", **args)

    def rejected(self, **kwargs):
        with self.assertRaisesRegex(ValueError, "^BL1_TRANSPORT_CAPTURE_INCOMPLETE$"):
            self.capture(**kwargs)
        with self.assertRaises((ValueError, FileNotFoundError)):
            transport.read_transport_capture(ROOT, self.directory, "sample")

    def no_credentials_retained(self):
        token = self.credential.encode()
        for directory in self.directories:
            for path in directory.rglob("*"):
                if path.is_file():
                    self.assertTrue(token not in path.read_bytes(), "credential retention detected")

    def test_success_has_closed_receipt_and_existing_envelope(self):
        receipt = self.capture()
        self.assertEqual(set(receipt), transport.FIELDS)
        self.assertEqual(self.calls, 1)
        self.assertEqual(receipt, transport.read_transport_capture(ROOT, self.directory, "sample"))
        self.assertEqual(receipt, contract.seal(receipt))
        metadata = contract.read(self.directory / "sample.json")
        self.assertEqual(metadata["request"], {"url": transport.URL, "params": {"season": 2026}})
        self.assertEqual(receipt["request_sha256"], contract.fingerprint(metadata["request"]))
        self.assertEqual(receipt["raw"], metadata["archived_observation"]["raw"])
        self.assertEqual(receipt["metadata"], contract.identity(ROOT, self.directory / "sample.json"))
        self.assertEqual(receipt["framing"]["content_length"], receipt["raw"]["size_bytes"])
        self.assertFalse((self.directory / "sample.transport.pending").exists())
        self.no_credentials_retained()

    def test_budget_reservation_precedes_dispatch_and_bounds_all_capture_files(self):
        events = []
        class Budget:
            def checkpoint(self): events.append("check")
            def reserve_capture(self, bound):
                self.bound = bound
                events.append("reserve")
            def dispatch(self): events.append("dispatch")
        budget = Budget()
        self.capture(budget=budget)
        self.assertLess(events.index("reserve"), events.index("dispatch"))
        self.assertEqual(events.count("dispatch"), 1)
        actual = sum(p.stat().st_size for p in self.directory.rglob("*") if p.is_file())
        self.assertLessEqual(actual, budget.bound)

    def test_budget_denial_prevents_dispatch_and_retention(self):
        class Budget:
            def checkpoint(self): pass
            def reserve_capture(self, bound): raise ValueError("fixture storage exhausted")
            def dispatch(self): raise AssertionError("dispatch after denial")
        self.rejected(budget=Budget())
        self.assertEqual(self.calls, 0)
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_budget_deadline_interrupts_body_without_completed_capture(self):
        class Budget:
            def checkpoint(self):
                if getattr(self, "sent", False): raise ValueError("fixture deadline")
            def reserve_capture(self, bound): pass
            def dispatch(self): self.sent = True
        self.rejected(budget=Budget())
        self.assertEqual(self.calls, 1)
        self.assertFalse((self.directory / "sample.json").exists())

    def test_prepared_once_and_same_object_returned_by_adapter(self):
        original = requests.Request.prepare
        calls = []
        def prepare(request):
            result = original(request)
            calls.append(result)
            return result
        def context(response, prepared):
            self.assertTrue(len(calls) == 1 and calls[0] is prepared and response.request is prepared)
        with patch.object(requests.Request, "prepare", prepare):
            self.capture(configure=context)
        self.assertEqual(len(calls), 1)

    def test_environment_cannot_supply_proxy_auth_or_ca_settings(self):
        with patch.dict(os.environ, {"HTTPS_PROXY": "http://proxy.invalid", "REQUESTS_CA_BUNDLE": "missing-ca",
                                    "NETRC": "missing-netrc"}):
            receipt = self.capture()
        self.assertEqual(receipt["controls"], transport.CONTROLS)

    def test_empty_body_with_zero_content_length(self):
        self.assertEqual(self.capture(body=b"")["framing"]["received_entity_bytes"], 0)

    def test_explicit_identity_coding_and_exact_ceiling(self):
        r = self.capture(body=b"123", headers=[("Content-Length", "3"), ("Content-Encoding", "identity")],
                         max_response_bytes=3)
        self.assertEqual(r["framing"]["received_entity_bytes"], 3)

    def test_missing_duplicate_conflicting_and_malformed_length(self):
        cases = [[], [("Content-Length", "3"), ("Content-Length", "3")],
                 [("Content-Length", "3"), ("Content-Length", "4")],
                 [("Content-Length", "3, 3")], [("Content-Length", "-1")],
                 [("Content-Length", " 3")], [("Content-Length", "+3")],
                 [("Content-Length", "bad")]]
        for headers in cases:
            with self.subTest(headers=headers):
                self.fresh(); self.rejected(body=b"abc", headers=headers)
                self.assertFalse((self.directory / "sample.json").exists())

    def test_transfer_encoding_always_rejected(self):
        for value in ("chunked", "identity", ""):
            with self.subTest(value=value):
                self.fresh(); self.rejected(body=b"abc", headers=[("Content-Length", "3"), ("Transfer-Encoding", value)])

    def test_compressed_and_multiple_codings_rejected(self):
        for value in ("gzip", "br", "identity, gzip", ""):
            with self.subTest(value=value):
                self.fresh(); self.rejected(body=b"abc", headers=[("Content-Length", "3"), ("Content-Encoding", value)])

    def test_duplicate_identity_coding_rejected(self):
        self.rejected(body=b"abc", headers=[("Content-Length", "3"),
            ("Content-Encoding", "identity"), ("Content-Encoding", "identity")])

    def test_truncation_detected_by_real_local_urllib3_reader(self):
        self.rejected(body=b"abc", headers=[("Content-Length", "5")])
        self.assertFalse((self.directory / "sample.json").exists())

    def test_excess_entity_bytes_rejected(self):
        self.rejected(body=b"abcde", headers=[("Content-Length", "3")])

    def test_declared_size_above_ceiling_rejected_before_body(self):
        class Unreadable(io.BytesIO):
            def read(self, count=-1): raise AssertionError("must not read over-budget body")
        self.rejected(body=b"abc", stream=Unreadable(b"abc"), max_response_bytes=2)

    def test_transport_failure_is_redacted_and_not_retried(self):
        self.rejected(send_error=True)
        self.assertEqual(self.calls, 1)
        self.no_credentials_retained()

    def test_public_error_does_not_retain_secret_exception_context(self):
        with self.assertRaises(ValueError) as caught:
            self.capture(send_error=True)
        self.assertTrue(caught.exception.__context__ is None)
        self.assertTrue(caught.exception.__cause__ is None)
        self.assertTrue(self.credential not in str(caught.exception))

    def test_public_failure_component_frame_has_no_secret_locals(self):
        try:
            self.capture(send_error=True)
        except ValueError as error:
            tb = error.__traceback__
            while tb and tb.tb_frame.f_code.co_filename != transport.__file__:
                tb = tb.tb_next
            self.assertTrue(tb is not None)
            formatted = "".join(traceback.TracebackException(type(error), error, tb,
                capture_locals=True).format())
            self.assertTrue(self.credential not in formatted)
            self.assertTrue(tb.tb_frame.f_locals["credential"] is None)
            self.assertTrue(tb.tb_frame.f_code.co_name == "capture_transport")
        else:
            self.fail("expected a safe failure")

    def test_permitted_reporting_uses_only_fixed_code(self):
        class UnsafeException(Exception):
            def __str__(self): raise AssertionError("must not format exception")
            def __repr__(self): raise AssertionError("must not format exception")
        sink = io.StringIO()
        logger = logging.Logger("offline-report", logging.ERROR)
        logger.propagate = False
        logger.addHandler(logging.StreamHandler(sink))
        logger.error(transport.format_transport_error(UnsafeException(self.credential)))
        self.assertEqual(sink.getvalue(), "BL1_TRANSPORT_CAPTURE_INCOMPLETE\n")

    def test_http_debug_logging_rejected_before_dispatch(self):
        with patch.object(http.client.HTTPConnection, "debuglevel", 1):
            self.rejected()
        self.assertEqual(self.calls, 0)
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_stream_interruption_never_completes(self):
        class Interrupted(io.BytesIO):
            def read(self, count=-1):
                if self.tell(): raise OSError("offline interruption")
                return super().read(min(count, 2))
        self.rejected(body=b"abcde", stream=Interrupted(b"abcde"))
        self.assertFalse((self.directory / "sample.json").exists())

    def test_credential_reflection_across_chunk_boundary_never_retained(self):
        token = self.credential.encode()
        self.rejected(body=b"x" * 65520 + token + b"tail")
        self.no_credentials_retained()

    def test_json_escaped_credentials_never_publish_raw_metadata_or_receipt(self):
        escaped = "".join("\\u%04x" % ord(c) for c in self.credential)
        cases = ['{"echo":"' + escaped + '"}',
                 '{"nested":[{"x":"prefix ' + escaped + ' suffix"}]}',
                 '{"' + escaped + '":"key reflection"}',
                 '{"echo":"' + escaped + '","echo":"overwritten"}',
                 '{"echo":"' + escaped]  # Cannot safely inspect malformed JSON.
        original_write = transport.capture._write_checked
        for index, body in enumerate(cases):
            self.fresh()
            writes = []
            def observed_write(stream, block):
                writes.append(len(block))
                return original_write(stream, block)
            with self.subTest(index=index), patch.object(transport.capture, "_write_checked", observed_write):
                self.rejected(body=body.encode())
            self.assertEqual(writes, [])
            for suffix in (".raw", ".json", ".transport.json"):
                self.assertFalse((self.directory / ("sample" + suffix)).exists())

    def test_json_escaped_slash_credential_rejected(self):
        self.credential = "transient/" + secrets.token_urlsafe(32)
        body = ('{"echo":"' + self.credential.replace("/", "\\/") + '"}').encode()
        self.rejected(body=body)
        self.assertFalse((self.directory / "sample.raw").exists())

    def test_accepted_json_preserves_exact_original_bytes(self):
        body = b'  { "nested" : ["caf\xc3\xa9", 1, true] }\r\n'
        result = self.capture(body=body)
        self.assertEqual((self.directory / "sample.raw").read_bytes(), body)
        self.assertEqual(result["raw"], contract.identity(ROOT, self.directory / "sample.raw"))

    def test_public_metadata_credential_overlap_rejected_before_artifacts(self):
        self.credential = "2026"
        self.rejected()
        self.assertEqual(self.calls, 0)
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_wrong_method_url_body_or_authentication_rejected(self):
        mutations = [lambda r, p: setattr(p, "method", "POST"),
                     lambda r, p: setattr(p, "url", transport.PREPARED_URL + "&extra=1"),
                     lambda r, p: setattr(p, "body", b"body"),
                     lambda r, p: p.headers.pop("X-Auth-Token"),
                     lambda r, p: p.headers.update({"Cookie": "unapproved"}),
                     lambda r, p: p.headers.update({"Accept-Encoding": "gzip"})]
        for index, change in enumerate(mutations):
            with self.subTest(index=index): self.fresh(); self.rejected(configure=change)

    def test_independent_response_request_and_wrong_response_url_rejected(self):
        for mutation in (lambda r, p: setattr(r, "request", p.copy()),
                         lambda r, p: setattr(r, "url", transport.URL)):
            self.fresh(); self.rejected(configure=mutation)

    def test_redirect_auth_failure_rate_limit_and_history_rejected(self):
        for status in (301, 302, 401, 403, 429, 500):
            with self.subTest(status=status):
                self.fresh(); self.rejected(configure=lambda r, p: setattr(r, "status_code", status))
        self.fresh(); self.rejected(configure=lambda r, p: setattr(r, "history", [requests.Response()]))

    def test_prebuffered_decoded_or_unenforced_raw_rejected(self):
        for mutation in (lambda r, p: setattr(r, "_content_consumed", True),
                         lambda r, p: setattr(r, "_content", b"abc"),
                         lambda r, p: setattr(r.raw, "decode_content", True),
                         lambda r, p: setattr(r.raw, "enforce_content_length", False)):
            self.fresh(); self.rejected(configure=mutation)

    def test_request_mutation_during_stream_rejected(self):
        def configure(response, prepared):
            original = response.raw.read
            def changed(*args, **kwargs):
                data = original(*args, **kwargs)
                prepared.url += "&changed=1"
                return data
            response.raw.read = changed
        self.rejected(configure=configure)

    def test_response_headers_mutation_during_stream_rejected(self):
        def configure(response, prepared):
            original = response.raw.read
            def changed(*args, **kwargs):
                data = original(*args, **kwargs)
                response.raw.headers["Content-Length"] = "99"
                return data
            response.raw.read = changed
        self.rejected(configure=configure)

    def test_missing_or_invalid_limits_rejected_before_send(self):
        for field, value in (("connect_timeout_seconds", 0), ("read_timeout_seconds", float("inf")),
                             ("read_timeout_seconds", True), ("max_response_bytes", 0),
                             ("max_response_bytes", True)):
            self.fresh(); self.rejected(**{field: value})
        self.assertEqual(self.calls, 0)

    def test_missing_or_invalid_credential_rejected_without_retention(self):
        for value in (None, "", "bad\r\nheader"):
            self.fresh(); self.rejected(credential=value)
        self.assertEqual(self.calls, 0)

    def test_duplicate_destination_refuses_send_and_preserves_artifacts(self):
        self.capture()
        before = {p.name: p.read_bytes() for p in self.directory.iterdir()}
        with self.assertRaisesRegex(ValueError, "^BL1_TRANSPORT_CAPTURE_INCOMPLETE$"):
            self.capture()
        transport.read_transport_capture(ROOT, self.directory, "sample")
        self.assertEqual(self.calls, 1)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.directory.iterdir()})

    def test_failed_operation_reservation_blocks_automatic_reuse(self):
        self.rejected(send_error=True)
        self.assertTrue((self.directory / "sample.transport.pending").is_dir())
        self.rejected()
        self.assertEqual(self.calls, 1)

    def test_interleaving_completed_winner_cannot_be_stranded_by_loser(self):
        original_mkdir = Path.mkdir
        entered = False
        winner = []
        def interleaved(path, *args, **kwargs):
            nonlocal entered
            if path == self.directory / "sample.transport.pending" and not entered:
                entered = True
                winner.append(self.capture())  # Winner finishes before loser's atomic acquire.
            return original_mkdir(path, *args, **kwargs)
        with patch.object(Path, "mkdir", interleaved):
            with self.assertRaisesRegex(ValueError, "^BL1_TRANSPORT_CAPTURE_INCOMPLETE$"):
                self.capture()
        self.assertEqual(self.calls, 1)
        self.assertEqual(winner[0], transport.read_transport_capture(ROOT, self.directory, "sample"))
        self.assertFalse((self.directory / "sample.transport.pending").exists())

    def test_interleaving_active_winner_keeps_its_reservation(self):
        def competitor(response, prepared):
            self.assertTrue((self.directory / "sample.transport.pending").exists())
            with self.assertRaises(ValueError): self.capture()
            self.assertTrue((self.directory / "sample.transport.pending").exists())
        result = self.capture(configure=competitor)
        self.assertEqual(self.calls, 1)
        self.assertEqual(result, transport.read_transport_capture(ROOT, self.directory, "sample"))

    def test_real_requests_adapter_reaches_controlled_urllib3_boundary(self):
        body = b'{"offline_fixture":true}'
        wire = b"HTTP/1.1 200 OK\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body
        class LocalSocket:
            def makefile(self, *args): return io.BytesIO(wire)
        calls = []
        def boundary(pool, **kwargs):
            calls.append(True)
            self.assertEqual(pool.host, "api.football-data.org")
            self.assertEqual(pool.port, 443)
            self.assertEqual(kwargs["method"], "GET")
            self.assertEqual(kwargs["url"], "/v4/competitions/BL1/teams?season=2026")
            self.assertIsNone(kwargs["body"])
            self.assertEqual(set(kwargs["headers"]), {"X-Auth-Token", "Accept-Encoding"})
            self.assertTrue(kwargs["headers"]["X-Auth-Token"] == self.credential)
            self.assertEqual(kwargs["headers"]["Accept-Encoding"], "identity")
            self.assertIs(kwargs["redirect"], False)
            self.assertIs(kwargs["preload_content"], False)
            self.assertIs(kwargs["decode_content"], False)
            self.assertIs(kwargs["chunked"], False)
            self.assertEqual(kwargs["retries"].total, 0)
            self.assertEqual(kwargs["timeout"].connect_timeout, 1.0)
            self.assertEqual(kwargs["timeout"].read_timeout, 2.0)
            self.assertEqual(pool.cert_reqs, "CERT_REQUIRED")
            parsed = http.client.HTTPResponse(LocalSocket())
            parsed.begin()
            return HTTPResponse(body=parsed, headers=HTTPHeaderDict(parsed.headers.items()),
                status=parsed.status, original_response=parsed, request_method="GET",
                preload_content=False, decode_content=False, enforce_content_length=True)
        # HTTPAdapter.send, request_url, cert_verify and build_response are real.
        # Only the pool boundary is replaced; no socket or TLS exchange occurs.
        with patch.object(HTTPSConnectionPool, "urlopen", boundary):
            result = transport.capture_transport(ROOT, self.directory, "sample", credential=self.credential,
                connect_timeout_seconds=1.0, read_timeout_seconds=2.0, max_response_bytes=200000)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result, transport.read_transport_capture(ROOT, self.directory, "sample"))

    def test_approved_completed_at_precedes_receipt_publication_not_success(self):
        events = []
        original_now = contract.now
        original_observation = transport.qualification.observation
        original_persist = transport.capture._persist
        original_publish = transport.capture._publish
        def clock():
            value = original_now(); events.append(("clock", value)); return value
        def observe(*args, **kwargs):
            value = original_observation(*args, **kwargs)
            events.append(("metadata_validated", None)); return value
        def persist(path, payload):
            if path.name == "receipt.json": events.append(("receipt_write", None))
            return original_persist(path, payload)
        def publish(source, destination):
            if destination.name.endswith(".transport.json"): events.append(("receipt_publish", None))
            return original_publish(source, destination)
        with patch.object(contract, "now", clock), patch.object(transport.qualification, "observation", observe), \
             patch.object(transport.capture, "_persist", persist), patch.object(transport.capture, "_publish", publish):
            result = self.capture()
        events.append(("success_returned", None))
        stamp = events.index(("clock", result["completed_at"]))
        self.assertTrue(any(e[0] == "metadata_validated" for e in events[:stamp]))
        self.assertLess(stamp, events.index(("receipt_write", None)))
        self.assertLess(events.index(("receipt_write", None)), events.index(("receipt_publish", None)))
        self.assertTrue(any(e[0] == "metadata_validated" for e in events[events.index(("receipt_publish", None)) + 1:]))
        self.assertEqual(result, transport.read_transport_capture(ROOT, self.directory, "sample"))

    def test_receipt_publication_failure_is_not_success_despite_valid_metadata(self):
        original = transport.capture._publish
        def fail_receipt(source, destination):
            if destination.name.endswith(".transport.json"): raise OSError("offline publication failure")
            original(source, destination)
        with patch.object(transport.capture, "_publish", fail_receipt): self.rejected()
        self.assertTrue((self.directory / "sample.raw").is_file())
        metadata = contract.read(self.directory / "sample.json")
        transport.qualification.observation(metadata, ROOT, cutoff=contract.now())
        self.assertTrue((self.directory / "sample.transport.pending").is_dir())
        self.assertFalse((self.directory / "sample.transport.json").exists())
        self.no_credentials_retained()

    def test_incomplete_receipt_write_is_never_consumable(self):
        original = transport.capture._persist
        def short_receipt(path, payload):
            if path.name == "receipt.json":
                path.write_bytes(b"{"); raise OSError("offline short write")
            original(path, payload)
        with patch.object(transport.capture, "_persist", short_receipt): self.rejected()
        self.assertTrue((self.directory / "sample.json").exists())

    def test_receipt_readback_corruption_keeps_incomplete_marker(self):
        original = transport.capture._publish
        def corrupt(source, destination):
            original(source, destination)
            if destination.name.endswith(".transport.json"): destination.write_bytes(b"{}")
        with patch.object(transport.capture, "_publish", corrupt): self.rejected()
        self.assertTrue((self.directory / "sample.transport.pending").exists())

    def test_consumer_requires_receipt_even_for_valid_capture_sidecar(self):
        self.capture()
        (self.directory / "sample.transport.json").unlink()
        with self.assertRaises(FileNotFoundError):
            transport.read_transport_capture(ROOT, self.directory, "sample")

    def test_consumer_rejects_tampered_raw_or_metadata(self):
        for suffix in (".raw", ".json"):
            self.fresh(); self.capture()
            (self.directory / ("sample" + suffix)).write_bytes(b"tampered")
            with self.assertRaises(ValueError): transport.read_transport_capture(ROOT, self.directory, "sample")

    def test_consumer_rejects_invalid_seal(self):
        receipt = self.capture(); receipt["record_sha256"] = "0" * 64
        (self.directory / "sample.transport.json").write_bytes(contract.canonical(receipt))
        with self.assertRaises(ValueError): transport.read_transport_capture(ROOT, self.directory, "sample")

    def test_consumer_rejects_missing_extra_or_resealed_wrong_fields(self):
        original = self.capture()
        path = self.directory / "sample.transport.json"
        variants = []
        missing = copy.deepcopy(original); del missing["method"]; variants.append(missing)
        extra = copy.deepcopy(original); extra["extra"] = 1; variants.append(extra)
        for field, value in (("method", "POST"), ("provider", "other"), ("prepared_url", transport.URL),
                             ("request_sha256", "0" * 64), ("http_status", 201),
                             ("completed_at", "1900-01-01T00:00:00Z"), ("producer_files", [])):
            changed = copy.deepcopy(original); changed[field] = value; variants.append(changed)
        changed = copy.deepcopy(original); changed["controls"]["allow_redirects"] = 0; variants.append(changed)
        changed = copy.deepcopy(original); changed["framing"]["received_entity_bytes"] = 999; variants.append(changed)
        changed = copy.deepcopy(original); changed["limits"]["max_response_bytes"] = True; variants.append(changed)
        changed = copy.deepcopy(original); changed["raw"]["path"] = "../escape"; variants.append(changed)
        for index, changed in enumerate(variants):
            path.write_bytes(contract.canonical(contract.seal(changed)))
            with self.subTest(index=index), self.assertRaises(ValueError):
                transport.read_transport_capture(ROOT, self.directory, "sample")

    def test_consumer_rejects_pending_marker_even_with_valid_receipt(self):
        self.capture(); (self.directory / "sample.transport.pending").mkdir()
        with self.assertRaisesRegex(ValueError, "INCOMPLETE"):
            transport.read_transport_capture(ROOT, self.directory, "sample")

    def test_traversal_names_rejected_before_send(self):
        for name in ("../escape", "a/b", "a\\b", "a:ads", "CON"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                transport.capture_transport(ROOT, self.directory, name, credential=self.credential,
                    connect_timeout_seconds=1, read_timeout_seconds=2, max_response_bytes=100)
        self.assertEqual(self.calls, 0)


if __name__ == "__main__":
    unittest.main()
