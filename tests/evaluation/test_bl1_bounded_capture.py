"""Offline mechanics only: all responses/providers and payloads are test fixtures.

Run with the isolated evidence runner, not shared repository fixture machinery.
BL1_CAPTURE_TEST_ROOT must point into the expressly authorized evidence directory.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch
import uuid

from src.evaluation import bl1_bounded_capture as capture
from src.evaluation import future_lineage_context as contract
from src.evaluation.future_lineage_qualification import observation


REPOSITORY = Path(__file__).resolve().parents[2]
EVIDENCE = REPOSITORY / "tmp/BL1-bounded-capture-envelope-implementation-20261024-01"
REQUEST = {"url": "https://example.invalid/offline-fixture", "params": {"season": 2026}}


class Response(io.BytesIO):
    status_code = 200

    def __init__(self, body):
        super().__init__(body)
        self.read_sizes = []

    def read(self, count):
        self.read_sizes.append(count)
        return super().read(count)


class CaptureTests(unittest.TestCase):
    def setUp(self):
        base = Path(os.environ["BL1_CAPTURE_TEST_ROOT"]).absolute()
        self.assertTrue(base.is_relative_to(EVIDENCE))
        self.directory = contract.safe_path(EVIDENCE, base.relative_to(EVIDENCE).as_posix())
        self.directory = contract.safe_path(self.directory, uuid.uuid4().hex)
        self.directory.mkdir(parents=True)

    def run_capture(self, body=b'{"offline_fixture":true}', **changes):
        args = dict(provider="offline-fixture", request=copy.deepcopy(REQUEST),
                    open_response=lambda: Response(body), max_response_bytes=1024)
        args.update(changes)
        return capture.capture_response(REPOSITORY, self.directory, "sample", **args)

    def assert_incomplete(self):
        self.assertFalse((self.directory / "sample.json").exists())
        self.assertFalse((self.directory / "sample.pending").exists())

    def test_round_trip_exact_body_request_and_existing_seal(self):
        body = b'{"offline_fixture":"\\u00e9"}\r\n'
        record = self.run_capture(body)
        self.assertEqual((self.directory / "sample.raw").read_bytes(), body)
        self.assertEqual(record, contract.read(self.directory / "sample.json"))
        self.assertEqual(record["raw"]["sha256"], hashlib.sha256(body).hexdigest())
        self.assertEqual(record["raw"]["size_bytes"], len(body))
        envelope = observation(record, REPOSITORY, cutoff=record["retrieved_at"])
        self.assertEqual(envelope["basis"], "archived_response_observation")
        self.assertEqual(envelope["provider"], record["provider"])
        self.assertEqual(envelope["raw"], record["raw"])
        self.assertEqual(envelope["request_sha256"], contract.fingerprint(REQUEST))
        self.assertEqual(envelope, contract.seal(envelope))
        self.assertLessEqual(contract.instant(record["started_at"]),
                             contract.instant(envelope["observed_by"]))

    def test_exact_size_limit_and_closed_response(self):
        response = Response(b"12345")
        record = self.run_capture(open_response=lambda: response, max_response_bytes=5)
        self.assertEqual(record["raw"]["size_bytes"], 5)
        self.assertEqual(response.read_sizes, [6, 1])
        self.assertTrue(response.closed)

    def test_oversize_stops_after_only_one_over_limit_byte(self):
        class Unlimited(Response):
            def read(self, count):
                self.read_sizes.append(count)
                return b"x" * count
        response = Unlimited(b"")
        with self.assertRaisesRegex(ValueError, "RESPONSE_TOO_LARGE"):
            self.run_capture(open_response=lambda: response, max_response_bytes=150000)
        self.assertEqual(sum(response.read_sizes), 150001)
        self.assertLessEqual(max(response.read_sizes), 65536)
        self.assertTrue(response.closed)
        self.assert_incomplete()
        self.assertFalse((self.directory / "sample.raw").exists())

    def test_empty_body_is_captured_without_claiming_valid_payload(self):
        record = self.run_capture(b"")
        self.assertEqual(record["raw"]["size_bytes"], 0)

    def test_expected_body_length_round_trip(self):
        record = self.run_capture(b"12345", expected_response_bytes=5)
        self.assertEqual(record["raw"]["size_bytes"], 5)

    def test_truncated_and_wrong_length_bodies_never_publish(self):
        for expected in (4, 6):
            with self.subTest(expected=expected), self.assertRaisesRegex(ValueError, "SIZE_MISMATCH"):
                self.run_capture(b"12345", expected_response_bytes=expected)
            self.assert_incomplete()
            self.assertFalse((self.directory / "sample.raw").exists())

    def test_invalid_expected_body_length_rejected(self):
        for expected in (-1, True, 1.5, 1025):
            with self.subTest(expected=expected), self.assertRaisesRegex(ValueError, "EXPECTED_SIZE_INVALID"):
                self.run_capture(expected_response_bytes=expected)
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_transport_reported_truncation_does_not_publish(self):
        class Truncated(Response):
            def read(self, count):
                raise EOFError("offline incomplete HTTP framing")
        with self.assertRaisesRegex(EOFError, "incomplete HTTP framing"):
            self.run_capture(open_response=lambda: Truncated(b"x"))
        self.assert_incomplete()

    def test_invalid_limits_reject_before_opening(self):
        for limit in (0, -1, True, 1.5, None):
            with self.subTest(limit=limit), patch.object(capture, "_publish") as publish:
                with self.assertRaisesRegex(ValueError, "SIZE_LIMIT_REQUIRED"):
                    self.run_capture(max_response_bytes=limit)
                publish.assert_not_called()
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_missing_malformed_provider_and_request(self):
        bad = [{"provider": None}, {"provider": " "}, {"request": None},
               {"request": {}}, {"request": {"url": 1, "params": {}}},
               {"request": {"url": "x", "params": []}}]
        for changes in bad:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.run_capture(**changes)
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_nonfinite_request_is_rejected(self):
        with self.assertRaises(ValueError):
            self.run_capture(request={"url": "offline", "params": {"x": float("nan")}})
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_request_mutation_cannot_change_captured_binding(self):
        request = copy.deepcopy(REQUEST)
        def factory():
            request["params"]["season"] = 9999
            return Response(b"x")
        record = self.run_capture(request=request, open_response=factory)
        self.assertEqual(record["request"], REQUEST)

    def test_transport_factory_failure(self):
        def fail():
            raise OSError("offline transport failure")
        with self.assertRaisesRegex(OSError, "offline transport"):
            self.run_capture(open_response=fail)
        self.assert_incomplete()

    def test_interrupted_stream_after_partial_body(self):
        class Interrupted(Response):
            def read(self, count):
                if self.tell():
                    raise OSError("offline interruption")
                return super().read(min(count, 3))
        response = Interrupted(b"abcdef")
        with self.assertRaisesRegex(OSError, "interruption"):
            self.run_capture(open_response=lambda: response)
        self.assertTrue(response.closed)
        self.assert_incomplete()
        self.assertFalse((self.directory / "sample.raw").exists())

    def test_close_failure_does_not_publish_complete_capture(self):
        class CloseFailure(Response):
            def close(self):
                if self.closed:
                    return
                super().close()
                raise OSError("offline close failure")
        with self.assertRaisesRegex(OSError, "close failure"):
            self.run_capture(open_response=lambda: CloseFailure(b"abc"))
        self.assert_incomplete()

    def test_invalid_status_rejects_before_read(self):
        for status in (401, 429, 302, "200", None):
            response = Response(b"not read")
            response.status_code = status
            with self.subTest(status=status), self.assertRaisesRegex(ValueError, "HTTP_STATUS"):
                self.run_capture(open_response=lambda: response)
            self.assertEqual(response.read_sizes, [])
            self.assertTrue(response.closed)
        self.assert_incomplete()

    def test_nonbytes_and_oversized_read_contract(self):
        for value in (None, "text", bytearray(b"x"), b"x" * 1026):
            response = Response(b"")
            response.read = lambda count: value
            with self.subTest(type=type(value).__name__), self.assertRaisesRegex(ValueError, "STREAM_INVALID"):
                self.run_capture(open_response=lambda: response)
        self.assert_incomplete()

    def test_short_raw_and_metadata_writes_do_not_publish(self):
        original = Path.open
        class ShortWriter:
            def __init__(self, stream): self.stream = stream
            def __enter__(self): self.stream.__enter__(); return self
            def __exit__(self, *args): return self.stream.__exit__(*args)
            def write(self, block): return self.stream.write(block[:-1])
        for filename in ("response.raw", "metadata.json"):
            def open_short(path, mode="r", *args, **kwargs):
                stream = original(path, mode, *args, **kwargs)
                return ShortWriter(stream) if mode == "xb" and path.name == filename else stream
            with self.subTest(filename=filename), patch.object(Path, "open", open_short):
                with self.assertRaisesRegex(ValueError, "INCOMPLETE_WRITE"):
                    self.run_capture()
            self.assert_incomplete()
            self.assertFalse((self.directory / "sample.raw").exists())

    def test_flush_failure_prevents_publication(self):
        with patch.object(capture.os, "fsync", side_effect=OSError("offline fsync failure")):
            with self.assertRaisesRegex(OSError, "fsync failure"):
                self.run_capture()
        self.assert_incomplete()

    def test_duplicate_destination_preserves_existing_capture(self):
        record = self.run_capture()
        before = {p.name: p.read_bytes() for p in self.directory.iterdir()}
        with self.assertRaisesRegex(ValueError, "DESTINATION_EXISTS"):
            self.run_capture(b"replacement")
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.directory.iterdir()})
        observation(record, REPOSITORY, cutoff=record["retrieved_at"])

    def test_publication_primitive_refuses_existing_destination(self):
        source = self.directory / "staged"
        destination = self.directory / "existing"
        source.write_bytes(b"new")
        destination.write_bytes(b"preserve")
        with self.assertRaises(FileExistsError):
            capture._publish(source, destination)
        self.assertEqual(destination.read_bytes(), b"preserve")
        self.assertEqual(source.read_bytes(), b"new")

    def test_existing_reparse_boundary_rejected_without_native_link_creation(self):
        original = Path.lstat
        def reparse(path, *args, **kwargs):
            info = original(path, *args, **kwargs)
            if path == self.directory:
                from types import SimpleNamespace
                return SimpleNamespace(st_file_attributes=0x400)
            return info
        with patch.object(Path, "lstat", reparse):
            with self.assertRaisesRegex(ValueError, "REPARSE_POINT"):
                self.run_capture()
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_existing_raw_or_sidecar_or_crash_reservation_blocks_reuse(self):
        for filename in ("sample.raw", "sample.json", "sample.pending"):
            path = self.directory / filename
            if filename.endswith("pending"):
                path.mkdir()
                with self.assertRaises(FileExistsError): self.run_capture()
                path.rmdir()
            else:
                path.write_bytes(b"existing user evidence")
                with self.assertRaisesRegex(ValueError, "DESTINATION_EXISTS"): self.run_capture()
                self.assertEqual(path.read_bytes(), b"existing user evidence")
                path.unlink()

    def test_path_traversal_and_reserved_names_rejected(self):
        for name in ("../escape", "a/b", "a\\b", "a:stream", "CON", "a.", ""):
            with self.subTest(name=name), self.assertRaises(ValueError):
                capture.capture_response(REPOSITORY, self.directory, name, provider="offline-fixture",
                    request=REQUEST, open_response=lambda: Response(b"x"), max_response_bytes=1)
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_boundary_escape_rejected(self):
        with self.assertRaises(ValueError):
            capture.capture_response(REPOSITORY, REPOSITORY.parent, "escape", provider="offline-fixture",
                request=REQUEST, open_response=lambda: Response(b"x"), max_response_bytes=1)

    def test_clock_rollback_rejected(self):
        with patch.object(capture, "now", side_effect=["2026-10-24T12:00:02Z",
                "2026-10-24T12:00:01Z", "2026-10-24T12:00:03Z"]):
            with self.assertRaisesRegex(ValueError, "CLOCK_CONTRADICTION"):
                self.run_capture()
        self.assert_incomplete()

    def test_completion_uses_local_clock_not_provider_payload_timestamps(self):
        record = self.run_capture(b'{"Date":"1900-01-01","lastUpdated":"2999-01-01"}')
        self.assertGreaterEqual(contract.instant(record["started_at"]).year, 2026)
        self.assertEqual(record["archived_observation"]["not_before"], record["started_at"])
        self.assertLessEqual(contract.instant(record["archived_observation"]["observed_by"]),
                             contract.instant(record["retrieved_at"]))

    def test_raw_publication_failure_has_no_complete_sidecar(self):
        with patch.object(capture, "_publish", side_effect=OSError("offline publication failure")):
            with self.assertRaisesRegex(OSError, "publication failure"): self.run_capture()
        self.assert_incomplete()

    def test_sidecar_publication_failure_leaves_explicit_orphan_not_success(self):
        original = capture._publish
        def fail_meta(source, destination):
            if destination.suffix == ".json": raise OSError("offline sidecar failure")
            original(source, destination)
        with patch.object(capture, "_publish", fail_meta):
            with self.assertRaisesRegex(OSError, "sidecar failure"): self.run_capture()
        self.assertTrue((self.directory / "sample.raw").exists())
        self.assert_incomplete()
        with self.assertRaisesRegex(ValueError, "DESTINATION_EXISTS"): self.run_capture()

    def test_raw_drift_before_sidecar_publication_is_rejected(self):
        original = capture._publish
        def corrupt_raw(source, destination):
            original(source, destination)
            if destination.suffix == ".raw": destination.write_bytes(b"tampered")
        with patch.object(capture, "_publish", corrupt_raw):
            with self.assertRaisesRegex(ValueError, "REFERENCE_DRIFT"): self.run_capture()
        self.assert_incomplete()

    def test_private_metadata_drift_rejected_before_publication(self):
        original = capture._persist
        def corrupt(path, payload):
            original(path, payload)
            path.write_bytes(b"{}")
        with patch.object(capture, "_persist", corrupt):
            with self.assertRaisesRegex(ValueError, "METADATA_READBACK"): self.run_capture()
        self.assert_incomplete()

    def test_existing_validator_rejects_missing_malformed_and_tampered_envelopes(self):
        record = self.run_capture()
        variants = []
        missing = copy.deepcopy(record); del missing["archived_observation"]; variants.append(missing)
        for field, value in (("basis", "invented"), ("provider", "another"),
                ("request_sha256", "0" * 64), ("record_sha256", "0" * 64),
                ("not_before", "timezone missing")):
            changed = copy.deepcopy(record)
            changed["archived_observation"][field] = value
            if field == "not_before":
                changed["archived_observation"] = contract.seal(changed["archived_observation"])
            variants.append(changed)
        changed = copy.deepcopy(record); changed["request"]["params"]["season"] = 1900; variants.append(changed)
        changed = copy.deepcopy(record); changed["raw"]["sha256"] = "0" * 64; variants.append(changed)
        for index, changed in enumerate(variants):
            with self.subTest(index=index), self.assertRaises(ValueError):
                observation(changed, REPOSITORY, cutoff=record["retrieved_at"])

    def test_late_and_after_kickoff_records_remain_rejected(self):
        record = self.run_capture()
        with self.assertRaisesRegex(ValueError, "LATE_OR_CONTRADICTORY"):
            observation(record, REPOSITORY, cutoff="1900-01-01T00:00:00Z")
        with self.assertRaisesRegex(ValueError, "AFTER_KICKOFF"):
            observation(record, REPOSITORY, cutoff=record["retrieved_at"], kickoff=record["started_at"])

    def test_unchanged_parser_mapper_preserve_sidecar_and_identity(self):
        from src.evaluation import phase11_current_source as legacy
        payload = {"competition": {"code": "BL1"}, "season": {"startDate": "2026-08-01"},
                   "teams": [{"id": 1, "name": "Alpha", "shortName": "Alpha"},
                             {"id": 2, "name": "Beta", "shortName": "Beta"}]}
        record = self.run_capture(json.dumps(payload).encode())
        original = copy.deepcopy(record)
        sidecar = (self.directory / "sample.json").read_bytes()
        parsed = legacy._json_payload(record)
        mapped, findings = legacy._mapped_teams(parsed, "BL1", 2026,
                                               {"ALPHA": {"ALPHA"}, "BETA": {"BETA"}})
        self.assertEqual(len(mapped), 2)
        self.assertEqual(findings, [])
        self.assertEqual(record, original)
        self.assertEqual((self.directory / "sample.json").read_bytes(), sidecar)
        observation(record, REPOSITORY, cutoff=record["retrieved_at"])
        # Normalized outputs are not independent evidence: keep the sidecar.
        self.assertNotIn("archived_observation", parsed)
        (self.directory / "sample.raw").write_bytes(b"tampered")
        with self.assertRaises(ValueError): legacy._json_payload(record)
        with self.assertRaises(ValueError): observation(record, REPOSITORY, cutoff=record["retrieved_at"])


if __name__ == "__main__":
    unittest.main()
