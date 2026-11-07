"""Bounded local fixtures only. No real credentials, models or provider traffic."""
import json
import os
from pathlib import Path
import secrets
import time
import unittest
from unittest.mock import patch
import uuid

from src.evaluation import bl1_acquisition_gate as gate

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / gate.EVIDENCE_ROOT
BODY = b'{"competition":{"code":"BL1"},"season":{"startDate":"2026-08-01"},"teams":[{"id":1,"name":"Alpha","shortName":"Alpha"}]}\n'
NAMES = {"ALPHA": ["ALPHA"]}


class CompletedWorker:
    """Supervisor-transition fixture; no producer/receipt validation claim."""
    returncode = 0

    def __init__(self):
        self.stdout = json.dumps(dict(ok=True, receipt=dict(record_sha256="0" * 64))).encode()

    def communicate(self, *args, **kwargs):
        return self.stdout, None

    def poll(self):
        return 0


class GateTests(unittest.TestCase):
    def setUp(self):
        self.base = Path(os.environ["BL1_GATE_TEST_ROOT"])
        self.assertTrue(self.base.is_relative_to(EVIDENCE))
        self.directory = self.base / uuid.uuid4().hex
        self.directory.mkdir(parents=True)
        self.limits = dict(max_attempts=2, max_requests=2, elapsed_seconds=60,
            max_retained_bytes=50000, capture_storage_bytes=16000, max_response_bytes=1024)

    def session(self, **changes):
        return gate.OfflineSession(ROOT, self.directory, authorization=dict(gate.OFFLINE),
                                   limits={**self.limits, **changes})

    def file_bytes(self):
        return {p.relative_to(self.directory).as_posix(): p.read_bytes()
                for p in self.directory.rglob("*") if p.is_file()}

    def test_unauthorized_before_files_or_application_imports(self):
        for authorization in (None, {}, {**gate.OFFLINE, "mode": "live"},
                              {**gate.OFFLINE, "network": 0}, {**gate.OFFLINE, "credential_source": ".env"}):
            with self.subTest(authorization=authorization), self.assertRaisesRegex(ValueError, "AUTHORIZATION"):
                gate.OfflineSession(ROOT, self.directory, authorization=authorization, limits=self.limits)
            self.assertEqual(list(self.directory.iterdir()), [])
        with patch("builtins.__import__", side_effect=AssertionError("unexpected import")):
            with self.assertRaisesRegex(ValueError, "AUTHORIZATION"):
                gate._worker({"authorization": {"mode": "live"}})

    def test_invalid_limits_rejected_without_writes(self):
        for key in self.limits:
            for value in (None, True, 0, -1, float("nan"), float("inf")):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    self.session(**{key: value})
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_existing_files_and_session_claim_preserved(self):
        path = self.directory / "existing.txt"
        path.write_bytes(b"preserve")
        with self.assertRaisesRegex(ValueError, "DESTINATION_EXISTS"):
            self.session()
        self.assertEqual(path.read_bytes(), b"preserve")
        self.assertEqual(len(list(self.directory.iterdir())), 1)

    def test_claimed_empty_boundary_cannot_reset_counters(self):
        self.session()
        before = self.file_bytes()
        with self.assertRaisesRegex(ValueError, "DESTINATION_EXISTS"):
            self.session()
        self.assertEqual(self.file_bytes(), before)

    def test_ambiguous_output_path_rejected_before_writes(self):
        ambiguous = self.directory / ".." / self.directory.name
        with self.assertRaisesRegex(ValueError, "OUTPUT_SCOPE"):
            gate.OfflineSession(ROOT, ambiguous, authorization=gate.OFFLINE, limits=self.limits)
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_receipt_first_round_trip_and_derived_view(self):
        session = self.session()
        result = session.capture_fixture(BODY, names=NAMES)
        self.assertEqual(result["validator_view"]["credential_headers_present"], ["x-auth-token"])
        self.assertNotIn("credential_headers_present", result["original_metadata"])
        self.assertEqual(result["mapped"]["1"]["canonical"], "ALPHA")
        self.assertEqual(result["findings"], [])
        slot = self.directory / "attempt-000001"
        self.assertEqual((slot / "primary-2026-teams.raw").read_bytes(), BODY)
        self.assertEqual(json.loads((slot / "primary-2026-teams.json").read_bytes()), result["original_metadata"])
        before = self.file_bytes()
        self.assertEqual(json.loads((slot / "result.json").read_bytes())["status"], "PROVISIONAL")
        self.assertEqual(session._accepted[1], result["receipt"]["record_sha256"])
        self.assertTrue(all(len(value) <= 256 for key, value in before.items() if key.endswith("result.json")))
        replay = session.intake_saved(1, names=NAMES)
        self.assertEqual(replay["validator_view"], result["validator_view"])
        self.assertEqual(self.file_bytes(), before)
        self.assertLessEqual(sum(len(v) for v in before.values()), session.reserved_bytes)
        self.assertLessEqual(session.reserved_bytes, self.limits["max_retained_bytes"])
        from src.evaluation import future_lineage_qualification as qualification
        from types import SimpleNamespace
        primary = dict(league="BL1", requested_season=2026, division="D1", sources=[result["validator_view"]])
        alternative = dict(league="BL1", season=2026, selected_season="2026/2027")
        with self.assertRaisesRegex(ValueError, "^SOURCE_INDEX_INCOMPLETE$"):
            qualification.validate_indexes(SimpleNamespace(), primary, alternative,
                                          as_of=result["receipt"]["completed_at"])

    def test_tampered_receipt_blocks_replay_without_changing_originals(self):
        session = self.session()
        session.capture_fixture(BODY)
        slot = self.directory / "attempt-000001"
        original = (slot / "primary-2026-teams.json").read_bytes()
        (slot / "primary-2026-teams.transport.json").write_bytes(b"{}")
        with self.assertRaisesRegex(ValueError, "^BL1_GATE_INCOMPLETE$"):
            session.intake_saved(1)
        self.assertEqual((slot / "primary-2026-teams.json").read_bytes(), original)

    def test_missing_receipt_never_invokes_parser(self):
        from src.evaluation import phase11_current_source as legacy
        with patch.object(legacy, "_json_payload", side_effect=AssertionError("parser called")) as parser:
            with self.assertRaises(FileNotFoundError):
                gate._verified_teams(ROOT, self.directory)
            parser.assert_not_called()

    def test_wrong_completion_binding_rejected_before_parser(self):
        from src.evaluation import phase11_current_source as legacy
        session = self.session()
        session.capture_fixture(BODY)
        slot = self.directory / "attempt-000001"
        with patch.object(legacy, "_json_payload", side_effect=AssertionError("parser called")) as parser:
            with self.assertRaisesRegex(ValueError, "^BL1_GATE_INCOMPLETE$"):
                gate._verified_teams(ROOT, slot, expected_receipt_sha256="0" * 64)
            parser.assert_not_called()

    def test_attempt_limit_preserves_first_capture(self):
        session = self.session(max_attempts=1)
        session.capture_fixture(BODY)
        before = self.file_bytes()
        with self.assertRaisesRegex(ValueError, "ATTEMPT_LIMIT"):
            session.capture_fixture(BODY)
        self.assertEqual(self.file_bytes(), before)
        self.assertEqual(session.requests, 1)

    def test_request_limit_no_second_dispatch(self):
        session = self.session(max_requests=1)
        session.capture_fixture(BODY)
        before = self.file_bytes()
        with self.assertRaisesRegex(ValueError, "REQUEST_LIMIT"):
            session.capture_fixture(BODY)
        self.assertEqual(session.attempts, 2)
        self.assertEqual(session.requests, 1)
        self.assertEqual(self.file_bytes(), before)

    def test_total_storage_denied_before_slot_or_worker(self):
        session = self.session(max_retained_bytes=1)
        with patch.object(gate.subprocess, "Popen", side_effect=AssertionError("worker launched")):
            with self.assertRaisesRegex(ValueError, "STORAGE_LIMIT"):
                session.capture_fixture(BODY)
        self.assertEqual(self.file_bytes(), {"session.reserved": b""})

    def test_worker_environment_excludes_inherited_credentials_and_errors(self):
        session = self.session()
        secret = secrets.token_urlsafe(32)
        environments = []
        def launch(*args, **kwargs):
            environments.append(kwargs["env"])
            raise OSError(secret)
        with patch.dict(os.environ, {"FOOTBALL_DATA_API_KEY": secret}), patch.object(gate.subprocess, "Popen", launch):
            with self.assertRaisesRegex(ValueError, "^BL1_GATE_INCOMPLETE$") as raised:
                session.capture_fixture(BODY)
        self.assertIsNone(raised.exception.__context__)
        self.assertEqual(len(environments), 1)
        self.assertLessEqual(set(environments[0]), {"SystemRoot", "WINDIR", "PYTHON_DOTENV_DISABLED"})
        self.assertEqual(environments[0]["PYTHON_DOTENV_DISABLED"], "1")
        self.assertTrue(all(secret.encode() not in body for body in self.file_bytes().values()))
        self.assertTrue(all(len(body) <= 256 for path, body in self.file_bytes().items() if path.endswith("result.json")))

    def test_incomplete_completion_marker_blocks_saved_intake(self):
        session = self.session()
        session.capture_fixture(BODY)
        marker = self.directory / "attempt-000001/result.json"
        marker.write_bytes(b'{"status":"INCOMPLETE"}')
        before = self.file_bytes()
        with self.assertRaisesRegex(ValueError, "^BL1_GATE_INCOMPLETE$"):
            session.intake_saved(1)
        self.assertEqual(self.file_bytes(), before)

    def test_capture_allowance_denied_before_transport_artifacts(self):
        session = self.session(capture_storage_bytes=1)
        with self.assertRaisesRegex(ValueError, "^BL1_GATE_INCOMPLETE$"):
            session.capture_fixture(BODY)
        self.assertFalse(any(p.suffix == ".raw" for p in self.directory.rglob("*")))
        self.assertLessEqual(sum(len(v) for v in self.file_bytes().values()), session.reserved_bytes)

    def test_aggregate_storage_reservations_not_refunded(self):
        session = self.session(max_retained_bytes=20000)
        session.capture_fixture(BODY)
        before = self.file_bytes()
        with self.assertRaisesRegex(ValueError, "STORAGE_LIMIT"):
            session.capture_fixture(BODY)
        self.assertEqual(self.file_bytes(), before)

    def test_failed_dispatch_is_fixed_code_and_session_cannot_resume(self):
        session = self.session()
        with self.assertRaisesRegex(ValueError, "^BL1_GATE_INCOMPLETE$") as raised:
            session.capture_fixture(BODY, fail_send=True)
        self.assertIsNone(raised.exception.__context__)
        self.assertEqual((session.attempts, session.requests), (1, 1))
        before = self.file_bytes()
        with self.assertRaisesRegex(ValueError, "SESSION_INCOMPLETE"):
            session.capture_fixture(BODY)
        self.assertEqual(self.file_bytes(), before)

    def test_deadline_kills_blocked_worker_and_never_resumes(self):
        session = self.session(elapsed_seconds=2)
        started = time.monotonic()
        with self.assertRaisesRegex(ValueError, "^BL1_GATE_INCOMPLETE$"):
            session.capture_fixture(BODY, read_delay=10)
        self.assertLess(time.monotonic() - started, 7)
        self.assertTrue((self.directory / "attempt-000001/primary-2026-teams.transport.pending").is_dir())
        before = self.file_bytes()
        time.sleep(0.05)
        self.assertEqual(self.file_bytes(), before)
        self.assertEqual(json.loads((self.directory / "attempt-000001/failure.json").read_bytes())["status"], "INCOMPLETE")
        with self.assertRaisesRegex(ValueError, "SESSION_INCOMPLETE"):
            session.intake_saved(1)

    def test_expired_budget_blocks_before_attempt_and_process(self):
        session = self.session()
        session.deadline = time.monotonic() - 1
        with patch.object(gate.subprocess, "Popen", side_effect=AssertionError("worker launched")):
            with self.assertRaisesRegex(ValueError, "DEADLINE"):
                session.capture_fixture(BODY)
        self.assertEqual(session.attempts, 0)

    def test_replay_expiry_after_decode_or_validation_rejects_success(self):
        original_loads = json.loads
        for stage in ("decode", "validation"):
            self.directory = self.base / uuid.uuid4().hex
            self.directory.mkdir()
            session = self.session()
            session.attempts = 1
            session._accepted[1] = "0" * 64
            worker = CompletedWorker()
            class LateValidation(dict):
                def get(self, key, *args):
                    session.deadline = time.monotonic() - 1
                    return super().get(key, *args)
            def decode(payload):
                result = original_loads(payload)
                if stage == "decode":
                    session.deadline = time.monotonic() - 1
                    return result
                return LateValidation(result)
            with self.subTest(stage=stage), patch.object(gate.subprocess, "Popen", return_value=worker), \
                    patch.object(gate.json, "loads", side_effect=decode):
                with self.assertRaisesRegex(ValueError, "^BL1_GATE_INCOMPLETE$"):
                    session.intake_saved(1)
            self.assertTrue(session.failed)
            self.assertEqual(session._accepted, {})
            self.assertEqual(self.file_bytes(), {"session.reserved": b""})

    def test_admission_partial_write_fsync_and_interruption_are_terminal(self):
        for kind in ("partial", "fsync", "interrupt"):
            self.directory = self.base / uuid.uuid4().hex
            self.directory.mkdir()
            session = self.session()
            persist = session._persist
            partial = []
            def fail_admission(path, payload):
                if path.name != "admission.json":
                    return persist(path, payload)
                if kind == "fsync":
                    with patch.object(gate.os, "fsync", side_effect=OSError("fixture fsync failure")):
                        persist(path, payload)
                else:
                    partial.append(payload[:10])
                    with path.open("xb") as stream:
                        stream.write(partial[-1])
                    if kind == "interrupt":
                        raise KeyboardInterrupt()
                    raise OSError("fixture admission failure")
            with self.subTest(kind=kind), patch.object(session, "_persist", side_effect=fail_admission), \
                    patch.object(gate.subprocess, "Popen", side_effect=AssertionError("worker launched")):
                with self.assertRaisesRegex(ValueError, "^BL1_GATE_INCOMPLETE$") as raised:
                    session.capture_fixture(BODY)
            self.assertIsNone(raised.exception.__context__)
            self.assertTrue(session.failed)
            self.assertEqual((session.attempts, session.requests), (1, 1))
            self.assertGreater(session.reserved_bytes, 0)
            self.assertEqual(session._accepted, {})
            slot = self.directory / "attempt-000001"
            if partial:
                self.assertEqual((slot / "admission.json").read_bytes(), partial[0])
            self.assertEqual(json.loads((slot / "failure.json").read_bytes())["status"], "INCOMPLETE")
            before = self.file_bytes()
            with self.assertRaisesRegex(ValueError, "SESSION_INCOMPLETE"):
                session.capture_fixture(BODY)
            with self.assertRaisesRegex(ValueError, "SESSION_INCOMPLETE"):
                session.intake_saved(1)
            with self.assertRaisesRegex(ValueError, "DESTINATION_EXISTS"):
                self.session()
            self.assertEqual(self.file_bytes(), before)
            self.assertLessEqual(sum(map(len, before.values())), session.reserved_bytes)

    def test_admission_directory_failure_keeps_charges_and_terminal_state(self):
        session = self.session()
        with patch.object(Path, "mkdir", side_effect=OSError("fixture mkdir failure")), \
                patch.object(gate.subprocess, "Popen", side_effect=AssertionError("worker launched")):
            with self.assertRaisesRegex(ValueError, "^BL1_GATE_INCOMPLETE$"):
                session.capture_fixture(BODY)
        self.assertTrue(session.failed)
        self.assertEqual((session.attempts, session.requests), (1, 1))
        self.assertGreater(session.reserved_bytes, 0)
        self.assertEqual(session.failure_evidence_status, "UNAVAILABLE")
        self.assertEqual(self.file_bytes(), {"session.reserved": b""})

    def test_completion_persistence_crossing_deadline_retains_only_provisional(self):
        session = self.session()
        persist = session._persist
        def expire_after_write(path, payload):
            persist(path, payload)
            if path.name == "result.json":
                self.assertEqual(session._accepted, {})
                session.deadline = time.monotonic() - 1
        with patch.object(session, "_persist", side_effect=expire_after_write), \
                patch.object(gate.subprocess, "Popen", return_value=CompletedWorker()):
            with self.assertRaisesRegex(ValueError, "^BL1_GATE_INCOMPLETE$"):
                session.capture_fixture(BODY)
        slot = self.directory / "attempt-000001"
        self.assertEqual(json.loads((slot / "result.json").read_bytes())["status"], "PROVISIONAL")
        self.assertEqual(json.loads((slot / "failure.json").read_bytes())["status"], "INCOMPLETE")
        self.assertEqual(session._accepted, {})
        self.assertTrue(session.failed)
        before = self.file_bytes()
        self.assertTrue(all(len(value) <= 256 for key, value in before.items()
                            if key.endswith(("result.json", "failure.json"))))
        self.assertLessEqual(sum(map(len, before.values())), session.reserved_bytes)
        with self.assertRaisesRegex(ValueError, "SESSION_INCOMPLETE"):
            session.intake_saved(1)
        self.assertEqual(self.file_bytes(), before)

    def test_partial_provisional_write_and_fsync_failure_preserve_evidence(self):
        for kind in ("partial", "fsync", "interrupt"):
            self.directory = self.base / uuid.uuid4().hex
            self.directory.mkdir()
            session = self.session()
            persist = session._persist
            retained = []
            def fail_completion(path, payload):
                if path.name != "result.json":
                    return persist(path, payload)
                if kind == "fsync":
                    retained.append(payload)
                    with patch.object(gate.os, "fsync", side_effect=OSError("fixture fsync failure")):
                        persist(path, payload)
                else:
                    retained.append(payload[:10])
                    with path.open("xb") as stream:
                        stream.write(retained[-1])
                    if kind == "interrupt":
                        raise KeyboardInterrupt()
                    raise OSError("fixture provisional failure")
            with self.subTest(kind=kind), patch.object(session, "_persist", side_effect=fail_completion), \
                    patch.object(gate.subprocess, "Popen", return_value=CompletedWorker()):
                with self.assertRaisesRegex(ValueError, "^BL1_GATE_INCOMPLETE$"):
                    session.capture_fixture(BODY)
            slot = self.directory / "attempt-000001"
            self.assertEqual((slot / "result.json").read_bytes(), retained[0])
            self.assertEqual(json.loads((slot / "failure.json").read_bytes())["status"], "INCOMPLETE")
            self.assertTrue(session.failed)
            self.assertEqual(session._accepted, {})
            self.assertLessEqual(sum(map(len, self.file_bytes().values())), session.reserved_bytes)

    def test_failure_evidence_write_error_never_reopens_session(self):
        session = self.session()
        def fail_writes(path, payload):
            with path.open("xb") as stream:
                stream.write(payload[:10])
            raise OSError("fixture persistent storage failure")
        with patch.object(session, "_persist", side_effect=fail_writes):
            with self.assertRaisesRegex(ValueError, "^BL1_GATE_INCOMPLETE$") as raised:
                session.capture_fixture(BODY)
        self.assertIsNone(raised.exception.__context__)
        self.assertTrue(session.failed)
        self.assertEqual(session.failure_evidence_status, "UNAVAILABLE")
        self.assertEqual(session._accepted, {})
        before = self.file_bytes()
        with self.assertRaisesRegex(ValueError, "SESSION_INCOMPLETE"):
            session.capture_fixture(BODY)
        self.assertEqual(self.file_bytes(), before)

    def test_acceptance_follows_persistence_and_cannot_be_rebuilt_from_disk(self):
        session = self.session()
        persist = session._persist
        def observe(path, payload):
            if path.name == "result.json":
                self.assertEqual(session._accepted, {})
            persist(path, payload)
            if path.name == "result.json":
                self.assertEqual(session._accepted, {})
        with patch.object(session, "_persist", side_effect=observe), \
                patch.object(gate.subprocess, "Popen", return_value=CompletedWorker()):
            session.capture_fixture(BODY)
        self.assertEqual(session._accepted, {1: "0" * 64})
        before = self.file_bytes()
        session._accepted.clear()  # loss of in-memory acceptance, never disk recovery
        with patch.object(gate.subprocess, "Popen", side_effect=AssertionError("worker launched")):
            with self.assertRaisesRegex(ValueError, "CAPTURE_NOT_ACCEPTED"):
                session.intake_saved(1)
        self.assertEqual(self.file_bytes(), before)

    def test_failure_record_vetoes_previously_accepted_intake(self):
        session = self.session()
        session.capture_fixture(BODY)
        failure = self.directory / "attempt-000001/failure.json"
        failure.write_bytes(b'{"status":"INCOMPLETE"}')
        before = self.file_bytes()
        with self.assertRaisesRegex(ValueError, "^BL1_GATE_INCOMPLETE$"):
            session.intake_saved(1)
        self.assertEqual(session._accepted, {})
        self.assertEqual(self.file_bytes(), before)

    def test_malformed_oversize_and_reflected_bodies_fail_closed(self):
        for body, changes in ((b"not-json", {}), (b"x" * 1025, {}), (BODY, {"reflect_credential": True})):
            self.directory = self.base / uuid.uuid4().hex
            self.directory.mkdir()
            session = self.session()
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "^BL1_GATE_INCOMPLETE$"):
                session.capture_fixture(body, **changes)
            self.assertFalse(any(p.suffix == ".raw" for p in self.directory.rglob("*")))

    def test_incomplete_framing_and_redirect_rejected(self):
        for changes in ({"headers": [["Transfer-Encoding", "chunked"]]},
                        {"headers": [["Content-Length", str(len(BODY) + 1)]]}, {"status": 302}):
            self.directory = self.base / uuid.uuid4().hex
            self.directory.mkdir()
            session = self.session()
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "^BL1_GATE_INCOMPLETE$"):
                session.capture_fixture(BODY, **changes)
            self.assertFalse(any(p.suffix == ".raw" for p in self.directory.rglob("*")))
