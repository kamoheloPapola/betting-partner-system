"""Synthetic offline vertical slice of the isolated future-lineage lifecycle.

Live preparation/acquisition/capture is deliberately unavailable. New live
grants and independently accepted runtime closure are required before a later
operational implementation can be enabled. This module issues no such grant.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from src.evaluation import future_lineage_context as context
from src.evaluation.future_lineage_qualification import (
    candidate_availability, observation,
)


def reconcile(metadata, qualification, protocol):
    # Reuse the exact original league-wide predicate AND diagnostic. Its
    # current_events path normalizes only; no feature/model construction occurs.
    from src.evaluation.prospective_collection_next import _events
    return _events("BL1", metadata, qualification, protocol)


def unique_fixture_outcomes(rows):
    """Descriptive accounting only: never settles, imports or rewrites a pair."""
    outcomes, vintages = {}, set()
    for row in rows:
        require_fields = ("league", "season", "provider_fixture_id", "lineage_id", "vintage_id")
        context.require(all(k in row for k in require_fields), "FUTURE_OUTCOME_IDENTITY")
        key = tuple(row[k] for k in require_fields[:3])
        context.require(key[:2] == ("BL1", 2026) and type(key[2]) is int and key[2] > 0,
                        "FUTURE_OUTCOME_SCOPE")
        vintage = (row["lineage_id"], row["vintage_id"])
        context.require(vintage not in vintages, "FUTURE_DUPLICATE_VINTAGE")
        vintages.add(vintage)
        score = (row["home_goals"], row["away_goals"])
        context.require(all(type(n) is int and n >= 0 for n in score), "FUTURE_OUTCOME_SCORE")
        context.require(key not in outcomes or outcomes[key] == score, "FUTURE_OUTCOME_CONFLICT")
        outcomes[key] = score
    return {"unique_fixture_outcomes": len(outcomes), "forecast_vintages": len(vintages)}


@dataclass
class SyntheticSession:
    """An implementation test session, never a preparation/collection grant.

    Every input, prediction and output must carry the synthetic label. A
    production backend/permission is not accepted through this interface.
    """
    store: context.EvidenceStore
    operations: tuple[str, ...]
    bindings: dict
    clock: object = context.now

    def check(self, operation):
        context.require(self.store.mode == "synthetic" and self.store.access is not None,
                        "FUTURE_SYNTHETIC_SESSION_REQUIRED")
        context.require(operation in self.operations, "FUTURE_SIMULATION_SCOPE")
        at = self.clock()
        self.store.access.check(context.now())
        self.store.access.check(at)
        context.require(self.bindings.get("dependency_manifest") ==
                        self.store.access.anchor["manifest"], "FUTURE_SESSION_DEPENDENCY_BINDING")
        for name, ref in self.bindings.items():
            if name == "dependency_manifest":
                continue  # Independently approved manifest was just checked by access.check.
            path = context.reference(self.store.repository, ref)
            context.require(path.is_relative_to(self.store.root), "FUTURE_FIXTURE_OUTSIDE_TEST_ROOT")
        return at

    def load(self, name):
        path = context.reference(self.store.repository, self.bindings[name])
        value = context.read(path)
        context.require(value.get("execution_mode") == "synthetic", "FUTURE_SYNTHETIC_LABEL_REQUIRED")
        return value

    def qualify(self, rebuild):
        self.check("qualify")
        context.require(callable(rebuild), "FUTURE_RECONSTRUCTOR_REQUIRED")
        result = rebuild()
        context.require(result.get("execution_mode") == "synthetic" and
                        result["assessment"]["qualified"] is True, "CURRENT_DATA_UNQUALIFIED")
        # Readiness cannot create any capture operation or runtime grant.
        return result

    def journal(self):
        rows, previous = [], None
        directory = self.store.root / "BL1/journal"
        # Incomplete temp artifacts are retained for inspection and block reuse.
        context.require(not list(directory.glob("*.tmp")), "FUTURE_ORPHAN_PUBLICATION")
        for number, path in enumerate(sorted(directory.glob("*")), 1):
            context.require(path.name == f"{number:08d}.json", "FUTURE_JOURNAL_GAP")
            row = context.read(path)
            context.sealed(row)
            context.require(row.get("execution_mode") == "synthetic" and
                            row["sequence"] == number and row["previous"] == previous and
                            row["lineage_id"] == self.store.relative_root and
                            row["qualification"] == self.bindings["qualification"] and
                            row["protocol"] == self.bindings["protocol"], "FUTURE_JOURNAL_BINDING")
            receipt_path = self.store.path("BL1/attempt_outcomes/" + row["attempt_id"] + ".json")
            context.require(receipt_path.is_file(), "FUTURE_PUBLICATION_INCOMPLETE")
            receipt = context.read(receipt_path)
            context.sealed(receipt)
            context.require(receipt.get("execution_mode") == "synthetic" and
                            receipt["journal"] == context.identity(self.store.repository, path) and
                            receipt["status"] == "ADMITTED" and
                            context.instant(row["pair"]["created_at"]) <=
                            context.instant(receipt["journal_observed_complete_at"]) <
                            context.instant(row["pair"]["kickoff_at"]),
                            "FUTURE_PUBLICATION_NOT_PREKICKOFF")
            rows.append(row)
            previous = row["record_sha256"]
        context.require(len({r["pair"]["fixture_id"] for r in rows}) == len(rows),
                        "FUTURE_DUPLICATE_FIXTURE")
        return rows

    def _eligible(self, event, q, protocol, at):
        context.require(event["league"] == protocol["league"] == "BL1" and
                        event["season"] == protocol["season"] == 2026, "FUTURE_FIXTURE_SCOPE")
        horizon = self.store.access.decision["accepted_decisions"]["S3"]["reference_horizon"]
        context.require(all(protocol[key] == horizon[key] for key in
                            ("start_at", "end_at", "frozen_at")), "FUTURE_HORIZON_DRIFT")
        context.require(context.instant(protocol["start_at"]) <= context.instant(event["kickoff_at"]) <
                        context.instant(protocol["end_at"]) and
                        context.instant(at) < context.instant(event["kickoff_at"]) and
                        event["status"] in ("TIMED", "SCHEDULED"), "FUTURE_NOT_ELIGIBLE")
        targets = [e for e in q["assessment"]["upcoming"] if e["fixture_id"] == event["fixture_id"]]
        context.require(len(targets) == 1, "FUTURE_UNKNOWN_OR_DUPLICATE_TARGET")
        from src.evaluation.prospective_collection_next import IDENTITY
        context.require(all(event[k] == targets[0][k] for k in IDENTITY),
                        "FUTURE_TARGET_IDENTITY_CHANGED")
        return targets[0]

    def collect(self, *, date_from, date_to, backend):
        """Exercise ordering/publication only with isolated synthetic backends."""
        at = self.check("collect")
        context.require(getattr(backend, "execution_mode", None) == "synthetic",
                        "FUTURE_LIVE_BACKEND_DENIED")
        # Covered legacy dependencies may execute import-time code. Verify their
        # approval before importing them, including on an already-created session.
        from src.evaluation.prospective_collection_next import _dates, IDENTITY
        from src.ml.bundle_activation import state_lock
        q, protocol, fresh = (self.load(name) for name in ("qualification", "protocol", "fresh"))
        context.sealed(q, "qualification_sha256")
        # Verify identities and fresh evidence, then global gate BEFORE date
        # filtering, feature reconstruction, model calls, locks or journal reads.
        observation(fresh, self.store.repository, cutoff=at)
        events = reconcile(fresh, q, protocol)
        first, last = _dates(date_from, date_to)
        backend.verify_qualification(q)
        index = backend.source_index
        context.require(index.get("execution_mode") == "synthetic", "FUTURE_SYNTHETIC_INDEX_REQUIRED")
        accepted, already = [], []
        self.check("collect")
        with state_lock(self.store.path("BL1/operation.lock")):
            self.check("collect")
            rows = self.journal()
            existing = {row["pair"]["fixture_id"]: row for row in rows}
            for event in events:
                kickoff = context.instant(event["kickoff_at"])
                if not first <= kickoff.date() <= last:
                    continue
                at = self.check("collect")
                if event["fixture_id"] in existing:
                    row = existing[event["fixture_id"]]
                    context.require(all(event[k] == row["fresh_fixture"][k] for k in IDENTITY),
                                    "FUTURE_RECORDED_FIXTURE_REVISED")
                    already.append(event["fixture_id"])
                    continue
                target = self._eligible(event, q, protocol, at)
                context.require(context.instant(fresh["retrieved_at"]) < context.instant(at),
                                "FUTURE_FRESH_SOURCE_NOT_BEFORE_PREDICTION")
                candidate_availability(index, self.store.repository, at=at, kickoff=event["kickoff_at"])
                observation(fresh, self.store.repository, cutoff=at, kickoff=event["kickoff_at"])
                features = backend.features(q, target, at)
                # Shared finite/schema/causal validation must happen for EACH
                # candidate; first-target qualification proves no other target.
                from src.evaluation.phase10_current_data import validate_features
                matrix, fixture = features
                validate_features(matrix, fixture, q["ordered_features"], at)
                outputs = backend.predict(features, at)
                context.require(set(outputs) == {"active", "candidate"}, "FUTURE_PAIRED_OUTPUT_REQUIRED")
                for label, record in outputs.items():
                    context.require(record.get("execution_mode") == "synthetic" and
                                    record.get("model_id") == q[label + "_bundle_id"] and
                                    record.get("features") == fixture[q["ordered_features"]].tolist(),
                                    "FUTURE_MODEL_OR_FEATURE_BINDING")
                    probabilities = record.get("probabilities", [])
                    context.require(len(probabilities) == 3 and
                                    all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1
                                        for v in probabilities) and
                                    abs(sum(probabilities) - 1) <= 1e-12, "FUTURE_PROBABILITY_INVALID")
                context.require(outputs["active"].get("rng") == outputs["candidate"].get("rng")
                                == protocol["rng"], "FUTURE_PAIRED_RNG_MISMATCH")
                backend.verify_outputs(outputs, features, at)
                publication_at = self.check("collect")
                self._eligible(event, q, protocol, publication_at)
                candidate_availability(index, self.store.repository, at=publication_at,
                                       kickoff=event["kickoff_at"])
                sequence = len(rows) + 1
                attempt = f"capture-{sequence:08d}"
                pair = context.seal({"execution_mode": "synthetic",
                    "fixture_id": event["fixture_id"], "created_at": at,
                    "kickoff_at": event["kickoff_at"], "outputs": outputs,
                    "vintage_id": context.fingerprint([self.store.relative_root,
                        self.bindings["qualification"], event["fixture_id"]])}, "pair_sha256")
                row = context.seal({"execution_mode": "synthetic", "sequence": sequence,
                    "previous": rows[-1]["record_sha256"] if rows else None,
                    "lineage_id": self.store.relative_root, "attempt_id": attempt,
                    "qualification": self.bindings["qualification"], "protocol": self.bindings["protocol"],
                    "fresh_source": self.bindings["fresh"], "fresh_fixture": event, "pair": pair})
                row_ref = self.store.publish(f"BL1/journal/{sequence:08d}.json", row,
                                             at=publication_at)
                # Completion time is observed AFTER fsync, atomic publication
                # and readback. A late/incomplete record is never admitted.
                completed = self.check("collect")
                valid = context.instant(completed) < kickoff
                self.store.publish("BL1/attempt_outcomes/" + attempt + ".json",
                    context.seal({"execution_mode": "synthetic", "journal": row_ref,
                        "journal_observed_complete_at": completed,
                        "status": "ADMITTED" if valid else "LATE_PUBLICATION"}), at=completed)
                context.require(valid, "FUTURE_LATE_PUBLICATION")
                rows.append(row)
                existing[event["fixture_id"]] = row
                accepted.append(event["fixture_id"])
        return {"execution_mode": "synthetic", "accepted": accepted, "already_recorded": already,
                "recorded": len(rows), "journal_head": rows[-1]["record_sha256"] if rows else None,
                "preparation_permission": False, "collection_authorization": False}

    def verify(self, backend):
        self.check("verify")
        q, protocol, fresh = (self.load(name) for name in ("qualification", "protocol", "fresh"))
        events = reconcile(fresh, q, protocol)
        backend.verify_qualification(q)
        rows = self.journal()
        for row in rows:
            context.require(row["fresh_source"] == self.bindings["fresh"], "FUTURE_ARCHIVED_SOURCE_MISMATCH")
            matches = [e for e in events if e["fixture_id"] == row["pair"]["fixture_id"]]
            context.require(len(matches) == 1 and matches[0] == row["fresh_fixture"],
                            "FUTURE_CAPTURE_SOURCE_REPLAY")
            at = row["pair"]["created_at"]
            self._eligible(matches[0], q, protocol, at)
            candidate_availability(backend.source_index, self.store.repository,
                                   at=at, kickoff=row["pair"]["kickoff_at"])
            features = backend.features(q, matches[0], at)
            backend.verify_outputs(row["pair"]["outputs"], features, at)
        return {"execution_mode": "synthetic", "verified": True, "recorded": len(rows)}
