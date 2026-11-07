"""Isolated approval, permission, storage and denied-effect contract tests."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import uuid

import pytest

from src.evaluation import future_lineage_context as c

REPOSITORY = Path(__file__).resolve().parents[2]
AT = "2026-10-15T09:00:00Z"
HORIZON = {"start_at": "2026-09-23T01:30:23.492703+00:00",
           "end_at": "2027-05-23T00:00:00+00:00",
           "frozen_at": "2026-09-23T01:30:21.492703+00:00"}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, allow_nan=False)


def manifest_fixture(repository, directory):
    sources, unresolved = c.dependency_requirements(repository)
    sources = sorted(set(sources) | set(c.ROLE_SOURCES.values()))
    entries = []
    for name in sources:
        roles = [role for role, path in c.ROLE_SOURCES.items() if path == name]
        entries.append({**c.identity(repository, repository / name),
                        "roles": roles or ["source:" + name.replace("/", ".").lower()]})
    for role in sorted(c.REQUIRED_ROLES - c.ROLE_SOURCES.keys()):
        path = directory / (role + ".fixture.json")
        save(path, {"execution_mode": "synthetic", "role": role,
                    "value": "synthetic dependency bytes; not a deployed runtime or model"})
        entries.append({**c.identity(repository, path), "roles": [role]})
    path = directory / "dependency_manifest.json"
    with path.open("xb") as stream:
        stream.write(c.canonical({"schema_version": c.MANIFEST_SCHEMA, "execution_mode": "synthetic",
            "runtime_closure": "UNQUALIFIED", "entries": entries, "unresolved": unresolved}))
    return c.identity(repository, path)


def make_authority(directory, *, repository=None, minimal=False):
    """Entirely synthetic authority fixtures; these issue no real permission."""
    repository = repository or REPOSITORY
    directory.mkdir()
    if minimal:
        for name in sorted(set(c.NEW_CODE) | set(c.ROLE_SOURCES.values())):
            path = repository / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# Synthetic retained dependency, never executed.\n", encoding="utf-8")
    manifest = manifest_fixture(repository, directory)
    refs = []
    for i in range(18):
        path = directory / f"bound-{i}.json"
        save(path, {"execution_mode": "synthetic", "ordinal": i})
        refs.append(c.identity(repository, path))
    candidate = {"execution_mode": "synthetic",
        "decisions": {f"S{i}": {"synthetic_fixture": True} for i in range(1, 8)},
        "bound_artifacts": refs[:13], "reviewed_design_documents": refs[13:],
        "preserved_phase20_boundary": refs[0],
        "owner": {"proposed_label": "Administrator — identity verification pending"},
        "issuer": {"proposed_label": "Administrator — identity verification pending"}}
    candidate["decisions"]["S3"]["reference_horizon"] = HORIZON
    candidate_path = directory / "candidate.json"
    save(candidate_path, candidate)
    procedure_path = directory / "procedure.json"
    save(procedure_path, {"execution_mode": "synthetic", "procedure": "TEST ONLY"})
    candidate_ref, procedure_ref = (c.identity(repository, p)
                                    for p in (candidate_path, procedure_path))
    user_text = ("i personally adopt and issue the completed statement. "
                 "SYNTHETIC TEST FIXTURE, NOT A HUMAN APPROVAL.")
    event_path = directory / "human-event.json"
    save(event_path, {"type": "response_item", "payload": {"role": "user", "id": "test-user",
        "content": [{"type": "input_text", "text": user_text}]}})
    text_path = directory / "human.txt"
    text_path.write_text(user_text, encoding="utf-8")
    brief_path = directory / "brief.txt"
    brief_path.write_text("SYNTHETIC TEST ONLY\n" + candidate_ref["sha256"] + "\n" +
                          procedure_ref["sha256"], encoding="utf-8")
    draft_path = directory / "statement.txt"
    draft_path.write_text("SYNTHETIC TEST ONLY", encoding="utf-8")
    evidence = {"retained_event": c.identity(repository, event_path),
        "original_message_id": "test-user", "verbatim_user_text": c.identity(repository, text_path),
        "original_uploaded_brief": {"retained_bytes": c.identity(repository, brief_path)},
        "adopted_statement": c.identity(repository, draft_path)}
    common = {"execution_mode": "synthetic", "candidate": candidate_ref, "procedure": procedure_ref,
        "human_evidence": evidence,
        "bound_artifacts": [{**ref, "verified": True} for ref in refs],
        "trust": {"declared_owner": candidate["owner"]["proposed_label"],
            "declared_issuer": candidate["issuer"]["proposed_label"],
            "independent_identity_verified": False,
            "cryptographic_signature_or_non_repudiation_claimed": False,
            "phase20_authority_reused": False},
        "preparation_permission": False, "collection_authorization": False,
        "dependency_manifest": manifest}
    decision = {**common, "schema_version": "ffps.future_lineage.decision_acceptance.v1",
        "decision_approval": True, "exact_candidate_S1_S7_accepted": True,
        "accepted_decisions": candidate["decisions"]}
    decision_path = directory / "decision.json"
    save(decision_path, decision)
    decision_ref = c.identity(repository, decision_path)
    grant = {**common, "schema_version": "ffps.future_lineage.implementation_permission.v1",
        "implementation_permission": True, "decision_acceptance": decision_ref,
        "valid_from_utc": "2026-10-15T06:00:00Z",
        "valid_until_utc_exclusive": "2026-10-16T06:00:00Z",
        "human_adoption_event_time_utc": "2026-10-15T08:12:54.303Z",
        "new_source_paths": list(c.NEW_CODE), "new_test_paths": list(c.NEW_TESTS),
        "test_output_directory": c.TEST_ROOT, "runtime_root_must_remain_uncreated": c.RUNTIME_ROOT,
        "existing_source_test_modifications_permitted": False, "withheld_operations": list(c.DENIED)}
    grant_path = directory / "grant.json"
    save(grant_path, grant)
    grant_ref = c.identity(repository, grant_path)
    save(directory / "trusted-anchor.json", {"schema_version": c.ANCHOR_SCHEMA,
        "decision": decision_ref, "grant": grant_ref, "manifest": manifest})
    kwargs = {"decision_pin": decision_ref["sha256"], "grant_pin": grant_ref["sha256"],
              "candidate_pin": candidate_ref["sha256"], "procedure_pin": procedure_ref["sha256"],
              "at": AT}
    return decision_ref, grant_ref, kwargs


def protected_snapshot(path=None):
    return {"elevated": False, "administrator": False, "dangerous_privileges": [],
        "application_account": c.APPLICATION_ACCOUNT,
        "user_sid": "S-1-5-21-1-2-3-1001", "expected_user_sid": "S-1-5-21-1-2-3-1001",
        "paths": [{"path": str(n), "protected": n >= 2, "owner": "S-1-5-32-544",
                   "dacl_protected": True, "effective_access": 0x120089,
                   "aces": [{"sid": "S-1-5-32-544", "mask": 0x1F01FF, "flags": 0},
                            {"sid": "S-1-5-18", "mask": 0x1F01FF, "flags": 0},
                            {"sid": "synthetic-reader", "mask": 0x120089, "flags": 0}]}
                  for n in range(6)]}


@pytest.fixture
def authority(tmp_path, monkeypatch):
    result = make_authority(tmp_path / "authority")
    monkeypatch.setattr(c, "TRUST_ANCHOR", tmp_path / "authority/trusted-anchor.json")
    monkeypatch.setattr(c, "_windows_anchor_snapshot", protected_snapshot)
    monkeypatch.setattr(c, "now", lambda: AT)  # Synthetic clock, not an expired real grant.
    return result


@pytest.fixture
def sandbox(authority):
    d, g, kwargs = authority
    access = c.verify_implementation(REPOSITORY, d, g, **kwargs)
    relative = c.TEST_ROOT + "/fixtures/" + uuid.uuid4().hex
    store = c.EvidenceStore(REPOSITORY, relative, "synthetic", access)
    store.create(at=AT)
    return store


def test_exact_external_approval_and_separate_permissions(authority):
    d, g, kwargs = authority
    access = c.verify_implementation(REPOSITORY, d, g, **kwargs)
    assert access.grant["preparation_permission"] is False
    assert access.grant["collection_authorization"] is False
    assert not (REPOSITORY / c.RUNTIME_ROOT).exists()


@pytest.mark.parametrize("pin", ["decision_pin", "grant_pin", "candidate_pin", "procedure_pin"])
def test_absent_or_wrong_external_pin_blocks(authority, pin):
    d, g, kwargs = authority
    kwargs[pin] = "0" * 64
    with pytest.raises(ValueError, match="EXTERNAL_PIN"):
        c.verify_implementation(REPOSITORY, d, g, **kwargs)


def test_self_resealed_replacement_has_no_authority(authority):
    d, g, kwargs = authority
    path = REPOSITORY / d["path"]
    value = c.read(path)
    value["decision_approval"] = False
    path.write_text(json.dumps(c.seal(value)), encoding="utf-8")
    with pytest.raises(ValueError, match="REFERENCE_DRIFT"):
        c.verify_implementation(REPOSITORY, d, g, **kwargs)


@pytest.mark.parametrize("mutation,reason", [
    ("scope", "IMPLEMENTATION_SCOPE"), ("prepare", "PERMISSION_CONFLATION"),
    ("role", "DECLARED_ROLE"), ("signature", "TRUST_MODEL"),
    ("design", "GRANT_DESIGN_BINDINGS"), ("no_decision", "DECISION_REQUIRED"),
])
def test_even_externally_pinned_bad_contract_fails(authority, mutation, reason):
    d, g, kwargs = authority
    decision, grant = c.read(REPOSITORY/d["path"]), c.read(REPOSITORY/g["path"])
    if mutation == "scope":
        grant["new_source_paths"].append("src/evaluation/forbidden.py")
    elif mutation == "prepare":
        grant["preparation_permission"] = True
    elif mutation == "role":
        grant["trust"]["declared_issuer"] = "inferred Windows account"
    elif mutation == "signature":
        grant["trust"]["independent_identity_verified"] = True
    elif mutation == "design":
        grant["bound_artifacts"] = []
    elif mutation == "no_decision":
        decision["exact_candidate_S1_S7_accepted"] = False
    # Rebuild fixture pins to exercise semantic checks beyond basic byte drift.
    (REPOSITORY/d["path"]).write_text(json.dumps(decision), encoding="utf-8")
    d = c.identity(REPOSITORY, REPOSITORY/d["path"])
    grant["decision_acceptance"] = d
    (REPOSITORY/g["path"]).write_text(json.dumps(grant), encoding="utf-8")
    g = c.identity(REPOSITORY, REPOSITORY/g["path"])
    kwargs.update(decision_pin=d["sha256"], grant_pin=g["sha256"])
    anchor = c.read(c.TRUST_ANCHOR)
    anchor.update(decision=d, grant=g)
    c.TRUST_ANCHOR.write_bytes(c.canonical(anchor))
    with pytest.raises(ValueError, match=reason):
        c.verify_implementation(REPOSITORY, d, g, **kwargs)


@pytest.mark.parametrize("at,reason", [
    ("2026-10-15T05:59:59Z", "EXPIRED"),
    ("2026-10-15T07:00:00Z", "PRE_ISSUANCE"),
    ("2026-10-16T06:00:00Z", "EXPIRED"),
])
def test_literal_window_and_no_backdating(authority, at, reason):
    d, g, kwargs = authority
    kwargs["at"] = at
    with pytest.raises(ValueError, match=reason):
        c.verify_implementation(REPOSITORY, d, g, **kwargs)


@pytest.mark.parametrize("path", ["../escape", "/absolute", "C:/escape", "x\\escape",
                                  "x//y", "x/../y", "x/NUL", "x/name:stream", "x/trailing."])
def test_unsafe_paths_rejected(tmp_path, path):
    with pytest.raises(ValueError, match="PATH"):
        c.safe_path(tmp_path, path)


def test_root_collision_and_finite_slots(sandbox):
    with pytest.raises(ValueError, match="ROOT_COLLISION"):
        sandbox.create(at=AT)
    for slot in ("../history.json", "BL1/journal/1.json", "unexpected.json",
                 "BL1/qualification/attempt-000001/raw_sources/unknown.raw"):
        with pytest.raises(ValueError, match="UNDECLARED_SLOT"):
            sandbox.path(slot)


def test_immutable_publication_and_collision(sandbox):
    row = {"execution_mode": "synthetic", "value": 1}
    sandbox.publish("lineage_specification.json", row, at=AT)
    before = sandbox.path("lineage_specification.json").read_bytes()
    with pytest.raises(ValueError, match="COLLISION"):
        sandbox.publish("lineage_specification.json", {**row, "value": 2}, at=AT)
    assert sandbox.path("lineage_specification.json").read_bytes() == before


def test_symlink_is_rejected(sandbox):
    target = sandbox.root / "actual"
    target.mkdir()
    link = sandbox.root / "link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Filesystem does not permit creating a symlink for this test")
    with pytest.raises(ValueError, match="REPARSE_POINT"):
        c.safe_path(sandbox.repository, sandbox.relative_root + "/link/file")


@pytest.mark.parametrize("operation", c.DENIED)
def test_live_operations_have_no_permission_or_effect(operation):
    before = (REPOSITORY / c.RUNTIME_ROOT).exists()
    with pytest.raises(ValueError, match="LIVE_PERMISSION_WITHHELD"):
        c.deny_live(operation)
    assert (REPOSITORY / c.RUNTIME_ROOT).exists() == before


def test_live_store_denies_before_directory_creation(authority):
    d, g, kwargs = authority
    access = c.verify_implementation(REPOSITORY, d, g, **kwargs)
    store = c.EvidenceStore(REPOSITORY, c.RUNTIME_ROOT, "live", access)
    with pytest.raises(ValueError, match="LIVE_PREPARATION_NOT_IMPLEMENTED"):
        store.create(at=AT)
    assert not store.root.exists()


def test_duplicate_and_nonfinite_json_block(tmp_path):
    path = tmp_path/"bad.json"
    path.write_text('{"x":1,"x":2}', encoding="utf-8")
    with pytest.raises(ValueError, match="DUPLICATE_JSON_KEY"):
        c.read(path)
    path.write_text('{"x":NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="NONFINITE_JSON"):
        c.read(path)


def test_simulation_clock_cannot_extend_actual_implementation_permission(sandbox, monkeypatch):
    monkeypatch.setattr(c, "now", lambda: "2026-10-16T06:00:00Z")
    with pytest.raises(ValueError, match="PERMISSION_EXPIRED"):
        sandbox.publish("lineage_specification.json", {"execution_mode": "synthetic"}, at=AT)
    assert not sandbox.path("lineage_specification.json").exists()


@pytest.fixture
def t3_repository(tmp_path, monkeypatch):
    root = tmp_path / "isolated-repository"
    root.mkdir()
    result = make_authority(root / "authority", repository=root, minimal=True)
    monkeypatch.setattr(c, "TRUST_ANCHOR", root / "authority/trusted-anchor.json")
    monkeypatch.setattr(c, "_windows_anchor_snapshot", protected_snapshot)
    monkeypatch.setattr(c, "now", lambda: AT)
    return root, result


def repin_fixture(root, authority, *, manifest=None):
    """Simulate explicit new owner pins to test semantics beyond byte drift."""
    d, g, kwargs = authority
    anchor = c.read(c.TRUST_ANCHOR)
    if manifest is not None:
        path = root / anchor["manifest"]["path"]
        path.write_bytes(c.canonical(manifest))
        anchor["manifest"] = c.identity(root, path)
    decision = c.read(root / d["path"])
    decision["dependency_manifest"] = anchor["manifest"]
    (root / d["path"]).write_bytes(c.canonical(decision))
    d = c.identity(root, root / d["path"])
    grant = c.read(root / g["path"])
    grant.update(decision_acceptance=d, dependency_manifest=anchor["manifest"])
    (root / g["path"]).write_bytes(c.canonical(grant))
    g = c.identity(root, root / g["path"])
    anchor.update(decision=d, grant=g)
    c.TRUST_ANCHOR.write_bytes(c.canonical(anchor))
    return d, g, {**kwargs, "decision_pin": d["sha256"], "grant_pin": g["sha256"]}


def verify_fixture(t3_repository):
    root, (d, g, kwargs) = t3_repository
    return c.verify_implementation(root, d, g, **kwargs)


def test_t3_positive_binding_and_boundary_recheck(t3_repository):
    access = verify_fixture(t3_repository)
    access.check(AT)
    assert access.manifest["runtime_closure"] == "UNQUALIFIED"
    assert "native_dynamic_runtime" in access.manifest["unresolved"]
    assert access.decision["dependency_manifest"] == access.anchor["manifest"]


@pytest.mark.parametrize("role", sorted(c.REQUIRED_ROLES) + ["source:src.evaluation.future_lineage.py"])
def test_t3_each_covered_dependency_change_invalidates_approval(t3_repository, role):
    root, _ = t3_repository
    access = verify_fixture(t3_repository)
    entry = next(e for e in access.manifest["entries"] if role in e["roles"])
    path = root / entry["path"]
    path.write_bytes(path.read_bytes() + b"\nchanged synthetic dependency")
    with pytest.raises(ValueError, match="REFERENCE_DRIFT"):
        access.check(AT)
    with pytest.raises(ValueError, match="REFERENCE_DRIFT"):
        verify_fixture(t3_repository)


@pytest.mark.parametrize("role", sorted(c.REQUIRED_ROLES))
def test_t3_each_required_role_must_be_present(t3_repository, role):
    root, authority = t3_repository
    access = verify_fixture(t3_repository)
    body = copy.deepcopy(access.manifest)
    for entry in body["entries"]:
        if role in entry["roles"]:
            entry["roles"].remove(role)
    body["entries"] = [e for e in body["entries"] if e["roles"]]
    t3_repository = root, repin_fixture(root, authority, manifest=body)
    with pytest.raises(ValueError, match="MEMBER_MISSING"):
        verify_fixture(t3_repository)


@pytest.mark.parametrize("change", ["bytes", "missing", "unreadable"])
def test_t3_manifest_retained_bytes_are_exact(t3_repository, monkeypatch, change):
    root, _ = t3_repository
    access = verify_fixture(t3_repository)
    path = root / access.anchor["manifest"]["path"]
    if change == "bytes":
        path.write_bytes(path.read_bytes() + b" ")
    elif change == "missing":
        path.unlink()
    else:
        original = Path.open
        def denied(self, *args, **kwargs):
            if self == path:
                raise PermissionError("synthetic unreadable manifest")
            return original(self, *args, **kwargs)
        monkeypatch.setattr(Path, "open", denied)
    with pytest.raises((ValueError, OSError)):
        access.check(AT)


@pytest.mark.parametrize("mutation,reason", [
    ("duplicate_role", "DUPLICATE_ROLE"), ("duplicate_path", "DUPLICATE_PATH"),
    ("case_alias", "DUPLICATE_PATH"), ("traversal", "PATH"),
    ("absolute", "PATH"), ("missing_file", "REFERENCE_MISSING"),
    ("wrong_digest", "REFERENCE_DRIFT"), ("wrong_size", "REFERENCE_DRIFT"),
    ("boolean_size", "ENTRY_SCHEMA"), ("wrong_schema", "MANIFEST_SCHEMA"),
    ("unknown_hidden", "UNKNOWN_HIDDEN"), ("duplicate_unknown", "UNKNOWN_HIDDEN"),
    ("claimed_qualified", "MANIFEST_SCHEMA"), ("wrong_role_path", "ROLE_PATH"),
])
def test_t3_manifest_semantics_reject_even_with_new_fixture_pins(t3_repository, mutation, reason):
    root, authority = t3_repository
    access = verify_fixture(t3_repository)
    body = copy.deepcopy(access.manifest)
    entry = next(e for e in body["entries"] if "model" in e["roles"])
    if mutation == "duplicate_role":
        entry["roles"].append("model")
    elif mutation in ("duplicate_path", "case_alias"):
        other = {**entry, "roles": ["extra:alias"]}
        if mutation == "case_alias":
            other["path"] = other["path"].upper()
        body["entries"].append(other)
    elif mutation == "traversal":
        entry["path"] = "../escape"
    elif mutation == "absolute":
        entry["path"] = "C:/escape"
    elif mutation == "missing_file":
        entry["path"] = "absent.json"
    elif mutation == "wrong_digest":
        entry["sha256"] = "0" * 64
    elif mutation == "wrong_size":
        entry["size_bytes"] += 1
    elif mutation == "boolean_size":
        entry["size_bytes"] = True
    elif mutation == "wrong_schema":
        body["schema_version"] = "not-approved"
    elif mutation == "unknown_hidden":
        body["unresolved"] = []
    elif mutation == "duplicate_unknown":
        body["unresolved"].append(body["unresolved"][0])
    elif mutation == "claimed_qualified":
        body["runtime_closure"] = "QUALIFIED"
    elif mutation == "wrong_role_path":
        body["entries"][0]["roles"].append("verifier")
        for e in body["entries"][1:]:
            e["roles"] = [r for r in e["roles"] if r != "verifier"]
    updated = root, repin_fixture(root, authority, manifest=body)
    with pytest.raises(ValueError, match=reason):
        verify_fixture(updated)


def test_t3_missing_dependency_file_and_unreadable_member(t3_repository, monkeypatch):
    root, _ = t3_repository
    access = verify_fixture(t3_repository)
    entry = next(e for e in access.manifest["entries"] if "model" in e["roles"])
    path = root / entry["path"]
    original = Path.open
    def denied(self, *args, **kwargs):
        if self == path:
            raise PermissionError("synthetic unreadable dependency")
        return original(self, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", denied)
        with pytest.raises(PermissionError):
            access.check(AT)
    path.unlink()
    with pytest.raises(ValueError, match="REFERENCE_MISSING"):
        access.check(AT)


def test_t3_transitive_imports_packages_dynamic_and_runtime_reads(t3_repository):
    root, _ = t3_repository
    (root / "src/__init__.py").write_text("# package\n", encoding="utf-8")
    entry = root / "scripts/run_future_lineage.py"
    entry.write_text("from src.evaluation import indirect\n", encoding="utf-8")
    (root / "src/evaluation/indirect.py").write_text(
        "from . import deeper\nimport importlib\nimportlib.import_module('optional_plugin')\n"
        "open('runtime-selected.json')\n", encoding="utf-8")
    (root / "src/evaluation/deeper.py").write_text("# nested dependency\n", encoding="utf-8")
    (root / "src/evaluation/literal_plugin.py").write_text("# dynamically named dependency\n", encoding="utf-8")
    with (root / "src/evaluation/indirect.py").open("a", encoding="utf-8") as stream:
        stream.write("importlib.import_module('src.evaluation.literal_plugin')\n")
    paths, unresolved = c.dependency_requirements(root)
    assert {"src/__init__.py", "src/evaluation/indirect.py", "src/evaluation/deeper.py",
            "src/evaluation/literal_plugin.py"} <= set(paths)
    assert any("import_module" in item for item in unresolved)
    assert any(":open" in item for item in unresolved)
    module = root / "src/evaluation/indirect"
    module.mkdir()
    (module / "__init__.py").write_text("# ambiguous resolution\n", encoding="utf-8")
    with pytest.raises(ValueError, match="AMBIGUOUS_MODULE"):
        c.dependency_requirements(root)


def test_t3_required_local_source_omission_rejected(t3_repository):
    root, authority = t3_repository
    access = verify_fixture(t3_repository)
    # This file has no separately required logical role; closure membership still applies.
    path = root / "src/evaluation/extra.py"
    path.write_text("# synthetic transitive module\n", encoding="utf-8")
    caller = root / "scripts/run_future_lineage.py"
    caller.write_text("from src.evaluation import extra\n", encoding="utf-8")
    body = copy.deepcopy(access.manifest)
    for entry in body["entries"]:
        if entry["path"] == "scripts/run_future_lineage.py":
            entry.update(c.identity(root, caller))
    _, body["unresolved"] = c.dependency_requirements(root)
    updated = root, repin_fixture(root, authority, manifest=body)
    with pytest.raises(ValueError, match="MEMBER_MISSING"):
        verify_fixture(updated)


def test_t3_different_or_old_approval_cannot_be_selected(t3_repository):
    root, authority = t3_repository
    access = verify_fixture(t3_repository)
    other = make_authority(root / "other", repository=root)
    with pytest.raises(ValueError, match="TRUSTED_APPROVAL_MISMATCH"):
        verify_fixture((root, other))
    # Changing the trusted source invalidates an existing access object, without fallback.
    c.TRUST_ANCHOR.write_bytes((root / "other/trusted-anchor.json").read_bytes())
    with pytest.raises(ValueError, match="TRUSTED_APPROVAL_DRIFT"):
        access.check(AT)


@pytest.mark.parametrize("mutation", ["absent", "unreadable", "malformed", "duplicate_key",
                                    "missing_field", "wrong_schema", "extra_field"])
def test_t3_anchor_failure_has_no_creation_or_fallback(t3_repository, monkeypatch, mutation):
    root, authority = t3_repository
    if mutation == "absent":
        c.TRUST_ANCHOR.unlink()
    elif mutation == "unreadable":
        def denied(path):
            raise PermissionError("synthetic security-query denial")
        monkeypatch.setattr(c, "_windows_anchor_snapshot", denied)
    elif mutation in ("malformed", "duplicate_key"):
        c.TRUST_ANCHOR.write_text("{" if mutation == "malformed" else '{"x":1,"x":2}', encoding="utf-8")
    else:
        anchor = c.read(c.TRUST_ANCHOR)
        if mutation == "missing_field":
            del anchor["manifest"]
        elif mutation == "wrong_schema":
            anchor["schema_version"] = "wrong"
        else:
            anchor["caller_override"] = True
        c.TRUST_ANCHOR.write_bytes(c.canonical(anchor))
    before = c.TRUST_ANCHOR.read_bytes() if c.TRUST_ANCHOR.exists() else None
    with pytest.raises((ValueError, OSError)):
        verify_fixture((root, authority))
    assert (c.TRUST_ANCHOR.read_bytes() if c.TRUST_ANCHOR.exists() else None) == before


@pytest.mark.parametrize("failure,reason", [
    ("elevated", "PRIVILEGED_TOKEN"), ("administrator", "PRIVILEGED_TOKEN"),
    ("privilege", "PRIVILEGED_TOKEN"), ("owner", "OWNER_OR_INHERITANCE"),
    ("inheritance", "OWNER_OR_INHERITANCE"), ("writer", "UNTRUSTED_WRITER"),
    ("write_access", "APPLICATION_CAN_WRITE"), ("parent_replace", "APPLICATION_CAN_WRITE"),
    ("missing_system_grant", "PROVISIONER_GRANT_MISSING"),
    ("missing_administrator_grant", "PROVISIONER_GRANT_MISSING"),
    ("missing_path", "SECURITY_PATHS"),
])
def test_t3_controlled_acl_and_token_fail_closed(t3_repository, monkeypatch, failure, reason):
    snapshot = protected_snapshot()
    if failure in ("elevated", "administrator"):
        snapshot[failure] = True
    elif failure == "privilege":
        snapshot["dangerous_privileges"] = ["SeRestorePrivilege"]
    elif failure == "owner":
        snapshot["paths"][-1]["owner"] = "synthetic-reader"
    elif failure == "inheritance":
        snapshot["paths"][-1]["dacl_protected"] = False
    elif failure == "writer":
        snapshot["paths"][-1]["aces"].append({"sid": "synthetic-reader", "mask": 0x40000000,
                                                "flags": 0})
    elif failure == "write_access":
        snapshot["paths"][-1]["effective_access"] |= 0x40000
    elif failure == "parent_replace":
        snapshot["paths"][0]["effective_access"] |= 0x40
    elif failure in ("missing_system_grant", "missing_administrator_grant"):
        sid = "S-1-5-18" if failure == "missing_system_grant" else "S-1-5-32-544"
        snapshot["paths"][-1]["aces"] = [ace for ace in snapshot["paths"][-1]["aces"] if ace["sid"] != sid]
    else:
        snapshot["paths"].pop()
    monkeypatch.setattr(c, "_windows_anchor_snapshot", lambda path: snapshot)
    with pytest.raises(ValueError, match=reason):
        verify_fixture(t3_repository)


def test_t3_simulated_anchor_reparse_and_security_change_reject(t3_repository, monkeypatch):
    from types import SimpleNamespace
    original = Path.lstat
    def reparse(self, *args, **kwargs):
        info = original(self, *args, **kwargs)
        if self == c.TRUST_ANCHOR:
            return SimpleNamespace(st_mode=info.st_mode,
                                   st_file_attributes=c.stat.FILE_ATTRIBUTE_REPARSE_POINT)
        return info
    with monkeypatch.context() as patch:
        patch.setattr(Path, "lstat", reparse)
        with pytest.raises(ValueError, match="REPARSE_POINT"):
            c._anchor_paths(c.TRUST_ANCHOR)
    snapshots = [protected_snapshot(), protected_snapshot()]
    snapshots[-1]["paths"][-1]["aces"].reverse()
    monkeypatch.setattr(c, "_windows_anchor_snapshot", lambda path: snapshots.pop(0))
    with pytest.raises(ValueError, match="ANCHOR_CHANGED"):
        verify_fixture(t3_repository)


def test_t3_approval_record_must_bind_manifest(t3_repository):
    root, authority = t3_repository
    d, g, kwargs = authority
    value = c.read(root / d["path"])
    del value["dependency_manifest"]
    (root / d["path"]).write_bytes(c.canonical(value))
    d = c.identity(root, root / d["path"])
    value = c.read(root / g["path"])
    value["decision_acceptance"] = d
    (root / g["path"]).write_bytes(c.canonical(value))
    g = c.identity(root, root / g["path"])
    anchor = c.read(c.TRUST_ANCHOR)
    anchor.update(decision=d, grant=g)
    c.TRUST_ANCHOR.write_bytes(c.canonical(anchor))
    with pytest.raises(ValueError, match="APPROVAL_MANIFEST_BINDING"):
        c.verify_implementation(root, d, g, **{**kwargs, "decision_pin": d["sha256"], "grant_pin": g["sha256"]})


def test_t3_cli_cannot_override_source_and_ignores_environment(authority, monkeypatch):
    import importlib.util
    root = REPOSITORY
    d, g, pins = authority
    path = root / "scripts/run_future_lineage.py"
    spec = importlib.util.spec_from_file_location("t3_cli_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = ["inspect-implementation", "--decision", d["path"], "--decision-sha256", pins["decision_pin"],
            "--implementation-grant", g["path"], "--implementation-grant-sha256", pins["grant_pin"],
            "--candidate-sha256", pins["candidate_pin"], "--procedure-sha256", pins["procedure_pin"]]
    monkeypatch.setenv("FFPS_T3_TRUST_ANCHOR", "caller-selected.json")
    assert module.main(args) == 0
    c.TRUST_ANCHOR.unlink()
    assert module.main(args) == 2
    with pytest.raises(SystemExit):
        module.main(args + ["--trust-anchor", "caller-selected.json"])


@pytest.mark.parametrize("raw,reason", [(b'{"x":1,"x":2}', "DUPLICATE_JSON_KEY"),
                                      (b'{"x":NaN}', "NONFINITE_JSON"),
                                      (b'{', "Expecting property"),
                                      (b'{} ', "NONCANONICAL_BYTES")])
def test_t3_reapproved_malformed_manifest_still_rejects(t3_repository, raw, reason):
    root, authority = t3_repository
    anchor = c.read(c.TRUST_ANCHOR)
    path = root / anchor["manifest"]["path"]
    path.write_bytes(raw)
    anchor["manifest"] = c.identity(root, path)
    c.TRUST_ANCHOR.write_bytes(c.canonical(anchor))
    updated = root, repin_fixture(root, authority)
    with pytest.raises(ValueError, match=reason):
        verify_fixture(updated)


def test_t3_postapproval_reparse_and_interrupted_read_reject(t3_repository, monkeypatch):
    from types import SimpleNamespace
    root, _ = t3_repository
    access = verify_fixture(t3_repository)
    entry = next(e for e in access.manifest["entries"] if "model" in e["roles"])
    path = root / entry["path"]
    original = Path.lstat
    def reparse(self, *args, **kwargs):
        info = original(self, *args, **kwargs)
        if self == path:
            return SimpleNamespace(st_mode=info.st_mode,
                                   st_file_attributes=c.stat.FILE_ATTRIBUTE_REPARSE_POINT)
        return info
    with monkeypatch.context() as patch:
        patch.setattr(Path, "lstat", reparse)
        with pytest.raises(ValueError, match="REPARSE_POINT"):
            access.check(AT)
    original_open = Path.open
    def interrupted(self, *args, **kwargs):
        if self == path:
            raise InterruptedError("synthetic interrupted dependency read")
        return original_open(self, *args, **kwargs)
    monkeypatch.setattr(Path, "open", interrupted)
    with pytest.raises(InterruptedError):
        access.check(AT)


def test_t3_native_query_error_adapter_fails_without_real_windows_calls(t3_repository, monkeypatch):
    import ctypes
    calls = []
    class Function:
        def __init__(self, name):
            self.name = name
        def __call__(self, *args):
            calls.append(self.name)
            return 0
    class DLL:
        def __init__(self):
            self.functions = {}
        def __getattr__(self, name):
            return self.functions.setdefault(name, Function(name))
    monkeypatch.setattr(ctypes, "WinDLL", lambda *args, **kwargs: DLL())
    monkeypatch.setattr(ctypes, "get_last_error", lambda: 5)
    # The fixture patches the public security adapter; explicitly exercise the
    # original function with mock DLLs, never native token/ACL operations.
    import ast
    module = ast.parse(Path(c.__file__).read_text(encoding="utf-8"))
    function = next(node for node in module.body if isinstance(node, ast.FunctionDef)
                    and node.name == "_windows_anchor_snapshot")
    namespace = dict(vars(c))
    exec(compile(ast.Module(body=[function], type_ignores=[]), c.__file__, "exec"), namespace)
    with pytest.raises(ValueError, match="IMPERSONATION_OR_TOKEN_QUERY"):
        namespace["_windows_anchor_snapshot"](c.TRUST_ANCHOR)
    assert calls == ["GetCurrentThread", "OpenThreadToken"]


def test_t3_cli_rejects_untrusted_paths_without_reading_them(authority, tmp_path, monkeypatch):
    import importlib.util
    d, g, pins = authority
    path = REPOSITORY / "scripts/run_future_lineage.py"
    spec = importlib.util.spec_from_file_location("t3_read_order_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    selected = tmp_path / "caller-selected.json"
    selected.write_bytes(b"synthetic untrusted operational input")
    original = Path.open
    def forbidden(self, *args, **kwargs):
        if self == selected:
            pytest.fail("CLI-selected file must not be opened to establish approval")
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Path, "open", forbidden)
    args = ["inspect-implementation", "--decision", selected.relative_to(REPOSITORY).as_posix(),
            "--decision-sha256", pins["decision_pin"], "--implementation-grant", g["path"],
            "--implementation-grant-sha256", pins["grant_pin"],
            "--candidate-sha256", pins["candidate_pin"], "--procedure-sha256", pins["procedure_pin"]]
    assert module.main(args) == 2
    c.TRUST_ANCHOR.unlink()
    assert module.main(args) == 2


def test_t3_reapproved_malformed_source_fails_closed(t3_repository):
    root, authority = t3_repository
    access = verify_fixture(t3_repository)
    body = copy.deepcopy(access.manifest)
    entry = next(e for e in body["entries"] if "scientific_code" in e["roles"])
    path = root / entry["path"]
    path.write_bytes(b"def broken(:\n")
    entry.update(c.identity(root, path))
    updated = root, repin_fixture(root, authority, manifest=body)
    with pytest.raises(ValueError, match="SOURCE_INVALID"):
        verify_fixture(updated)


@pytest.mark.parametrize("failure,reason", [("invalid_decision", "DECISION_REQUIRED"),
                                          ("expired_grant", "PERMISSION_EXPIRED")])
def test_t3_invalid_approval_precedes_manifest_and_member_reads(t3_repository, monkeypatch, failure, reason):
    root, authority = t3_repository
    access = verify_fixture(t3_repository)
    d, g, _ = authority
    if failure == "invalid_decision":
        record = c.read(root / d["path"])
        record["decision_approval"] = False
        (root / d["path"]).write_bytes(c.canonical(record))
    else:
        record = c.read(root / g["path"])
        record["valid_until_utc_exclusive"] = AT
        (root / g["path"]).write_bytes(c.canonical(record))
    updated = root, repin_fixture(root, authority)
    model = next(e for e in access.manifest["entries"] if "model" in e["roles"])
    forbidden = {root / model["path"], root / access.anchor["manifest"]["path"]}
    original = Path.open
    def denied(self, *args, **kwargs):
        if self in forbidden:
            pytest.fail("Invalid approval must reject before manifest/member reads")
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Path, "open", denied)
    with pytest.raises(ValueError, match=reason):
        verify_fixture(updated)


@pytest.mark.parametrize("stage", ["verification", "effect_boundary"])
def test_t3_expiry_during_dependency_validation_cannot_return_success(t3_repository, monkeypatch, stage):
    end = "2026-10-16T06:00:00Z"
    if stage == "verification":
        original = c.dependency_manifest
        def advance(*args, **kwargs):
            result = original(*args, **kwargs)
            monkeypatch.setattr(c, "now", lambda: end)
            return result
        monkeypatch.setattr(c, "dependency_manifest", advance)
        with pytest.raises(ValueError, match="PERMISSION_EXPIRED"):
            verify_fixture(t3_repository)
    else:
        access = verify_fixture(t3_repository)
        original = c._recheck_members
        def advance(*args, **kwargs):
            result = original(*args, **kwargs)
            monkeypatch.setattr(c, "now", lambda: end)
            return result
        monkeypatch.setattr(c, "_recheck_members", advance)
        with pytest.raises(ValueError, match="PERMISSION_EXPIRED"):
            access.check(AT)


@pytest.mark.parametrize("index", [2, 3, 4, 5])
def test_t3_remediation_system_owner_is_rejected_on_each_protected_path(index):
    snapshot = protected_snapshot()
    c._check_anchor_protection(snapshot)
    snapshot["paths"][index]["owner"] = "S-1-5-18"
    with pytest.raises(ValueError, match="OWNER_OR_INHERITANCE"):
        c._check_anchor_protection(snapshot)


@pytest.mark.parametrize("field,value", [
    ("user_sid", "S-1-5-21-1-2-3-1002"), ("user_sid", None),
    ("user_sid", "not-a-sid"), ("user_sid", "S-1-5-"),
    ("expected_user_sid", None), ("expected_user_sid", "not-a-sid"),
    ("application_account", r"OTHER\papol"),
])
def test_t3_remediation_identity_policy_rejects_wrong_or_missing_information(field, value):
    snapshot = protected_snapshot()
    c._check_anchor_protection(snapshot)
    snapshot[field] = value
    with pytest.raises(ValueError, match="APPLICATION_IDENTITY"):
        c._check_anchor_protection(snapshot)
    del snapshot[field]
    with pytest.raises(ValueError, match="APPLICATION_IDENTITY"):
        c._check_anchor_protection(snapshot)


@pytest.mark.parametrize("sid", ["S-1-5-18", "S-1-5-32-544"])
@pytest.mark.parametrize("index", [2, 3, 4, 5])
def test_t3_remediation_inherit_only_does_not_grant_on_protected_object(sid, index):
    snapshot = protected_snapshot()
    grant = next(a for a in snapshot["paths"][index]["aces"] if a["sid"] == sid)
    grant["flags"] = 0x0B  # Object/container inheritance plus INHERIT_ONLY.
    with pytest.raises(ValueError, match="PROVISIONER_GRANT_MISSING"):
        c._check_anchor_protection(snapshot)
    snapshot["paths"][index]["aces"].append({**grant, "flags": 3})
    c._check_anchor_protection(snapshot)  # A separate applicable grant suffices.


@pytest.mark.parametrize("flags", [0, 1, 2, 3, 4, 0x10, 0x13])
def test_t3_remediation_applicable_allow_flags_preserve_grants(flags):
    snapshot = protected_snapshot()
    for item in snapshot["paths"]:
        for ace in item["aces"]:
            ace["flags"] = flags
    c._check_anchor_protection(snapshot)


@pytest.mark.parametrize("field,value", [
    ("flags", None), ("flags", True), ("flags", -1), ("flags", 0x20),
    ("flags", 0x40), ("flags", "0"), ("mask", True), ("mask", -1),
    ("mask", 0x100000000), ("sid", None),
])
def test_t3_remediation_malformed_ace_metadata_rejects(field, value):
    snapshot = protected_snapshot()
    snapshot["paths"][-1]["aces"][0][field] = value
    with pytest.raises(ValueError, match="ACE_INVALID"):
        c._check_anchor_protection(snapshot)
    del snapshot["paths"][-1]["aces"][0][field]
    with pytest.raises(ValueError, match="ACE_INVALID"):
        c._check_anchor_protection(snapshot)


def test_t3_remediation_inherit_only_does_not_relax_untrusted_writer_policy():
    snapshot = protected_snapshot()
    snapshot["paths"][-1]["aces"].append(
        {"sid": "synthetic-reader", "mask": 0x40000000, "flags": 0x0B})
    with pytest.raises(ValueError, match="UNTRUSTED_WRITER"):
        c._check_anchor_protection(snapshot)


@pytest.fixture
def t3_remediation_native(tmp_path, monkeypatch):
    """Complete fake DLL adapter, including real-shaped in-memory SID/ACL data.

    No WinDLL, token, account, ACL or descriptor API is called on Windows. The
    mock enforces query flags/order and lets tests corrupt outputs independently.
    """
    import ctypes as w
    from ctypes import wintypes as t
    import struct

    class User(w.Structure):
        _fields_ = [("sid", w.c_void_p), ("attributes", t.DWORD)]

    class Function:
        def __init__(self, state, name):
            self.state, self.name = state, name

        def __call__(self, *args):
            self.state.calls.append(self.name)
            return self.state.call(self.name, args)

    class DLL:
        def __init__(self, state):
            self.state, self.functions = state, {}

        def __getattr__(self, name):
            return self.functions.setdefault(name, Function(self.state, name))

    class State:
        failure = None
        error = 1008

        def __init__(self):
            self.calls, self.buffers, self.freed, self.closed = [], [], [], []
            self.lookups, self.security_flags, self.descriptors = [], [], []
            self.expected = self.sid_bytes("S-1-5-21-1-2-3-1001")
            self.owner = self.allocate(self.sid_bytes("S-1-5-32-544"))
            self.group = self.allocate(self.sid_bytes("S-1-5-32-545"))
            self.acl = None
            self.ace_offsets = []
            self.validated = set()

        def allocate(self, raw):
            buffer = w.create_string_buffer(raw, len(raw))
            self.buffers.append(buffer)
            return w.addressof(buffer)

        @staticmethod
        def sid_bytes(text):
            parts = [int(p) for p in text.split("-")[1:]]
            return bytes((parts[0], len(parts) - 2)) + parts[1].to_bytes(6, "big") + b"".join(
                p.to_bytes(4, "little") for p in parts[2:])

        @staticmethod
        def address(value):
            return value if isinstance(value, int) else w.cast(value, w.c_void_p).value

        def sid_raw(self, value):
            address = self.address(value)
            count = t.BYTE.from_address(address + 1).value
            return w.string_at(address, 8 + count * 4)

        @staticmethod
        def output(argument, value):
            argument._obj.value = value

        def make_acl(self):
            aces = []
            for text, mask in [("S-1-5-32-544", 0x1F01FF), ("S-1-5-18", 0x1F01FF),
                               ("S-1-5-21-1-2-3-1001", 0x120089)]:
                sid = self.sid_bytes(text)
                flags = 0x0B if self.failure == "inherit_only" and mask == 0x1F01FF else 3
                kind = 1 if self.failure == "ace_type" else 0
                if self.failure == "ace_flags":
                    flags = 0x80
                size = 8 + len(sid)
                if self.failure == "ace_size":
                    size = 8
                elif self.failure == "ace_extent":
                    size = 0xFFFC
                if self.failure == "ace_sid":
                    sid = bytes((2,)) + sid[1:]
                elif self.failure == "ace_sid_extent":
                    sid = bytes((1, 15)) + sid[2:]
                aces.append(struct.pack("<BBHI", kind, flags, size, mask) + sid)
            raw = struct.pack("<BBHHH", 2, 0, 8 + sum(map(len, aces)), len(aces), 0)
            self.ace_offsets = []
            for ace in aces:
                self.ace_offsets.append(len(raw))
                raw += ace
            self.acl = self.allocate(raw)

        def call(self, name, a):
            if name in {"GetCurrentThread", "GetCurrentProcess"}:
                return -2 if name == "GetCurrentThread" else -1
            if name == "OpenThreadToken":
                self.error = 1008
                return 0
            if name in {"OpenProcessToken", "DuplicateToken"}:
                self.output(a[-1], 101 if name == "OpenProcessToken" else 102)
                return 1
            if name == "LookupAccountNameW":
                assert a[0] is None and a[1] == r"FRIDAY\papol"
                self.lookups.append(a[1])
                if a[2] is None:
                    self.output(a[3], 0 if self.failure == "lookup_empty" else
                                69 if self.failure == "lookup_oversize" else len(self.expected))
                    self.output(a[5], 7)
                    self.error = 1332 if self.failure == "lookup_failure" else 122
                    return 0
                if self.failure == "lookup_query":
                    return 0
                raw = bytes((2,)) + self.expected[1:] if self.failure == "lookup_sid" else self.expected
                w.memmove(a[2], raw, len(raw))
                a[4].value = "OTHER" if self.failure == "lookup_domain" else "FRIDAY"
                self.output(a[5], 6)
                self.output(a[6], 2 if self.failure == "lookup_type" else 1)
                if self.failure == "lookup_return_size":
                    self.output(a[3], 100)
                return 1
            if name == "GetTokenInformation":
                if a[1] == 1:
                    if self.failure == "token_query":
                        return 0
                    raw = self.sid_bytes("S-1-5-21-1-2-3-1002") if self.failure == "wrong_user" else self.expected
                    if self.failure == "token_sid":
                        raw = bytes((2,)) + raw[1:]
                    elif self.failure == "token_sid_extent":
                        raw = bytes((1, 15)) + raw[2:]
                    address = w.addressof(a[2]) + w.sizeof(User)
                    w.memmove(address, raw, len(raw))
                    user = User.from_buffer(a[2])
                    user.sid = None if self.failure == "token_missing" else address - w.sizeof(User) - 1 if self.failure == "token_outside" else address
                    length = w.sizeof(User) - 1 if self.failure == "token_short" else w.sizeof(User) + len(raw)
                    self.output(a[4], length)
                elif a[1] == 20:
                    self.output(a[2], 0)
                    self.output(a[4], 4)
                elif a[1] == 3:
                    t.DWORD.from_buffer(a[2]).value = 0
                    self.output(a[4], 4)
                else:
                    pytest.fail("Unexpected token information class")
                return 1
            if name == "IsValidSid":
                address = self.address(a[0])
                if self.failure == "owner_sid" and address == self.owner:
                    return 0
                if self.failure == "group_sid" and address == self.group:
                    return 0
                return int(t.BYTE.from_address(address).value == 1 and
                           t.BYTE.from_address(address + 1).value <= 15)
            if name == "EqualSid":
                return int(self.sid_raw(a[0]) == self.sid_raw(a[1]))
            if name == "ConvertStringSidToSidW":
                self.output(a[1], self.allocate(self.sid_bytes(a[0])))
                return 1
            if name == "ConvertSidToStringSidW":
                raw = self.sid_raw(a[0])
                text = "S-" + str(raw[0]) + "-" + str(int.from_bytes(raw[2:8], "big")) + "".join(
                    "-" + str(int.from_bytes(raw[n:n+4], "little")) for n in range(8, len(raw), 4))
                buffer = w.create_unicode_buffer(text)
                self.buffers.append(buffer)
                self.output(a[1], w.addressof(buffer))
                return 1
            if name == "CheckTokenMembership":
                self.output(a[2], 0)
                return 1
            if name == "GetNamedSecurityInfoW":
                assert a[1] == 1 and a[2] == 7 and all(a[i] is not None for i in (3, 4, 5, 7))
                self.security_flags.append(a[2])
                self.validated.clear()
                self.make_acl()
                descriptor = self.allocate(b"\0" * 64)
                self.descriptors.append(descriptor)
                for index, value in [(3, self.owner), (4, self.group), (5, self.acl), (7, descriptor)]:
                    missing = {3: "owner_missing", 4: "group_missing", 5: "null_dacl", 7: "descriptor_missing"}[index]
                    self.output(a[index], None if self.failure == missing else value)
                return 5 if self.failure == "descriptor_query" else 0
            if name in {"IsValidSecurityDescriptor", "IsValidAcl"}:
                self.validated.add(name)
                return int(self.failure != ("descriptor_invalid" if name == "IsValidSecurityDescriptor" else "acl_invalid"))
            if name in {"GetSecurityDescriptorOwner", "GetSecurityDescriptorGroup"}:
                self.validated.add(name)
                if self.failure == name:
                    return 0
                is_owner = name.endswith("Owner")
                absent = "descriptor_owner_absent" if is_owner else "descriptor_group_absent"
                self.output(a[1], None if self.failure == absent else self.owner if is_owner else self.group)
                self.output(a[2], 0)
                return 1
            if name == "GetSecurityDescriptorDacl":
                self.validated.add(name)
                if self.failure == name:
                    return 0
                self.output(a[1], 0 if self.failure == "descriptor_dacl_absent" else 1)
                self.output(a[2], None if self.failure == "descriptor_dacl_null" else self.acl)
                self.output(a[3], 0)
                return 1
            if name == "GetSecurityDescriptorControl":
                self.output(a[1], 0x1000)
                self.output(a[2], 1)
                return 1
            if name == "GetAce":
                if self.failure == "ace_query":
                    return 0
                self.output(a[2], None if self.failure == "ace_missing" else self.acl + self.ace_offsets[a[1]])
                return 1
            if name == "AccessCheck":
                assert self.validated == {"IsValidSecurityDescriptor", "IsValidAcl",
                    "GetSecurityDescriptorOwner", "GetSecurityDescriptorGroup", "GetSecurityDescriptorDacl"}
                assert a[2] == 0x2000000
                if self.failure == "access_query":
                    return 0
                self.output(a[6], 0x120089)
                self.output(a[7], 0 if self.failure == "access_denied" else 1)
                return 1
            if name == "LocalFree":
                self.freed.append(self.address(a[0]))
                return None
            if name == "CloseHandle":
                self.closed.append(a[0].value)
                return 1
            pytest.fail("Unexpected fake Windows call: " + name)

    state = State()
    monkeypatch.setattr(w, "WinDLL", lambda *args, **kwargs: DLL(state))
    monkeypatch.setattr(w, "get_last_error", lambda: state.error)
    path = tmp_path / "FFPS/Trust/T3/approval-reference.json"
    path.parent.mkdir(parents=True)
    path.write_text("synthetic path metadata only", encoding="utf-8")
    paths = (tmp_path.parent, tmp_path, *reversed(path.parents[:3]), path)
    monkeypatch.setattr(c, "_anchor_paths", lambda supplied: paths)
    return state, path


def test_t3_remediation_complete_mock_adapter_binds_identity_and_validates_descriptor(t3_remediation_native, monkeypatch):
    state, path = t3_remediation_native
    monkeypatch.setenv("USERNAME", "other")
    monkeypatch.setenv("FFPS_T3_APPLICATION_ACCOUNT", r"OTHER\caller")
    snapshot = c._windows_anchor_snapshot(path)
    c._check_anchor_protection(snapshot)
    assert snapshot["application_account"] == r"FRIDAY\papol"
    assert snapshot["user_sid"] == snapshot["expected_user_sid"] == "S-1-5-21-1-2-3-1001"
    assert state.lookups == [r"FRIDAY\papol", r"FRIDAY\papol"]
    assert state.security_flags == [7] * 6 and state.calls.count("AccessCheck") == 6
    assert all(d in state.freed for d in state.descriptors)
    assert state.closed == [102, 101]
    assert all(a["flags"] == 3 for p in snapshot["paths"] if p["protected"] for a in p["aces"])


@pytest.mark.parametrize("failure,reason", [
    ("lookup_failure", "ACCOUNT_RESOLUTION"), ("lookup_empty", "ACCOUNT_RESOLUTION"),
    ("lookup_oversize", "ACCOUNT_RESOLUTION"), ("lookup_query", "SECURITY_QUERY_FAILED"),
    ("lookup_domain", "ACCOUNT_RESOLUTION"), ("lookup_type", "ACCOUNT_RESOLUTION"),
    ("lookup_sid", "SID_INVALID"), ("lookup_return_size", "ACCOUNT_RESOLUTION"),
    ("token_query", "SECURITY_QUERY_FAILED"), ("token_missing", "SID_INVALID"),
    ("token_outside", "SID_INVALID"), ("token_short", "TOKEN_USER_INVALID"),
    ("token_sid", "SID_INVALID"), ("token_sid_extent", "SID_INVALID"),
    ("wrong_user", "APPLICATION_IDENTITY"),
])
def test_t3_remediation_mock_identity_failures_precede_acl_reads(t3_remediation_native, failure, reason):
    state, path = t3_remediation_native
    state.failure = failure
    with pytest.raises(ValueError, match=reason):
        c._windows_anchor_snapshot(path)
    assert "GetNamedSecurityInfoW" not in state.calls and "AccessCheck" not in state.calls
    assert state.closed == [102, 101]


@pytest.mark.parametrize("failure,reason", [
    ("descriptor_query", "SECURITY_QUERY_FAILED"), ("descriptor_missing", "DESCRIPTOR_INVALID"),
    ("descriptor_invalid", "DESCRIPTOR_INVALID"), ("owner_missing", "DESCRIPTOR_INVALID"),
    ("group_missing", "DESCRIPTOR_INVALID"), ("owner_sid", "DESCRIPTOR_INVALID"),
    ("group_sid", "DESCRIPTOR_INVALID"), ("null_dacl", "NULL_DACL"),
    ("acl_invalid", "ACL_INVALID"), ("GetSecurityDescriptorOwner", "SECURITY_QUERY_FAILED"),
    ("GetSecurityDescriptorGroup", "SECURITY_QUERY_FAILED"),
    ("GetSecurityDescriptorDacl", "SECURITY_QUERY_FAILED"),
    ("descriptor_owner_absent", "DESCRIPTOR_INVALID"), ("descriptor_group_absent", "DESCRIPTOR_INVALID"),
    ("descriptor_dacl_absent", "DESCRIPTOR_INVALID"), ("descriptor_dacl_null", "DESCRIPTOR_INVALID"),
])
def test_t3_remediation_mock_descriptor_failures_precede_access_check(t3_remediation_native, failure, reason):
    state, path = t3_remediation_native
    state.failure = failure
    with pytest.raises(ValueError, match=reason):
        c._windows_anchor_snapshot(path)
    assert "AccessCheck" not in state.calls and state.closed == [102, 101]
    if failure != "descriptor_missing":
        assert all(d in state.freed for d in state.descriptors)


@pytest.mark.parametrize("failure,reason", [
    ("ace_query", "SECURITY_QUERY_FAILED"), ("ace_missing", "ACE_INVALID"),
    ("ace_type", "UNSUPPORTED_ACE"), ("ace_flags", "UNSUPPORTED_ACE"),
    ("ace_size", "ACE_INVALID"), ("ace_extent", "ACE_INVALID"),
    ("ace_sid", "SID_INVALID"), ("ace_sid_extent", "SID_INVALID"),
    ("access_query", "SECURITY_QUERY_FAILED"), ("access_denied", "ACCESS_DENIED"),
])
def test_t3_remediation_mock_acl_decode_and_access_failures_cleanup(t3_remediation_native, failure, reason):
    state, path = t3_remediation_native
    state.failure = failure
    with pytest.raises(ValueError, match=reason):
        c._windows_anchor_snapshot(path)
    # The two unprotected ancestors may already have passed AccessCheck.
    assert all(d in state.freed for d in state.descriptors)
    assert state.closed == [102, 101]


def test_t3_remediation_mock_decoder_retains_inherit_only_flags(t3_remediation_native):
    state, path = t3_remediation_native
    state.failure = "inherit_only"
    snapshot = c._windows_anchor_snapshot(path)
    with pytest.raises(ValueError, match="PROVISIONER_GRANT_MISSING"):
        c._check_anchor_protection(snapshot)
    assert all(p["aces"][0]["flags"] == 0x0B for p in snapshot["paths"] if p["protected"])
