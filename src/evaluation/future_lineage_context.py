"""Exact approval bindings and contained future-lineage storage.

No issuer, acquisition, model loading or implicit authorization is provided.
Whole-file pins come from the owner's independently provisioned fixed anchor.
Caller/CLI values are comparison inputs and cannot replace that trust source.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import ast
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat

RUNTIME_ROOT = "prospective_evidence/prospective-20261014-bl1-v1"
TEST_ROOT = "tmp/BL1-future-lineage-implementation-20261015-01"
NEW_CODE = (
    "src/evaluation/future_lineage_qualification.py",
    "src/evaluation/future_lineage_context.py",
    "src/evaluation/future_lineage.py", "scripts/run_future_lineage.py",
)
NEW_TESTS = (
    "tests/unit/test_future_lineage_qualification.py",
    "tests/unit/test_future_lineage_authorization.py",
    "tests/unit/test_future_lineage_collection.py",
)
DENIED = (
    "network", "acquisition", "live_source_qualification",
    "prospective_prediction_generation", "collection_admission", "journal_mutation",
    "production_replay", "settlement", "scheduling", "activation", "promotion",
    "live_operation_grant_issuance",
)
TRUST_ANCHOR = Path(r"C:\ProgramData\FFPS\Trust\T3\approval-reference.json")
APPLICATION_ACCOUNT = r"FRIDAY\papol"
ANCHOR_SCHEMA = "ffps.future_lineage.trusted_approval_reference.v1"
MANIFEST_SCHEMA = "ffps.future_lineage.dependency_manifest.v1"
REQUIRED_ROLES = frozenset(("entry_point", "adapter", "qualifier", "loader",
    "scientific_code", "model", "mapping_schema", "protocol", "gate", "verifier",
    "configuration", "runtime"))
ROLE_SOURCES = {"entry_point": "scripts/run_future_lineage.py",
    "adapter": "src/evaluation/phase11_current_source.py",
    "qualifier": "src/evaluation/future_lineage_qualification.py",
    "loader": "src/evaluation/future_lineage_context.py",
    "gate": "src/evaluation/future_lineage_context.py",
    "verifier": "src/evaluation/future_lineage_context.py",
    "scientific_code": "src/evaluation/phase10_current_data.py"}
_WRITE = 0x10000000 | 0x40000000 | 0xD0156
_PARENT_REPLACE = 0xD0040
_DACL_ACE_FLAGS = 0x1F  # Inheritance/propagation flags supported for allow ACEs.
_INHERIT_ONLY_ACE = 0x08


def _anchor_paths(path):
    paths = (*reversed(path.parents), path)
    for current in paths:
        info = current.lstat()
        require(not current.is_symlink() and not (getattr(info, "st_file_attributes", 0)
                & stat.FILE_ATTRIBUTE_REPARSE_POINT), "FUTURE_ANCHOR_REPARSE_POINT")
    return paths


def _windows_anchor_snapshot(path):
    """Read security metadata only; no provisioning, subprocess or ACL mutation.

    Unsupported ACL forms and query failures reject. Native enforcement/races are
    not qualified by the synthetic tests; host administrators remain trusted.
    """
    require(os.name == "nt", "FUTURE_ANCHOR_WINDOWS_REQUIRED")
    paths = _anchor_paths(path)
    import ctypes as w
    from ctypes import wintypes as t
    adv = w.WinDLL("advapi32", use_last_error=True)
    kernel = w.WinDLL("kernel32", use_last_error=True)
    ptr = w.c_void_p
    declarations = {
        "OpenProcessToken": ([t.HANDLE, t.DWORD, w.POINTER(t.HANDLE)], t.BOOL),
        "OpenThreadToken": ([t.HANDLE, t.DWORD, t.BOOL, w.POINTER(t.HANDLE)], t.BOOL),
        "DuplicateToken": ([t.HANDLE, w.c_int, w.POINTER(t.HANDLE)], t.BOOL),
        "GetTokenInformation": ([t.HANDLE, w.c_int, ptr, t.DWORD, w.POINTER(t.DWORD)], t.BOOL),
        "CheckTokenMembership": ([t.HANDLE, ptr, w.POINTER(t.BOOL)], t.BOOL),
        "ConvertStringSidToSidW": ([t.LPCWSTR, w.POINTER(ptr)], t.BOOL),
        "ConvertSidToStringSidW": ([ptr, w.POINTER(ptr)], t.BOOL),
        "IsValidSid": ([ptr], t.BOOL),
        "EqualSid": ([ptr, ptr], t.BOOL),
        "LookupAccountNameW": ([t.LPCWSTR, t.LPCWSTR, ptr, w.POINTER(t.DWORD),
            t.LPWSTR, w.POINTER(t.DWORD), w.POINTER(w.c_int)], t.BOOL),
        "LookupPrivilegeNameW": ([t.LPCWSTR, ptr, t.LPWSTR, w.POINTER(t.DWORD)], t.BOOL),
        "GetNamedSecurityInfoW": ([t.LPWSTR, w.c_int, t.DWORD, w.POINTER(ptr),
            w.POINTER(ptr), w.POINTER(ptr), ptr, w.POINTER(ptr)], t.DWORD),
        "IsValidSecurityDescriptor": ([ptr], t.BOOL),
        "IsValidAcl": ([ptr], t.BOOL),
        "GetSecurityDescriptorOwner": ([ptr, w.POINTER(ptr), w.POINTER(t.BOOL)], t.BOOL),
        "GetSecurityDescriptorGroup": ([ptr, w.POINTER(ptr), w.POINTER(t.BOOL)], t.BOOL),
        "GetSecurityDescriptorDacl": ([ptr, w.POINTER(t.BOOL), w.POINTER(ptr),
            w.POINTER(t.BOOL)], t.BOOL),
        "GetSecurityDescriptorControl": ([ptr, w.POINTER(w.c_ushort), w.POINTER(t.DWORD)], t.BOOL),
        "GetAce": ([ptr, t.DWORD, w.POINTER(ptr)], t.BOOL),
        "AccessCheck": ([ptr, t.HANDLE, t.DWORD, ptr, ptr, w.POINTER(t.DWORD),
                         w.POINTER(t.DWORD), w.POINTER(t.BOOL)], t.BOOL),
    }
    for name, (args, result) in declarations.items():
        function = getattr(adv, name)
        function.argtypes, function.restype = args, result
    kernel.GetCurrentProcess.restype = t.HANDLE
    kernel.GetCurrentThread.restype = t.HANDLE
    kernel.LocalFree.argtypes, kernel.LocalFree.restype = [ptr], ptr
    kernel.CloseHandle.argtypes, kernel.CloseHandle.restype = [t.HANDLE], t.BOOL

    def ok(value):
        require(bool(value), "FUTURE_ANCHOR_SECURITY_QUERY_FAILED")

    def sid_text(sid):
        require(bool(sid) and adv.IsValidSid(sid), "FUTURE_ANCHOR_SID_INVALID")
        value = ptr()
        ok(adv.ConvertSidToStringSidW(sid, w.byref(value)))
        try:
            require(bool(value.value), "FUTURE_ANCHOR_SID_INVALID")
            return w.wstring_at(value)
        finally:
            kernel.LocalFree(value)

    def bounded_sid(sid, start, end):
        # Check the header and subauthority extent before handing a SID from a
        # variable-sized token/account/ACE buffer to a native SID operation.
        address = sid.value if isinstance(sid, ptr) else sid
        require(address is not None and start <= address <= end - 8,
                "FUTURE_ANCHOR_SID_INVALID")
        count = t.BYTE.from_address(address + 1).value
        require(count <= 15 and address + 8 + 4 * count <= end and adv.IsValidSid(sid),
                "FUTURE_ANCHOR_SID_INVALID")

    class Mapping(w.Structure):
        _fields_ = [(key, t.DWORD) for key in ("read", "write", "execute", "all")]

    class ACL(w.Structure):
        _fields_ = [("revision", t.BYTE), ("reserved", t.BYTE),
                    ("size", t.WORD), ("count", t.WORD), ("reserved2", t.WORD)]

    class Privilege(w.Structure):
        _fields_ = [("low", t.DWORD), ("high", t.LONG), ("attributes", t.DWORD)]

    class TokenUser(w.Structure):
        _fields_ = [("sid", ptr), ("attributes", t.DWORD)]

    token, duplicate, thread_token, admin_sid = t.HANDLE(), t.HANDLE(), t.HANDLE(), ptr()
    try:
        impersonating = adv.OpenThreadToken(kernel.GetCurrentThread(), 0x8, True, w.byref(thread_token))
        require(not impersonating and w.get_last_error() == 1008,
                "FUTURE_ANCHOR_IMPERSONATION_OR_TOKEN_QUERY")
        ok(adv.OpenProcessToken(kernel.GetCurrentProcess(), 0xA, w.byref(token)))
        ok(adv.DuplicateToken(token, 2, w.byref(duplicate)))
        # The expected identity comes from the OS account database for this fixed
        # fully qualified account, never the current token, CLI, environment or
        # application-writable approval/evidence. Machine administrators remain
        # trusted, including their control of the account database.
        sid_size, domain_size, sid_use = t.DWORD(), t.DWORD(), w.c_int()
        resolved = adv.LookupAccountNameW(None, APPLICATION_ACCOUNT, None, w.byref(sid_size),
            None, w.byref(domain_size), w.byref(sid_use))
        require(not resolved and w.get_last_error() == 122 and
                8 <= sid_size.value <= 68 and 1 <= domain_size.value <= 256,
                "FUTURE_ANCHOR_ACCOUNT_RESOLUTION")
        expected_sid = w.create_string_buffer(sid_size.value)
        domain = w.create_unicode_buffer(domain_size.value)
        ok(adv.LookupAccountNameW(None, APPLICATION_ACCOUNT, expected_sid, w.byref(sid_size),
            domain, w.byref(domain_size), w.byref(sid_use)))
        require(8 <= sid_size.value <= len(expected_sid) and
                0 < domain_size.value <= len(domain) and sid_use.value == 1 and
                domain.value.casefold() == APPLICATION_ACCOUNT.split("\\", 1)[0].casefold(),
                "FUTURE_ANCHOR_ACCOUNT_RESOLUTION")
        bounded_sid(w.cast(expected_sid, ptr), w.addressof(expected_sid),
                    w.addressof(expected_sid) + sid_size.value)
        user_data, user_size = w.create_string_buffer(4096), t.DWORD()
        ok(adv.GetTokenInformation(token, 1, user_data, len(user_data), w.byref(user_size)))
        require(w.sizeof(TokenUser) <= user_size.value <= len(user_data),
                "FUTURE_ANCHOR_TOKEN_USER_INVALID")
        user_sid = TokenUser.from_buffer(user_data).sid
        bounded_sid(user_sid, w.addressof(user_data) + w.sizeof(TokenUser),
                    w.addressof(user_data) + user_size.value)
        require(adv.EqualSid(user_sid, expected_sid), "FUTURE_ANCHOR_APPLICATION_IDENTITY")
        elevated, needed = t.DWORD(), t.DWORD()
        ok(adv.GetTokenInformation(token, 20, w.byref(elevated), w.sizeof(elevated), w.byref(needed)))
        ok(adv.ConvertStringSidToSidW("S-1-5-32-544", w.byref(admin_sid)))
        admin = t.BOOL()
        ok(adv.CheckTokenMembership(duplicate, admin_sid, w.byref(admin)))
        data = w.create_string_buffer(16384)
        ok(adv.GetTokenInformation(token, 3, data, len(data), w.byref(needed)))
        count = t.DWORD.from_buffer(data).value
        require(4 + count * w.sizeof(Privilege) <= len(data), "FUTURE_ANCHOR_TOKEN_INVALID")
        reader_privileges = {"SeChangeNotifyPrivilege", "SeShutdownPrivilege",
            "SeTimeZonePrivilege", "SeUndockPrivilege", "SeIncreaseWorkingSetPrivilege"}
        dangerous = []
        for index in range(count):
            item = Privilege.from_buffer(data, 4 + index * w.sizeof(Privilege))
            name, size = w.create_unicode_buffer(256), t.DWORD(256)
            ok(adv.LookupPrivilegeNameW(None, w.byref(item), name, w.byref(size)))
            if name.value not in reader_privileges:  # Disabled privileges can be enabled.
                dangerous.append(name.value)
        result = {"elevated": bool(elevated.value), "administrator": bool(admin.value),
                  "dangerous_privileges": dangerous, "application_account": APPLICATION_ACCOUNT,
                  "user_sid": sid_text(user_sid),
                  "expected_user_sid": sid_text(w.cast(expected_sid, ptr)), "paths": []}
        protected = set(path.parents[:3]) | {path}
        for current in paths:
            info = current.lstat()
            require(not current.is_symlink() and not (getattr(info, "st_file_attributes", 0)
                    & stat.FILE_ATTRIBUTE_REPARSE_POINT), "FUTURE_ANCHOR_REPARSE_POINT")
            owner, group, dacl, descriptor = ptr(), ptr(), ptr(), ptr()
            try:
                require(adv.GetNamedSecurityInfoW(str(current), 1, 7, w.byref(owner), w.byref(group),
                    w.byref(dacl), None, w.byref(descriptor)) == 0,
                    "FUTURE_ANCHOR_SECURITY_QUERY_FAILED")
                require(bool(descriptor.value) and adv.IsValidSecurityDescriptor(descriptor),
                        "FUTURE_ANCHOR_DESCRIPTOR_INVALID")
                require(bool(owner.value) and bool(group.value) and
                        adv.IsValidSid(owner) and adv.IsValidSid(group),
                        "FUTURE_ANCHOR_DESCRIPTOR_INVALID")
                require(bool(dacl.value), "FUTURE_ANCHOR_NULL_DACL")
                require(adv.IsValidAcl(dacl), "FUTURE_ANCHOR_ACL_INVALID")
                # IsValidSecurityDescriptor alone does not require components to
                # exist. Verify the descriptor actually contains the queried
                # owner, group and present/non-null DACL before AccessCheck.
                component, defaulted, present = ptr(), t.BOOL(), t.BOOL()
                for getter, expected in ((adv.GetSecurityDescriptorOwner, owner),
                                         (adv.GetSecurityDescriptorGroup, group)):
                    ok(getter(descriptor, w.byref(component), w.byref(defaulted)))
                    require(component.value == expected.value, "FUTURE_ANCHOR_DESCRIPTOR_INVALID")
                ok(adv.GetSecurityDescriptorDacl(descriptor, w.byref(present),
                    w.byref(component), w.byref(defaulted)))
                require(present.value and component.value == dacl.value,
                        "FUTURE_ANCHOR_DESCRIPTOR_INVALID")
                control, revision = w.c_ushort(), t.DWORD()
                ok(adv.GetSecurityDescriptorControl(descriptor, w.byref(control), w.byref(revision)))
                aces = []
                if current in protected:
                    header = w.cast(dacl, w.POINTER(ACL)).contents
                    require(header.revision in (2, 4) and header.size >= w.sizeof(ACL),
                            "FUTURE_ANCHOR_ACL_INVALID")
                    next_ace, end = dacl.value + w.sizeof(ACL), dacl.value + header.size
                    for index in range(header.count):
                        ace = ptr()
                        ok(adv.GetAce(dacl, index, w.byref(ace)))
                        require(ace.value == next_ace and next_ace + 4 <= end,
                                "FUTURE_ANCHOR_ACE_INVALID")
                        flags = t.BYTE.from_address(ace.value + 1).value
                        size = t.WORD.from_address(ace.value + 2).value
                        require(size >= 16 and size % 4 == 0 and ace.value + size <= end,
                                "FUTURE_ANCHOR_ACE_INVALID")
                        require(w.c_ubyte.from_address(ace.value).value == 0,
                                "FUTURE_ANCHOR_UNSUPPORTED_ACE")
                        require(not flags & ~_DACL_ACE_FLAGS, "FUTURE_ANCHOR_UNSUPPORTED_ACE")
                        bounded_sid(ptr(ace.value + 8), ace.value + 8, ace.value + size)
                        aces.append({"sid": sid_text(ptr(ace.value + 8)),
                            "mask": t.DWORD.from_address(ace.value + 4).value, "flags": flags})
                        next_ace += size
                mapping = Mapping(0x120089, 0x120116, 0x1200A0, 0x1F01FF)
                privileges, length = w.create_string_buffer(16384), t.DWORD(16384)
                granted, allowed = t.DWORD(), t.BOOL()
                ok(adv.AccessCheck(descriptor, duplicate, 0x2000000, w.byref(mapping), privileges,
                    w.byref(length), w.byref(granted), w.byref(allowed)))
                require(bool(allowed.value), "FUTURE_ANCHOR_ACCESS_DENIED")
                result["paths"].append({"path": str(current), "protected": current in protected,
                    "owner": sid_text(owner), "group": sid_text(group),
                    "dacl_protected": bool(control.value & 0x1000),
                    "effective_access": granted.value, "aces": aces})
            finally:
                if descriptor.value:
                    kernel.LocalFree(descriptor)
        return result
    finally:
        if admin_sid.value:
            kernel.LocalFree(admin_sid)
        for handle in (thread_token, duplicate, token):
            if handle.value:
                kernel.CloseHandle(handle)


def _check_anchor_protection(snapshot):
    require(not snapshot["elevated"] and not snapshot["administrator"] and
            not snapshot["dangerous_privileges"], "FUTURE_ANCHOR_PRIVILEGED_TOKEN")
    user, expected = snapshot.get("user_sid"), snapshot.get("expected_user_sid")
    require(snapshot.get("application_account") == APPLICATION_ACCOUNT and
            isinstance(user, str) and isinstance(expected, str) and
            re.fullmatch(r"S-1-[0-9]+(?:-[0-9]+){1,15}", user) and
            re.fullmatch(r"S-1-[0-9]+(?:-[0-9]+){1,15}", expected) and user == expected,
            "FUTURE_ANCHOR_APPLICATION_IDENTITY")
    trusted = {"S-1-5-18", "S-1-5-32-544"}
    require(len(snapshot["paths"]) >= 5 and sum(p["protected"] for p in snapshot["paths"]) == 4,
            "FUTURE_ANCHOR_SECURITY_PATHS")
    for item in snapshot["paths"]:
        require(not item["effective_access"] & (_WRITE if item["protected"] else _PARENT_REPLACE),
                "FUTURE_ANCHOR_APPLICATION_CAN_WRITE")
        if item["protected"]:
            require(item["dacl_protected"] and item["owner"] == "S-1-5-32-544",
                    "FUTURE_ANCHOR_OWNER_OR_INHERITANCE")
            require(all(isinstance(ace, dict) and set(ace) == {"sid", "mask", "flags"} and
                        isinstance(ace["sid"], str) and ace["sid"] and
                        type(ace["mask"]) is int and 0 <= ace["mask"] <= 0xFFFFFFFF and
                        type(ace["flags"]) is int and 0 <= ace["flags"] <= _DACL_ACE_FLAGS
                        for ace in item["aces"]), "FUTURE_ANCHOR_ACE_INVALID")
            require(all(ace["sid"] in trusted or not ace["mask"] & _WRITE
                        for ace in item["aces"]), "FUTURE_ANCHOR_UNTRUSTED_WRITER")
            require(all(any(ace["sid"] == sid and not ace["flags"] & _INHERIT_ONLY_ACE and
                        (ace["mask"] & 0x1F01FF == 0x1F01FF
                        or ace["mask"] & 0x10000000) for ace in item["aces"]) for sid in trusted),
                    "FUTURE_ANCHOR_PROVISIONER_GRANT_MISSING")


def trusted_approval(root):
    """Fixed application-owned input. No caller/CLI/environment-selected source."""
    path = TRUST_ANCHOR
    before = _windows_anchor_snapshot(path)
    _check_anchor_protection(before)
    require(path.is_file() and path.stat().st_size <= 1048576, "FUTURE_ANCHOR_SIZE")
    raw = path.read_bytes()
    anchor = read(path)
    after = _windows_anchor_snapshot(path)
    _check_anchor_protection(after)
    require(before == after and raw == path.read_bytes(), "FUTURE_ANCHOR_CHANGED")
    require(set(anchor) == {"schema_version", "decision", "grant", "manifest"} and
            anchor["schema_version"] == ANCHOR_SCHEMA, "FUTURE_ANCHOR_SCHEMA")
    for key in ("decision", "grant", "manifest"):
        _reference_shape(anchor[key])
    return anchor, hashlib.sha256(raw).hexdigest()


@lru_cache(maxsize=512)
def _source_tree(body):
    try:
        return ast.parse(body)
    except (SyntaxError, ValueError):
        raise ValueError("FUTURE_DEPENDENCY_SOURCE_INVALID") from None


def dependency_requirements(root):
    """Discover local Python closure without imports; disclose unresolved reads.

    This is a conservative source walk plus explicit approved input roles, not a
    runtime closure proof. In particular injected backends/native code are unknown.
    """
    root = Path(root)
    pending = list(dict.fromkeys((*NEW_CODE, *ROLE_SOURCES.values())))
    found, modules = set(), set()
    unresolved = {"native_dynamic_runtime", "injected_backends_and_runtime_values"}

    def add_module(name):
        if name in modules:
            return
        modules.add(name)
        parts = name.split(".")
        candidates = ["/".join(parts) + ".py", "/".join(parts) + "/__init__.py"]
        # Import identifiers are syntax-checked by ast; these are metadata probes
        # only. Every discovered file passes safe_path before any bytes are read.
        matches = [p for p in candidates if (root / p).is_file()]
        require(len(matches) <= 1, "FUTURE_DEPENDENCY_AMBIGUOUS_MODULE")
        if not matches:
            unresolved.add("external_or_unresolved_module:" + name)
            return
        pending.extend(matches)
        for end in range(1, len(parts)):
            init = "/".join(parts[:end]) + "/__init__.py"
            if (root / init).is_file():
                pending.append(init)

    while pending:
        name = pending.pop()
        if name in found:
            continue
        path = safe_path(root, name)
        require(path.is_file(), "FUTURE_DEPENDENCY_SOURCE_MISSING")
        name = path.resolve().relative_to(root.resolve()).as_posix()
        if name in found:
            continue
        found.add(name)
        tree = _source_tree(path.read_bytes())
        package = name.removesuffix(".py").split("/")[:-1]
        if name.endswith("/__init__.py"):
            package = name.split("/")[:-1]
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    add_module(alias.name)
            elif isinstance(node, ast.ImportFrom):
                prefix = ".".join(package[:len(package) - node.level + 1]) if node.level else ""
                module = ".".join(p for p in (prefix, node.module) if p)
                if module:
                    add_module(module)
                    for alias in node.names:
                        child = module + "." + alias.name
                        if alias.name != "*" and any((root / p).is_file() for p in
                            (child.replace(".", "/") + ".py", child.replace(".", "/") + "/__init__.py")):
                            add_module(child)
            elif isinstance(node, ast.Call):
                function = ast.unparse(node.func)
                if function.split(".")[-1] in {"__import__", "import_module"} and node.args:
                    literal = node.args[0]
                    if isinstance(literal, ast.Constant) and isinstance(literal.value, str) and all(
                            part.isidentifier() for part in literal.value.split(".")):
                        add_module(literal.value)
                if function.split(".")[-1] in {"__import__", "import_module", "find_spec",
                    "spec_from_file_location", "load", "loads", "loadtxt", "read_csv", "read_parquet",
                    "read_text", "read_bytes", "open", "getenv"}:
                    unresolved.add(f"runtime_resolution:{name}:{node.lineno}:{function}")
    return sorted(found), sorted(unresolved)


def dependency_manifest(root, ref):
    path = reference(root, ref)
    body = read(path)
    require(path.read_bytes() == canonical(body), "FUTURE_DEPENDENCY_NONCANONICAL_BYTES")
    require(isinstance(body, dict) and set(body) == {"schema_version", "execution_mode",
        "runtime_closure", "entries", "unresolved"} and body["schema_version"] == MANIFEST_SCHEMA
        and body["execution_mode"] == "synthetic" and body["runtime_closure"] == "UNQUALIFIED",
        "FUTURE_DEPENDENCY_MANIFEST_SCHEMA")
    require(isinstance(body["entries"], list) and isinstance(body["unresolved"], list),
            "FUTURE_DEPENDENCY_MANIFEST_SCHEMA")
    paths, roles, names = set(), set(), set()
    for entry in body["entries"]:
        require(isinstance(entry, dict) and set(entry) == {"roles", "path", "sha256", "size_bytes"}
                and isinstance(entry["roles"], list) and entry["roles"], "FUTURE_DEPENDENCY_ENTRY_SCHEMA")
        local_roles = entry["roles"]
        require(all(isinstance(r, str) and re.fullmatch(r"[a-z][a-z0-9_.:-]{0,191}", r)
                    for r in local_roles) and len(set(local_roles)) == len(local_roles)
                and not roles.intersection(local_roles), "FUTURE_DEPENDENCY_DUPLICATE_ROLE")
        roles.update(local_roles)
        require(all(r not in ROLE_SOURCES or entry["path"] == ROLE_SOURCES[r]
                    for r in local_roles), "FUTURE_DEPENDENCY_ROLE_PATH")
        item = {k: entry[k] for k in ("path", "sha256", "size_bytes")}
        require(type(item["size_bytes"]) is int and item["size_bytes"] >= 0 and
                isinstance(item["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", item["sha256"]),
                "FUTURE_DEPENDENCY_ENTRY_SCHEMA")
        _reference_shape(item)
        path = Path(root).absolute().joinpath(*_relative_parts(item["path"]))
        normalized = str(path.resolve()).casefold()
        require(normalized not in paths, "FUTURE_DEPENDENCY_DUPLICATE_PATH")
        paths.add(normalized)
        names.add(item["path"])
    # Reject altered declared bytes before deriving anything from source syntax.
    _recheck_members(root, body["entries"])
    required, unresolved = dependency_requirements(root)
    require(REQUIRED_ROLES <= roles and set(required) <= names, "FUTURE_DEPENDENCY_MEMBER_MISSING")
    require(all(isinstance(x, str) for x in body["unresolved"]) and
            len(set(body["unresolved"])) == len(body["unresolved"]) and
            set(unresolved) <= set(body["unresolved"]), "FUTURE_DEPENDENCY_UNKNOWN_HIDDEN")
    return body


def _recheck_members(root, entries, *, capture=()):
    """Fresh byte checks at every boundary, sharing ancestor checks only locally.

    Compare open descriptors and final path metadata; recheck the ancestor set
    before returning. This is not native race/loaded-memory qualification.
    """
    checked, captured = {}, {}

    def stamp(info):
        return (info.st_dev, info.st_ino, info.st_mode,
                (info.st_size, info.st_mtime_ns) if stat.S_ISREG(info.st_mode) else None)

    def guard(path, relative):
        if path in checked:
            return checked[path]
        try:
            info = path.lstat()
        except FileNotFoundError:
            raise ValueError("FUTURE_REFERENCE_MISSING:" + relative) from None
        require(not stat.S_ISLNK(info.st_mode) and not (getattr(info, "st_file_attributes", 0)
                & stat.FILE_ATTRIBUTE_REPARSE_POINT), "FUTURE_REPARSE_POINT")
        checked[path] = stamp(info)
        return checked[path]

    for item in entries:
        _reference_shape({k: item[k] for k in ("path", "sha256", "size_bytes")})
        path = Path(root).absolute().joinpath(*_relative_parts(item["path"]))
        for ancestor in reversed(path.parents):
            guard(ancestor, item["path"])
        initial = guard(path, item["path"])
        require(initial[-1] is not None, "FUTURE_REFERENCE_MISSING:" + item["path"])
        with path.open("rb") as stream:
            opened = os.fstat(stream.fileno())
            require(stamp(opened) == initial and opened.st_size == item["size_bytes"],
                    "FUTURE_REFERENCE_DRIFT:" + item["path"])
            h, size, blocks = hashlib.sha256(), 0, []
            while True:
                block = stream.read(min(1024 * 1024, item["size_bytes"] - size + 1))
                if not block:
                    break
                size += len(block)
                require(size <= item["size_bytes"], "FUTURE_REFERENCE_DRIFT:" + item["path"])
                h.update(block)
                if item["path"] in capture:
                    blocks.append(block)
            require(stamp(os.fstat(stream.fileno())) == initial and
                    size == item["size_bytes"] and h.hexdigest() == item["sha256"],
                    "FUTURE_REFERENCE_DRIFT:" + item["path"])
            if item["path"] in capture:
                captured[item["path"]] = b"".join(blocks)
    for path, initial in checked.items():
        try:
            info = path.lstat()
        except FileNotFoundError:
            raise ValueError("FUTURE_REFERENCE_MISSING:" + str(path)) from None
        require(not (getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT)
                and stamp(info) == initial, "FUTURE_DEPENDENCY_PATH_CHANGED")
    return captured


def require(condition, code):
    if not condition:
        raise ValueError(code)


def instant(value):
    require(isinstance(value, str), "FUTURE_TIME_INVALID")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        raise ValueError("FUTURE_TIME_INVALID") from None
    require(result.tzinfo is not None, "FUTURE_TIME_ZONE_REQUIRED")
    return result.astimezone(timezone.utc)


def now():
    return datetime.now(timezone.utc).isoformat()


def _permission_time(grant, at):
    require(instant(grant["valid_from_utc"]) <= instant(at) <
            instant(grant["valid_until_utc_exclusive"]), "FUTURE_PERMISSION_EXPIRED")
    require(instant(at) >= instant(grant["human_adoption_event_time_utc"]),
            "FUTURE_PRE_ISSUANCE_OPERATION")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def fingerprint(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def seal(value, key="record_sha256"):
    body = {k: v for k, v in value.items() if k != key}
    return {**body, key: fingerprint(body)}


def sealed(value, key="record_sha256"):
    require(isinstance(value, dict) and value.get(key) == seal(value, key)[key],
            "FUTURE_SEAL_INVALID:" + key)


def _decode_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "FUTURE_DUPLICATE_JSON_KEY")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError("FUTURE_NONFINITE_JSON:" + value)

    return json.loads(text,
                      object_pairs_hook=pairs, parse_constant=invalid)


def read(path):
    return _decode_json(Path(path).read_text(encoding="utf-8"))


def _relative_parts(relative):
    require(isinstance(relative, str) and relative and "\\" not in relative
            and ":" not in relative, "FUTURE_PATH_INVALID")
    parts = PurePosixPath(relative).parts
    require(not relative.startswith("/") and all(
        p and p not in (".", "..") and p == p.rstrip(" .") and
        not re.fullmatch(r"(?i:con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?", p)
        for p in relative.split("/")), "FUTURE_PATH_INVALID")
    require(parts, "FUTURE_PATH_INVALID")
    return parts


def _reference_shape(ref):
    require(isinstance(ref, dict) and set(ref) == {"path", "sha256", "size_bytes"} and
            type(ref["size_bytes"]) is int and ref["size_bytes"] >= 0 and
            isinstance(ref["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", ref["sha256"]),
            "FUTURE_REFERENCE_SCHEMA")
    _relative_parts(ref["path"])


def safe_path(root, relative):
    """Reject traversal, device/ADS names and *all* existing reparse ancestors."""
    parts = _relative_parts(relative)
    root = Path(root).absolute()
    path = root.joinpath(*parts)
    for item in (*reversed(root.parents), root,
                 *[root.joinpath(*parts[:n]) for n in range(1, len(parts) + 1)]):
        if item.exists() or item.is_symlink():
            info = item.lstat()
            require(not item.is_symlink() and not
                    (getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT),
                    "FUTURE_REPARSE_POINT")
    require(path.resolve().is_relative_to(root.resolve()) and path != root,
            "FUTURE_PATH_ESCAPE")
    return path


def identity(root, path):
    path = Path(path)
    relative = path.absolute().relative_to(Path(root).absolute()).as_posix()
    path = safe_path(root, relative)
    require(path.is_file(), "FUTURE_REFERENCE_MISSING:" + relative)
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return {"path": relative, "sha256": h.hexdigest(), "size_bytes": path.stat().st_size}


def reference(root, ref):
    require(isinstance(ref, dict) and set(ref) == {"path", "sha256", "size_bytes"},
            "FUTURE_REFERENCE_SCHEMA")
    path = safe_path(root, ref["path"])
    require(identity(root, path) == ref, "FUTURE_REFERENCE_DRIFT:" + ref["path"])
    return path


def pinned(root, ref, expected_hash, *, json_required=True):
    require(isinstance(expected_hash, str) and re.fullmatch(r"[0-9a-f]{64}", expected_hash)
            and ref.get("sha256") == expected_hash, "FUTURE_EXTERNAL_PIN_REQUIRED")
    path = reference(root, ref)
    return read(path) if json_required else path


@dataclass(frozen=True)
class ImplementationAccess:
    repository: Path
    decision: dict
    grant: dict
    decision_ref: dict
    grant_ref: dict
    anchor: dict
    anchor_sha256: str
    manifest: dict
    member_refs: tuple

    def check_dependencies(self):
        current, digest = trusted_approval(self.repository)
        require(current == self.anchor and digest == self.anchor_sha256,
                "FUTURE_TRUSTED_APPROVAL_DRIFT")
        require(current["decision"] == self.decision_ref and current["grant"] == self.grant_ref,
                "FUTURE_TRUSTED_APPROVAL_MISMATCH")
        refs = [current[key] for key in ("decision", "grant", "manifest")]
        members = [dict(zip(("path", "sha256", "size_bytes"), row)) for row in self.member_refs]
        raw = _recheck_members(self.repository, refs + members,
                               capture={r["path"] for r in refs})
        documents = {key: _decode_json(raw[current[key]["path"]].decode("utf-8"))
                     for key in ("decision", "grant", "manifest")}
        require(documents["manifest"] == self.manifest,
                "FUTURE_DEPENDENCY_MANIFEST_DRIFT")
        require(documents["decision"] == self.decision and documents["grant"] == self.grant,
                "FUTURE_APPROVAL_DRIFT")

    def check(self, at):
        _permission_time(self.grant, at)
        _permission_time(self.grant, now())
        self.check_dependencies()
        # Records and all dependency bytes were just re-read and checked together.
        _permission_time(self.grant, at)
        _permission_time(self.grant, now())

    def test_path(self, relative):
        self.check(now())
        require(relative.startswith(TEST_ROOT + "/"), "FUTURE_TEST_SCOPE")
        return safe_path(self.repository, relative)


def verify_implementation(root, decision_ref, grant_ref, *, decision_pin, grant_pin,
                          candidate_pin, procedure_pin, at):
    root = Path(root).resolve()
    anchor, anchor_sha256 = trusted_approval(root)
    require(anchor["decision"] == decision_ref and anchor["grant"] == grant_ref,
            "FUTURE_TRUSTED_APPROVAL_MISMATCH")
    decision = pinned(root, decision_ref, decision_pin)
    grant = pinned(root, grant_ref, grant_pin)
    require(decision.get("dependency_manifest") == grant.get("dependency_manifest") ==
            anchor["manifest"], "FUTURE_APPROVAL_MANIFEST_BINDING")
    require(decision.get("candidate", {}).get("sha256") == candidate_pin and
            decision.get("procedure", {}).get("sha256") == procedure_pin,
            "FUTURE_EXTERNAL_PIN_REQUIRED")
    require(decision.get("schema_version") == "ffps.future_lineage.decision_acceptance.v1"
            and decision.get("decision_approval") is True and
            decision.get("exact_candidate_S1_S7_accepted") is True, "FUTURE_DECISION_REQUIRED")
    require(grant.get("schema_version") == "ffps.future_lineage.implementation_permission.v1"
            and grant.get("implementation_permission") is True
            and grant.get("decision_acceptance") == decision_ref,
            "FUTURE_IMPLEMENTATION_GRANT_REQUIRED")
    candidate = pinned(root, decision["candidate"], candidate_pin)
    pinned(root, decision["procedure"], procedure_pin, json_required=False)
    require(grant["candidate"] == decision["candidate"] and
            grant["procedure"] == decision["procedure"] and
            decision["accepted_decisions"] == candidate["decisions"] and
            set(candidate["decisions"]) == {f"S{i}" for i in range(1, 8)},
            "FUTURE_DECISION_BINDING")
    require(grant["new_source_paths"] == list(NEW_CODE) and
            grant["new_test_paths"] == list(NEW_TESTS) and
            grant["test_output_directory"] == TEST_ROOT and
            grant["runtime_root_must_remain_uncreated"] == RUNTIME_ROOT and
            set(grant["withheld_operations"]) == set(DENIED) and
            grant["existing_source_test_modifications_permitted"] is False,
            "FUTURE_IMPLEMENTATION_SCOPE")
    require(grant["bound_artifacts"] == decision["bound_artifacts"], "FUTURE_GRANT_DESIGN_BINDINGS")
    for row in (decision, grant):
        require(row["preparation_permission"] is False and
                row["collection_authorization"] is False, "FUTURE_PERMISSION_CONFLATION")
        require(row["trust"]["declared_owner"] == candidate["owner"]["proposed_label"] and
                row["trust"]["declared_issuer"] == candidate["issuer"]["proposed_label"],
                "FUTURE_DECLARED_ROLE_REQUIRED")
        require(row["trust"]["independent_identity_verified"] is False and
                row["trust"]["cryptographic_signature_or_non_repudiation_claimed"] is False and
                row["trust"]["phase20_authority_reused"] is False, "FUTURE_TRUST_MODEL")
    expected = candidate["bound_artifacts"] + candidate["reviewed_design_documents"]
    require(len(expected) == 18 and [{k: r[k] for k in ("path", "sha256", "size_bytes")}
            for r in decision["bound_artifacts"]] == expected, "FUTURE_DESIGN_BINDINGS")
    for ref in expected + [candidate["preserved_phase20_boundary"]]:
        reference(root, ref)
    evidence = decision["human_evidence"]
    require(grant["human_evidence"] == evidence, "FUTURE_HUMAN_EVIDENCE_BINDING")
    event = read(reference(root, evidence["retained_event"]))
    require(event["type"] == "response_item" and event["payload"]["role"] == "user"
            and event["payload"]["id"] == evidence["original_message_id"],
            "FUTURE_ORIGINAL_HUMAN_MESSAGE_REQUIRED")
    text = "".join(c["text"] for c in event["payload"]["content"]
                   if c.get("type") == "input_text")
    require(reference(root, evidence["verbatim_user_text"]).read_bytes() == text.encode("utf-8")
            and text.startswith("i personally adopt and issue the completed statement."),
            "FUTURE_HUMAN_ADOPTION_MISMATCH")
    brief = reference(root, evidence["original_uploaded_brief"]["retained_bytes"]).read_text(
        encoding="utf-8-sig")
    require(candidate_pin in brief and procedure_pin in brief, "FUTURE_HUMAN_PIN_MISSING")
    reference(root, evidence["adopted_statement"])
    # Only a semantically valid, currently active approval can drive dependency
    # member reads. Recheck actual time again after potentially slow validation.
    _permission_time(grant, at)
    _permission_time(grant, now())
    manifest = dependency_manifest(root, anchor["manifest"])
    access = ImplementationAccess(root, decision, grant, decision_ref, grant_ref,
                                  anchor, anchor_sha256, manifest,
                                  tuple(tuple(e[k] for k in ("path", "sha256", "size_bytes"))
                                        for e in manifest["entries"]))
    access.check(at)
    require(instant(grant["valid_from_utc"]) <
            instant(grant["valid_until_utc_exclusive"]), "FUTURE_TIME_WINDOW_INVALID")
    return access


CONTROL = ("lineage_specification", "prospective_protocol", "dependency_manifest",
           "protected_before", "preparation_permission", "qualification_acceptance",
           "collection_authorization")
SLOT = re.compile(
    r"(?:(" + "|".join(CONTROL) + r")\.json|"
    r"BL1/qualification/attempt-[0-9]{6}/(?:"
    r"(?:run_specification|source_index|fotmob_index)\.json|raw_sources/"
    r"(?:primary-(?:2025|2026)-(?:matches|teams)|statistics-(?:2025|2026)|"
    r"fotmob-league|fotmob-match-[0-9]+)\.(?:json|raw)|"
    r"assessments/assessment-[0-9]{3}/(?:qualification|qualified_source_index|"
    r"current_source_contract|season_boundary_evidence|team_mapping_evidence|"
    r"provider_overlap_evidence|feature_continuity_evidence|verification)\.json)|"
    r"BL1/acquisition/attempt-[0-9]{6}/matches\.(?:json|raw)|"
    r"BL1/journal/[0-9]{8}\.json|BL1/verification/[0-9a-f]{64}\.json|"
    r"BL1/attempt_outcomes/[A-Za-z0-9_-]{1,96}\.json|BL1/operation\.lock)"
)


@dataclass(frozen=True)
class EvidenceStore:
    repository: Path
    relative_root: str
    mode: str
    access: ImplementationAccess | None = None

    def __post_init__(self):
        require(self.mode in ("synthetic", "live"), "FUTURE_EXECUTION_MODE")
        if self.mode == "live":
            require(self.relative_root == RUNTIME_ROOT, "FUTURE_RUNTIME_ROOT")
        else:
            require(re.fullmatch(re.escape(TEST_ROOT) + r"/fixtures/[A-Za-z0-9_-]+",
                                 self.relative_root) is not None, "FUTURE_SYNTHETIC_ROOT")
        safe_path(self.repository, self.relative_root)

    @property
    def root(self):
        return safe_path(self.repository, self.relative_root)

    def path(self, slot):
        require(SLOT.fullmatch(slot) is not None, "FUTURE_UNDECLARED_SLOT")
        return safe_path(self.repository, self.relative_root + "/" + slot)

    def create(self, *, at=None):
        require(self.access is not None and self.access.repository.resolve() == self.repository.resolve(),
                "FUTURE_IMPLEMENTATION_ACCESS_REQUIRED")
        self.access.check(at or now())
        self.access.check(now())
        require(not self.root.exists(), "FUTURE_ROOT_COLLISION")
        # Runtime creation requires a later preparation implementation/review.
        require(self.mode == "synthetic", "FUTURE_LIVE_PREPARATION_NOT_IMPLEMENTED")
        self.root.mkdir(parents=True, exist_ok=False)

    def publish(self, slot, value, *, at=None):
        require(self.access is not None, "FUTURE_IMPLEMENTATION_ACCESS_REQUIRED")
        self.access.check(at or now())
        self.access.check(now())
        require(self.mode == "synthetic", "FUTURE_LIVE_OPERATION_NOT_IMPLEMENTED")
        require(value.get("execution_mode") == "synthetic", "FUTURE_SYNTHETIC_LABEL_REQUIRED")
        path = self.path(slot)
        require(not path.exists(), "FUTURE_ARTIFACT_COLLISION")
        # The unchanged repository primitive fsyncs, hard-links exclusively and
        # checks readback. Never replace existing scientific evidence.
        from src.ml.bundle_activation import persist_new
        persist_new(path, value)
        return identity(self.repository, path)


def deny_live(operation):
    require(operation not in DENIED, "FUTURE_LIVE_PERMISSION_WITHHELD:" + operation)
