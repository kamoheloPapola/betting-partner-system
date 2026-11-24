"""Snapshot the canonical database and verify a retained copy on the Drive mount."""

import argparse
from contextlib import closing
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys
from tempfile import TemporaryDirectory
from urllib.error import HTTPError
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# The backup job relies on the frozen module's private transport name and return contract.
from src.canonical.heartbeat import TIMEOUT, _https_get

DATABASE = ROOT / "data/canonical/football.sqlite3"
DESTINATION = Path("G:/My Drive/evidence-backups/betting_partner_system/canonical-snapshots")
STAGING = ROOT / ".tmp/canonical-backup-staging"
LOG = ROOT / "logs/canonical-backup.jsonl"
HEARTBEAT_ENV = "CANONICAL_BACKUP_HEARTBEAT_URL"
SNAPSHOT_TIMESTAMP = "%Y%m%dT%H%M%S%fZ"
SNAPSHOT_NAME = re.compile(r"canonical-db-(?P<timestamp>[0-9]{8}T[0-9]{12}Z)\.sqlite3")
DAILY_KEEP = 30
MONTHLY_KEEP = 12
BACKUP_TIMEOUT = 35


def heartbeat_endpoint():
    value = os.environ.get(HEARTBEAT_ENV)
    if value is None:
        return None
    if not re.fullmatch(r"https://hc-ping\.com/[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", value):
        raise ValueError("Invalid backup heartbeat URL")
    return urlsplit(value)


def notify(endpoint, step, transport):
    if endpoint is None:
        return "DISABLED", None
    body = b""
    if step is not None:
        endpoint = endpoint._replace(path=endpoint.path + "/fail")
        body = json.dumps(step).encode("utf-8")
    for _ in range(2):
        try:
            reply = transport(endpoint, body, TIMEOUT)
        except Exception as exc:
            reply = type(exc)
        if type(reply) is int:
            if 200 <= reply < 300:
                return "SENT", None
            error_type = HTTPError
            retry = reply == 429 or 500 <= reply < 600
        elif isinstance(reply, type) and issubclass(reply, BaseException):
            error_type = reply
            retry = issubclass(reply, OSError)
        else:
            error_type = TimeoutError if reply is None else TypeError
            retry = reply is None
        if not retry:
            break
    return "FAILED", error_type.__name__


def take_snapshot(database, directory):
    # A failed CLI attempt may leave a partial file. A retry always uses a fresh name.
    for attempt in (1, 2):
        target = directory / f"snapshot-{attempt}.sqlite3"
        try:
            proc = subprocess.run(
                [sys.executable, "-B", str(ROOT / "scripts/run_canonical.py"),
                 "--database", str(database), "backup", str(target)],
                cwd=ROOT, capture_output=True, text=True, timeout=BACKUP_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            if attempt == 2:
                raise
            continue
        if proc.returncode == 0:
            return target
        response = json.loads(proc.stdout)
        # The canonical CLI masks SQLite OperationalError messages.
        error_code = response.get("error_code")
        if error_code == "OPERATOR_OPERATIONALERROR":
            failure = sqlite3.OperationalError
        elif error_code == "BACKUP_TIMEOUT_PARTIAL_RETAINED":
            failure = TimeoutError
        else:
            raise subprocess.CalledProcessError(proc.returncode, proc.args)
        if attempt == 2:
            raise failure("Snapshot retry exhausted")


def inspect_snapshot(path):
    # The completed backup is standalone and must not create WAL sidecars on Drive.
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)) as db:
        if db.execute("PRAGMA integrity_check").fetchmany(2) != [("ok",)]:
            raise ValueError("Invalid SQLite snapshot")
        digest = sha256()
        for line in db.iterdump():
            digest.update(line.encode("utf-8") + b"\n")
        return digest.hexdigest()


def verified_sidecar(path):
    if not SNAPSHOT_NAME.fullmatch(path.name):
        raise ValueError("Unexpected snapshot name")
    sidecar = path.with_suffix(".sqlite3.json")
    with sidecar.open(encoding="utf-8") as stream:
        record = json.loads(stream.read(8193))
    digest = record.get("logical_sha256")
    if (record.get("snapshot") != path.name or not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or record.get("copy_logical_sha256") != digest
            or record.get("local_integrity") != "ok" or record.get("copy_integrity") != "ok"):
        raise ValueError("Invalid backup sidecar")
    return digest


def snapshot_timestamp(path):
    match = SNAPSHOT_NAME.fullmatch(path.name)
    if match is None:
        raise ValueError("Unexpected snapshot name")
    return datetime.strptime(match["timestamp"], SNAPSHOT_TIMESTAMP)


def prune_completed(destination, current):
    created = snapshot_timestamp(current)
    current_month = created.year * 12 + created.month
    completed = []
    for sidecar in destination.glob("canonical-db-*.sqlite3.json"):
        snapshot = sidecar.with_suffix("")
        if (snapshot == current or not snapshot.is_file()
                or snapshot.is_symlink() or sidecar.is_symlink() or not sidecar.is_file()):
            continue
        try:
            timestamp = snapshot_timestamp(snapshot)
            verified_sidecar(snapshot)
        except (ValueError, AttributeError, FileNotFoundError):
            continue
        completed.append((timestamp, snapshot, sidecar))

    completed.sort(reverse=True)
    # Reserve one slot for the verified snapshot from this run, even after a clock rollback.
    keep = {snapshot for _, snapshot, _ in completed[:DAILY_KEEP - 1]}
    monthly_keepers = {}
    for timestamp, snapshot, _ in completed:
        month = timestamp.year * 12 + timestamp.month
        if current_month - MONTHLY_KEEP <= month < current_month:
            monthly_keepers.setdefault(month, snapshot)
    keep.update(monthly_keepers.values())

    excess = [(snapshot, sidecar) for _, snapshot, sidecar in completed if snapshot not in keep]
    for snapshot, sidecar in excess:
        sidecar.unlink(missing_ok=True)
        snapshot.unlink(missing_ok=True)
    return len(excess)


def same_location(left, right):
    return os.path.normcase(os.path.abspath(left)) == os.path.normcase(os.path.abspath(right))


def production_paths(database, destination):
    return same_location(database, DATABASE) and same_location(destination, DESTINATION)


def run_backup(database, destination, staging, *, require_heartbeat=False, transport=_https_get, now=None):
    endpoint = None
    snapshot_name = None
    step = "heartbeat_configuration"
    try:
        if production_paths(database, destination):
            endpoint = heartbeat_endpoint()
        if require_heartbeat and endpoint is None:
            raise ValueError("Backup heartbeat is required")
        step = "destination"
        if not destination.is_dir():
            raise FileNotFoundError("Drive destination is missing")
        step = "staging"
        staging.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix="canonical-backup-", dir=staging) as temporary:
            step = "snapshot"
            if not database.is_file():
                raise FileNotFoundError("Canonical database is missing")
            source = take_snapshot(database, Path(temporary))
            step = "verify_local"
            logical = inspect_snapshot(source)
            created = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
            target = destination / f"canonical-db-{created:{SNAPSHOT_TIMESTAMP}}.sqlite3"
            snapshot_name = target.name
            sidecar = target.with_suffix(".sqlite3.json")
            step = "copy"
            if sidecar.exists():
                raise FileExistsError("Snapshot sidecar already exists")
            with source.open("rb") as reader, target.open("xb") as writer:
                shutil.copyfileobj(reader, writer)
            step = "verify_copy"
            if inspect_snapshot(target) != logical:
                raise ValueError("Copied snapshot hash differs")
            step = "sidecar"
            record = {
                "snapshot": target.name, "created_utc": created.isoformat(),
                "logical_sha256": logical, "local_integrity": "ok",
                "copy_integrity": "ok", "copy_logical_sha256": logical,
            }
            with sidecar.open("x", encoding="utf-8", newline="\n") as writer:
                json.dump(record, writer, indent=2)
                writer.write("\n")
            step = "staging"
        step = "prune"
        pruned = prune_completed(destination, target)
    except Exception as exc:
        heartbeat, _ = notify(endpoint, step, transport)
        return {"state": "FAILED", "step": step, "error_type": type(exc).__name__,
                "snapshot": snapshot_name, "heartbeat": heartbeat}
    heartbeat, error_type = notify(endpoint, None, transport)
    if heartbeat == "FAILED":
        return {"state": "FAILED", "step": "heartbeat", "error_type": error_type,
                "heartbeat": heartbeat, **record, "pruned": pruned}
    return {"state": "BACKUP_VERIFIED", "heartbeat": heartbeat, **record, "pruned": pruned}


def restore_check(download, scratch):
    if download.resolve() == DATABASE.resolve() or scratch.resolve() == DATABASE.resolve():
        raise ValueError("Restore checks require scratch paths")
    expected = verified_sidecar(download)
    if inspect_snapshot(download) != expected:
        raise ValueError("Downloaded snapshot hash differs")
    with scratch.open("xb"):
        pass
    try:
        with closing(sqlite3.connect(download.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)) as source:
            with closing(sqlite3.connect(scratch)) as restored:
                source.backup(restored)
        if inspect_snapshot(scratch) != expected:
            raise ValueError("Restored snapshot hash differs")
    except Exception:
        scratch.unlink(missing_ok=True)
        raise
    return {"state": "RESTORE_VERIFIED", "snapshot": download.name, "logical_sha256": expected}


REGISTER = r"""
$ErrorActionPreference = 'Stop'
$name = 'BettingPartnerCanonicalBackup'
if ([Environment]::GetEnvironmentVariable('CANONICAL_BACKUP_HEARTBEAT_URL', 'User') -cne $env:CANONICAL_BACKUP_HEARTBEAT_URL) { throw 'The backup heartbeat URL must be set for the Windows user.' }
$before = @(Get-ScheduledTask | Select-Object TaskName,TaskPath,State)
$forecast = @($before | Where-Object { $_.TaskName -eq 'BettingPartnerCanonical' -and $_.TaskPath -eq '\' })
if ($forecast.Count -ne 1 -or $forecast[0].State -eq 'Disabled') { throw 'Forecast task is unavailable.' }
if (@($before | Where-Object TaskName -eq $name).Count) { throw 'Backup task already exists.' }
$base = Get-ScheduledTask -TaskName 'BettingPartnerCanonical' -TaskPath '\'
$daily = @($base.Triggers)
if ($daily.Count -ne 1 -or $daily[0].CimClass.CimClassName -ne 'MSFT_TaskDailyTrigger' -or
    $daily[0].DaysInterval -ne 1 -or $daily[0].StartBoundary -notmatch 'T18:00:00') { throw 'Forecast trigger is not daily at 18:00.' }
$principal = New-ScheduledTaskPrincipal -UserId ([Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
$action = New-ScheduledTaskAction -Execute $env:CANONICAL_BACKUP_TASK_PYTHON -Argument ('-B "' + $env:CANONICAL_BACKUP_TASK_SCRIPT + '" run --require-heartbeat') -WorkingDirectory $env:CANONICAL_BACKUP_TASK_ROOT
$trigger = New-ScheduledTaskTrigger -Daily -At ([datetime]::Today.AddHours(18).AddMinutes(30))
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 15) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName $name -TaskPath '\' -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Description 'Verified canonical SQLite snapshots and retention on the Drive mount.' | Out-Null
$after = @(Get-ScheduledTask | Select-Object TaskName,TaskPath,State)
$oldAfter = @($after | Where-Object TaskName -ne $name)
$delta = Compare-Object ($before | Sort-Object TaskPath,TaskName | ConvertTo-Json -Compress) ($oldAfter | Sort-Object TaskPath,TaskName | ConvertTo-Json -Compress)
if ($delta) { throw 'An existing task changed during registration.' }
"""


def register():
    if heartbeat_endpoint() is None or not DATABASE.is_file() or not DESTINATION.is_dir():
        raise ValueError("Missing database, backup destination, or backup heartbeat configuration")
    environment = {
        **os.environ,
        "CANONICAL_BACKUP_TASK_PYTHON": sys.executable,
        "CANONICAL_BACKUP_TASK_SCRIPT": str(Path(__file__).resolve()),
        "CANONICAL_BACKUP_TASK_ROOT": str(ROOT),
    }
    subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", REGISTER],
        cwd=ROOT, env=environment, capture_output=True, check=True, timeout=30,
    )
    return {"state": "REGISTERED", "task": "BettingPartnerCanonicalBackup", "daily_at": "18:30"}


def append_log(command, result, *, production):
    entry = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "state": result["state"],
        "step": result.get("step", command),
        "error_type": result.get("error_type"),
        "snapshot": result.get("snapshot"),
        "production": production,
        "heartbeat": result.get("heartbeat", "DISABLED"),
    }
    if "pruned" in result:
        entry["pruned"] = result["pruned"]
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(entry, sort_keys=True) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--database", type=Path, default=DATABASE)
    run.add_argument("--destination", type=Path, default=DESTINATION)
    run.add_argument("--staging", type=Path, default=STAGING)
    run.add_argument("--require-heartbeat", action="store_true")
    commands.add_parser("register")
    restore = commands.add_parser("restore-check")
    restore.add_argument("download", type=Path)
    restore.add_argument("scratch", type=Path)
    args = parser.parse_args(argv)
    production = args.command == "run" and production_paths(args.database, args.destination)
    try:
        if args.command == "run":
            result = run_backup(args.database, args.destination, args.staging, require_heartbeat=args.require_heartbeat)
        elif args.command == "register":
            result = register()
        else:
            result = restore_check(args.download, args.scratch)
    except Exception as exc:
        result = {"state": "FAILED", "step": args.command, "error_type": type(exc).__name__}
        if args.command == "restore-check" and SNAPSHOT_NAME.fullmatch(args.download.name):
            result["snapshot"] = args.download.name
    try:
        append_log(args.command, result, production=production)
    except OSError as exc:
        result["log_error_type"] = type(exc).__name__
    print(json.dumps(result, sort_keys=True))
    return int(result["state"] == "FAILED" or "log_error_type" in result)


if __name__ == "__main__":
    raise SystemExit(main())
