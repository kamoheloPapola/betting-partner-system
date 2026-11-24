"""Canonical local operator entry point. Environment key only; no dotenv import."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import re
import sys
import sqlite3

from . import heartbeat
from .clock import CanonicalError,canonical
from .store import Store
from .provider import FootballData,LEAGUES
from .service import run_once,health
from .markets import probabilities
from .notifications import notify_local

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_DB=ROOT/"data/canonical/football.sqlite3"


def forecast_rows(store,league=None,limit=200):
    result=[]
    for row in store.forecasts(league,limit):
        item=dict(row)
        item["markets"]=probabilities(row["home_xg"],row["away_xg"],version=row["market_version"])
        result.append(item)
    return result


def parser():
    p=argparse.ArgumentParser(description="Canonical goals-only local football loop")
    p.add_argument("--database",type=Path,default=DEFAULT_DB)
    commands=p.add_subparsers(dest="command",required=True)
    run=commands.add_parser("run",help="Fetch, freeze eligible upcoming forecasts and settle finished fixtures once")
    run.add_argument("--stored-key",action="store_true",help="Use the current Windows user's DPAPI-protected key")
    run.add_argument("--fixture-waiver", type=Path, help="Owner-approved waiver for one due fixture")
    run.add_argument("--fixture-waiver-sha256", help="Independently reviewed waiver digest")
    commands.add_parser("status",help="Read freshness, team mapping rejections, failures and halt state")
    forecasts=commands.add_parser("forecasts",help="Read frozen forecasts; never regenerate")
    forecasts.add_argument("--league",choices=LEAGUES)
    forecasts.add_argument("--limit",type=int,default=200)
    server=commands.add_parser("serve",help="Read-only local dashboard/API")
    server.add_argument("--port",type=int,default=8765)
    backup=commands.add_parser("backup",help="Create an exclusive SQLite backup snapshot")
    backup.add_argument("destination",type=Path)
    adoption=commands.add_parser("adopt",help="Explicitly adopt a hash-pinned model after all foundation gates")
    adoption.add_argument("artifact",type=Path)
    adoption.add_argument("--sha256",required=True)
    adoption.add_argument("--reason",required=True)
    for command in ("rollback","resume"):
        decision=commands.add_parser(command,help="Record an explicit operator recovery decision")
        decision.add_argument("--reason",required=True)
    check=commands.add_parser("check-model",help="Read-only activation gate; does not adopt")
    check.add_argument("artifact",type=Path)
    check.add_argument("--sha256",required=True)
    commands.add_parser("configure-key",help="Protect the environment key for this Windows user's scheduled job")
    commands.add_parser("scheduler-ready",help="Read-only checks before registering the canonical task")
    return p


def main(argv=None):
    args=parser().parse_args(argv)
    heartbeat_attempt = heartbeat.Attempt() if args.command == "run" else None
    try:
        if args.command=="configure-key":
            from .credentials import save_key
            result=save_key(args.database,os.environ.get("FOOTBALL_DATA_API_KEY"))
            print(canonical(result))
            return 0
        if args.command=="scheduler-ready":
            from .credentials import load_key
            from .lifecycle import validate_adoption
            with Store(args.database,readonly=True) as store:
                reference=store.active_model()
                validate_adoption(reference["artifact_path"],reference["artifact_sha256"])
                if store.halted():raise CanonicalError("LIVE_BUG_GUARD_HALT")
                state=health(store)
                if state["reasons"] or set(state["warnings"]).difference(heartbeat.WARNINGS.values()):
                    raise CanonicalError("SUCCESSFUL_LIVE_SMOKE_REQUIRED")
                configuration = heartbeat.status()
                if configuration["state"] == "NOT_CONFIGURED":
                    raise CanonicalError("HEARTBEAT_NOT_CONFIGURED")
                if state["heartbeat"]["state"] == "FAILED":
                    raise CanonicalError("HEARTBEAT_FAILED")
            load_key(args.database)
            print(canonical({"state":"SCHEDULER_READY","model_id":reference["id"],"heartbeat":configuration}))
            return 0
        if args.command in ("check-model","adopt"):
            from .lifecycle import validate_adoption,adopt
            model,pin,gate=validate_adoption(args.artifact,args.sha256)
            if args.command=="check-model":
                print(canonical({"state":"ADOPTION_GATES_PASSED","model_id":model.model_id,"artifact":pin}))
                return 0
            with Store(args.database) as store:
                result=adopt(store,args.artifact,args.sha256,args.reason,datetime.now(timezone.utc))
            print(canonical(result))
            return 0
        if args.command=="serve":
            import uvicorn
            from .api import create_app
            if not 1024<=args.port<=65535:raise CanonicalError("INVALID_LOCAL_PORT")
            uvicorn.run(create_app(args.database),host="127.0.0.1",port=args.port,access_log=False)
            return 0
        if args.command=="run":
            if (args.fixture_waiver is None) != (args.fixture_waiver_sha256 is None):
                raise CanonicalError("FORECAST_WAIVER_PATH_AND_PIN_REQUIRED")
            waiver = (args.fixture_waiver, args.fixture_waiver_sha256) if args.fixture_waiver is not None else None
            # Validate the key before creating a database or making a request.
            if args.stored_key:
                from .credentials import load_key
                key=load_key(args.database)
            else:key=os.environ.get("FOOTBALL_DATA_API_KEY")
            client=FootballData(key)
            with Store(args.database) as store:
                run_result=run_once(store,client,notifier=notify_local,fixture_waiver=waiver,
                                    heartbeat_attempt=heartbeat_attempt)
            print(canonical(run_result))
            return run_result["exit_code"]
        if not args.database.exists():
            result={"state":"NEEDS_SETUP","error_code":"DATABASE_NOT_INITIALIZED"}
            if args.command=="status":
                result["team_mapping"]={"state":"NOT_EVALUATED"}
                result["heartbeat"] = heartbeat.status()
                warning = heartbeat.WARNINGS.get(result["heartbeat"]["state"])
                if warning:
                    result["warnings"] = [warning]
            print(canonical(result))
            return 2
        if args.command=="status":
            with Store(args.database,readonly=True) as store:result=health(store)
        elif args.command=="forecasts":
            with Store(args.database,readonly=True) as store:result={"forecasts":forecast_rows(store,args.league,args.limit)}
        elif args.command in ("rollback","resume"):
            from .lifecycle import rollback,resume
            action=rollback if args.command=="rollback" else resume
            with Store(args.database) as store:result=action(store,args.reason,datetime.now(timezone.utc))
        else:
            with Store(args.database) as store:path=store.backup(args.destination)
            result={"state":"BACKUP_CREATED","path":str(path)}
        print(canonical(result))
        return 0
    except KeyboardInterrupt:
        print(canonical({"state":"INTERRUPTED"}))
        return 130
    except Exception as exc:
        if args.command == "scheduler-ready":
            failure = {"state": "NOT_READY", "error_type": type(exc).__name__}
            if isinstance(exc, CanonicalError) and isinstance(exc.code, str) and re.fullmatch(r"[A-Z][A-Z0-9_]{0,99}", exc.code):
                failure["error_code"] = exc.code
            print(canonical(failure))
            return 2
        if isinstance(exc, (CanonicalError,OSError,ValueError,KeyError,sqlite3.Error)):
            code=exc.code if isinstance(exc,CanonicalError) else "OPERATOR_"+type(exc).__name__.upper()
            failure = {"state":"FAILED","error_code":code}
            if heartbeat_attempt is not None:
                failure["heartbeat"] = heartbeat_attempt.report("FAILED")
            print(canonical(failure))
            if args.command=="run":notify_local(code)
            return 2
        if heartbeat_attempt is not None:
            heartbeat_attempt.report("FAILED")
        raise


if __name__=="__main__":
    raise SystemExit(main())

