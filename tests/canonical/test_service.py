import json
import hashlib
import sqlite3
from contextlib import contextmanager
from datetime import datetime,timezone,timedelta
from types import SimpleNamespace
from uuid import uuid4
import pytest

import test_provider as provider_fixtures
from src.canonical.clock import CanonicalError, utc
from src.canonical import cli, heartbeat, notifications as desktop, service
from src.canonical.store import Store
from src.canonical.model import Model,MODEL_SCHEMA,PARAMS,TARGET_CAP
from src.canonical.features import FEATURE_NAMES,FEATURE_VERSION
from src.canonical.markets import MARKET_VERSION
from src.canonical.service import _settled_guard_rows, health, guardrail
from src.canonical.lifecycle import resume
from src.canonical.provider import FootballData, LEAGUES
from src.ingest.provider_snapshots import scan_files
from test_team_mapping_consumer import binding, publication, publish, state


class Clock:
    def __init__(self):self.at=datetime(2026,8,1,tzinfo=timezone.utc)
    def utcnow(self):return self.at


class Client:
    def __init__(self,records=None,error=None,*,fetched_at=None,missing=(),empty=()):
        self.records=records or [];self.error=error;self.receipts=[];self.calls=0
        self.fetched_at, self.missing, self.empty = fetched_at, set(missing), set(empty)

    def fetch(self,pairs):
        self.calls+=1
        if self.error:raise self.error
        records = list(self.records)
        for league, season in pairs:
            if league in self.missing or league in self.empty:
                continue
            if not any((r["league"], r["season"]) == (league, season) for r in records):
                records.append(fixture(800000 + season * 10 + LEAGUES.index(league), f"{season}-07-01T00:00:00Z",
                                       "CANCELLED", league=league, season=season))
        completed = utc(self.fetched_at or self.clock.at)
        self.receipts = [{"league": league, "season": season, "http_status": 200, "completed_utc": completed}
                         for league, season in pairs if league not in self.missing]
        return records, self.receipts


def run_once(store, client, *, clock, **kwargs):
    if isinstance(client, Client):
        client.clock = clock
    return service.run_once(store, client, clock=clock, **kwargs)


def fixture(mid,kickoff,status="TIMED",**extra):
    return {"id":mid,"league":"PL","season":2026,"home_id":57,"away_id":61,
      "home_name":"Home","away_name":"Away","kickoff_utc":kickoff,"status":status,**extra}


@pytest.fixture
def setup(tmp_path, publication, monkeypatch, request):
    _, document, _, _, config = publication
    document["entries"].extend([
        binding(57, "HOME LEGACY", league="PL"),
        binding(61, "AWAY LEGACY", league="PL"),
    ])
    seed = config["feature_seed"]
    seed["last_day"] = "2026-07-01"
    home_state = {**state(), "elo": 1600.}
    seed["states"].extend([
        {"league": "PL", "team_id": "HOME LEGACY", "state": home_state},
        {"league": "PL", "team_id": "AWAY LEGACY", "state": state()},
    ])
    if getattr(request, "param", None) == "all_leagues":
        for league in LEAGUES:
            if league != "PL":
                document["entries"].extend([binding(57, "HOME LEGACY", league=league),
                                            binding(61, "AWAY LEGACY", league=league)])
                seed["states"].extend([{"league": league, "team_id": label, "state": state()}
                                       for label in ("HOME LEGACY", "AWAY LEGACY")])
    if getattr(request, "param", None) == "bridged":
        document["entries"][0] = binding(81, "BARCELONA")
        ids = {row["legacy_label"]: str(row["provider_id"]) for row in document["entries"]
               if row["kind"] == "BINDING"}
        seed.update(identity_key="provider_id", last_day="2026-07-29")
        for row in seed["states"]:
            row["team_id"] = ids[row["team_id"]]
    publish(publication)
    monkeypatch.setattr(service, "ROOT", tmp_path)
    coeff={"mean":[0.]*17,"scale":[1.]*17,"coef":[0.]*17,"intercept":0.}
    home_coeff = {**coeff, "coef": [0.] * 14 + [.001, 0., 0.], "intercept": -1.5}
    payload={"schema":MODEL_SCHEMA,"feature_names":list(FEATURE_NAMES),"feature_version":FEATURE_VERSION,
      "market_version":MARKET_VERSION,"parameters":dict(PARAMS),"target_cap":TARGET_CAP,
      "coefficients":{"home":home_coeff,"away":dict(coeff)},"feature_seed":seed,
      "baseline_1x2":{"PL":[.4,.3,.3]},
      "training":{"source_manifest_sha256":"0" * 64},
      "serving":{"source_manifest_sha256":config["source_manifest_sha256"],
                 "team_mapping":{"mapping_sha256":config["mapping_sha256"],
                                 "accepted_input_sha256":config["accepted_input_sha256"]}}}
    model=Model(payload);path=tmp_path/"model.json"
    validation={"status":"PASS_LOCKED_REPLAY","model_id":model.model_id,"foundation_proofs":{
        "source_manifest":{"path":str(config["source_manifest"]),"sha256":config["source_manifest_sha256"]},
        "identity_decisions":{"path":str(config["identity_decisions"]),
                              "sha256":config["accepted_input_sha256"]["decision_register"]},
    }}
    path.write_text(json.dumps({"model":payload,"validation":validation}))
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    with Store(tmp_path/"store.sqlite3") as store:
        store.register_model(model.model_id,path,sha,"2024-07-01T00:00:00Z")
        store.activate_model(model.model_id,"2024-07-01T00:00:00Z","SYNTHETIC_TEST")
        yield store,Clock(),path,model


def test_fetch_freeze_settle_second_run_is_business_noop(setup):
    store,clock,path,model=setup
    rows=[fixture(1,"2026-07-28T15:00:00Z","FINISHED",home_goals=1,away_goals=0),
          fixture(2,"2026-08-02T15:00:00Z")]
    first=run_once(store,Client(rows),clock=clock)
    assert first["state"]=="SUCCESS" and first["counts"]["forecasts_frozen"]==1
    original=dict(store.db.execute("SELECT * FROM forecasts").fetchone())
    assert (original["run_id"], original["waiver_sha256"]) == (first["run_id"], None)
    assert (first["counts"]["run_id"], first["counts"]["waiver_sha256"]) == (first["run_id"], None)
    assert json.loads(store.last_run()["counts_json"]) == first["counts"]
    assert first["counts"]["team_mapping"]["history"]["resolved"] == 1
    assert original["home_xg"] > 1.105170918
    clock.at+=timedelta(minutes=1)
    second=run_once(store,Client(rows),clock=clock)
    assert second["counts"]["forecasts_frozen"]==0 and second["counts"]["results_added"]==0
    assert dict(store.db.execute("SELECT * FROM forecasts").fetchone())==original
    status = health(store, at=clock.at)
    assert status["state"] == "YELLOW" and status["warnings"] == ["HEARTBEAT_NOT_CONFIGURED", "MISSED_FORECASTS"]


@pytest.mark.parametrize(("scenario", "state", "exit_code", "withheld"), [
    ("clean", "SUCCESS", 0, []),
    ("withheld", "PARTIAL_FORECASTS", 1, ["PL"]),
    ("whole-run-block", "SETTLEMENT_ONLY", 1, sorted(LEAGUES)),
    ("mapping-withhold", "SUCCESS", 0, ["PL"]),
    ("guard-halt", "HALTED", 3, []),
    ("exception", "FAILED", 1, []),
])
def test_run_heartbeat_follows_the_committed_receipt(setup, monkeypatch, scenario, state, exit_code, withheld):
    store, clock, _, _ = setup
    monkeypatch.setenv(heartbeat.ENV_VAR, "https://heartbeat.example.test/check?key=test-value")
    records = [fixture(1, clock.at + timedelta(hours=9), league="PD", home_id=2, away_id=3)]
    client = Client(records)
    if scenario == "clean":
        # An eligible historical miss has no effect on the current run's heartbeat.
        client.records.append(fixture(2, clock.at - timedelta(hours=1), "IN_PLAY"))
    elif scenario == "withheld":
        client.records.append(fixture(2, clock.at - timedelta(hours=4), "TIMED"))
    elif scenario == "whole-run-block":
        client.missing.add("SA")
    elif scenario == "mapping-withhold":
        client.records.append(fixture(2, clock.at - timedelta(hours=2), "FINISHED",
                                      home_id=999, home_goals=1, away_goals=0))
    elif scenario == "guard-halt":
        store.halt({"code": "SYNTHETIC_HALT"})
    elif scenario == "exception":
        client.error = RuntimeError("synthetic unhandled fetch error")

    calls = []

    def transport(url, body, timeout):
        with Store(store.path, readonly=True) as reader:
            row = reader.last_run()
        calls.append((url.path, url.query, json.loads(body), timeout, row["status"], json.loads(row["counts_json"])))
        return 200

    result = run_once(store, client, clock=clock, heartbeat_transport=transport)
    assert (result["state"], result["exit_code"]) == (state, exit_code)
    assert len(calls) == 1
    path, query, body, timeout, db_status, receipt = calls[0]
    assert query == "key=test-value" and timeout == 10
    assert body == {"run_id": result["run_id"], "state": state, "withheld_leagues": withheld}
    assert receipt["run_id"] == result["run_id"] and "heartbeat" not in receipt
    assert db_status == ("FAILED" if exit_code == 1 else "SUCCESS")
    assert path == ("/check" if scenario == "clean" else "/check/fail")
    assert result["heartbeat"] == {"state": "SENT", "kind": "SUCCESS" if scenario == "clean" else "FAILURE"}
    assert json.loads(store.last_run()["counts_json"])["heartbeat"] == result["heartbeat"]
    status = health(store, at=clock.at)
    assert status["heartbeat"] == result["heartbeat"]
    if scenario == "clean":
        assert status["warnings"] == ["MISSED_FORECASTS"]


@pytest.mark.parametrize("blocked", [False, True])
def test_heartbeat_failure_is_masked_and_preserves_run(setup, monkeypatch, capsys, caplog, tmp_path, blocked):
    store, clock, _, _ = setup
    token = uuid4().hex
    monkeypatch.setenv(heartbeat.ENV_VAR, f"https://heartbeat.example.test/{token}")
    client = Client([fixture(1, clock.at + timedelta(hours=9))], missing=["SA"] if blocked else [])
    client.clock = clock
    attempts = []

    def unreachable(url, body, timeout):
        attempts.append((json.loads(body), timeout))
        raise TimeoutError(url.geturl())

    monkeypatch.setattr(heartbeat, "_https_get", unreachable)
    monkeypatch.setattr(service, "RealClock", lambda: clock)
    monkeypatch.setattr(cli, "health", lambda store: health(store, at=clock.at))
    monkeypatch.setattr(cli, "FootballData", lambda key: client)
    monkeypatch.setattr(cli, "notify_local", lambda code: None)
    args = ["--database", str(store.path)]
    assert cli.main([*args, "run"]) == int(blocked)
    run = capsys.readouterr()
    result = json.loads(run.out)
    assert result["state"] == ("SETTLEMENT_ONLY" if blocked else "SUCCESS")
    assert result["counts"]["forecasts_frozen"] == len(store.forecasts()) == int(not blocked)
    assert attempts == [({"run_id": result["run_id"], "state": result["state"],
                          "withheld_leagues": sorted(LEAGUES) if blocked else []}, 10)] * 2
    assert result["heartbeat"] == {"state": "FAILED", "kind": "FAILURE" if blocked else "SUCCESS",
                                   "error_type": "TimeoutError"}
    assert cli.main([*args, "status"]) == 0
    output = capsys.readouterr()
    status = json.loads(output.out)
    assert status["state"] == ("RED" if blocked else "YELLOW")
    assert status["warnings"] == ["HEARTBEAT_FAILED"]
    assert status["heartbeat"] == result["heartbeat"]
    assert token not in run.out + run.err + output.out + output.err + caplog.text
    scan_files(tmp_path, token)


def test_interrupted_run_sends_no_heartbeat(setup, monkeypatch):
    store, clock, _, _ = setup
    monkeypatch.setenv(heartbeat.ENV_VAR, "https://heartbeat.example.test/check")
    with pytest.raises(KeyboardInterrupt):
        run_once(store, Client(error=KeyboardInterrupt()), clock=clock, heartbeat_transport=lambda *_: pytest.fail())
    assert store.last_run()["status"] == "RUNNING"


def test_cli_startup_error_sends_failure_without_run_id(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(heartbeat.ENV_VAR, "https://heartbeat.example.test/check")
    monkeypatch.delenv("FOOTBALL_DATA_API_KEY", raising=False)
    monkeypatch.setattr(cli, "notify_local", lambda code: None)
    calls = []

    def transport(url, body, timeout):
        calls.append((url.path, json.loads(body)))
        return 200

    monkeypatch.setattr(heartbeat, "_https_get", transport)
    db = tmp_path / "missing.sqlite3"
    assert cli.main(["--database", str(db), "run"]) == 2
    assert not db.exists()
    assert calls == [("/check/fail", {"run_id": None, "state": "FAILED", "withheld_leagues": []})]
    assert json.loads(capsys.readouterr().out)["heartbeat"] == {"state": "SENT", "kind": "FAILURE"}


@pytest.mark.parametrize("reraises", [False, True], ids=["caught", "reported-then-reraised"])
def test_cli_run_exception_reports_once_with_run_id(setup, monkeypatch, capsys, reraises):
    store, clock, _, _ = setup
    client = Client(error=RuntimeError("synthetic fetch failure"))
    monkeypatch.setenv(heartbeat.ENV_VAR, "https://heartbeat.example.test/check")
    monkeypatch.setattr(cli, "FootballData", lambda _: client)
    monkeypatch.setattr(service, "RealClock", lambda: clock)
    monkeypatch.setattr(cli, "notify_local", lambda _: None)
    calls = []

    def transport(url, body, timeout):
        calls.append((url.path, json.loads(body)))
        return 200

    monkeypatch.setattr(heartbeat, "_https_get", transport)
    args = ["--database", str(store.path), "run"]
    if reraises:
        error = TypeError("synthetic error after reporting")

        def run_then_raise(*args, **kwargs):
            service.run_once(*args, **kwargs)
            raise error

        monkeypatch.setattr(cli, "run_once", run_then_raise)
        with pytest.raises(TypeError) as raised:
            cli.main(args)
        assert raised.value is error
    else:
        assert cli.main(args) == 1
        assert json.loads(capsys.readouterr().out)["heartbeat"] == {"state": "SENT", "kind": "FAILURE"}
    assert calls == [("/check/fail", {"run_id": store.last_run()["id"], "state": "FAILED", "withheld_leagues": []})]


def test_cli_unexpected_startup_error_reports_once_and_preserves_exception(tmp_path, monkeypatch):
    monkeypatch.setenv(heartbeat.ENV_VAR, "https://heartbeat.example.test/check")
    error = TypeError("synthetic startup error")
    calls = []

    def unavailable(_):
        raise error

    def transport(url, body, timeout):
        calls.append((url.path, json.loads(body)))
        return 200

    monkeypatch.setattr(cli, "FootballData", unavailable)
    monkeypatch.setattr(heartbeat, "_https_get", transport)
    with pytest.raises(TypeError) as raised:
        cli.main(["--database", str(tmp_path / "missing.sqlite3"), "run"])
    assert raised.value is error
    assert calls == [("/check/fail", {"run_id": None, "state": "FAILED", "withheld_leagues": []})]


def test_cli_delivery_receipt_failure_keeps_success_and_never_sends_failure(setup, monkeypatch, capsys):
    store, clock, _, _ = setup
    client = Client([fixture(1, clock.at + timedelta(hours=9))])
    client.clock = clock
    monkeypatch.setenv(heartbeat.ENV_VAR, "https://heartbeat.example.test/check")
    monkeypatch.setattr(cli, "FootballData", lambda _: client)
    monkeypatch.setattr(service, "RealClock", lambda: clock)
    calls, toasts = [], []
    monkeypatch.setattr(cli, "notify_local", toasts.append)

    def transport(url, body, timeout):
        calls.append((url.path, json.loads(body)))
        return 200

    def authorizer(action, arg1, arg2, database, trigger):
        if action == sqlite3.SQLITE_FUNCTION and arg2 == "json_set":
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    def store_with_failed_heartbeat_write(path):
        job_store = Store(path)
        job_store.db.set_authorizer(authorizer)
        return job_store

    monkeypatch.setattr(heartbeat, "_https_get", transport)
    monkeypatch.setattr(cli, "Store", store_with_failed_heartbeat_write)
    assert cli.main(["--database", str(store.path), "run"]) == 0
    result = json.loads(capsys.readouterr().out)
    receipt = store.last_run()
    assert result["state"] == receipt["status"] == "SUCCESS"
    assert result["exit_code"] == 0 and len(store.forecasts()) == 1
    assert result["heartbeat"] == {"state": "SENT", "kind": "SUCCESS"}
    assert result["heartbeat_record_error_type"] == "OperationalError"
    assert "heartbeat" not in json.loads(receipt["counts_json"])
    assert calls == [("/check", {"run_id": receipt["id"], "state": "SUCCESS", "withheld_leagues": []})]
    assert toasts == []


def test_scheduler_readiness_requires_config_and_blocks_failed_delivery(setup, monkeypatch, capsys):
    from src.canonical import credentials, lifecycle

    store, clock, _, _ = setup
    result = run_once(store, Client(), clock=clock)
    assert result["heartbeat"] == {"state": "DISABLED", "kind": "SUCCESS"}
    status = health(store, at=clock.at)
    assert status["state"] == "YELLOW"
    assert status["heartbeat"] == {"state": "NOT_CONFIGURED", "kind": "SUCCESS"}
    assert status["warnings"] == ["HEARTBEAT_NOT_CONFIGURED"]

    monkeypatch.setattr(cli, "health", lambda store: health(store, at=clock.at))
    monkeypatch.setattr(lifecycle, "validate_adoption", lambda *_: None)
    monkeypatch.setattr(credentials, "load_key", lambda _: None)
    args = ["--database", str(store.path), "scheduler-ready"]
    for invalid_url in (None, "http://heartbeat.example.test/check"):
        if invalid_url is not None:
            monkeypatch.setenv(heartbeat.ENV_VAR, invalid_url)
        assert cli.main(args) == 2
        assert json.loads(capsys.readouterr().out) == {
            "state": "NOT_READY", "error_type": "CanonicalError", "error_code": "HEARTBEAT_NOT_CONFIGURED",
        }
    monkeypatch.setenv(heartbeat.ENV_VAR, "https://heartbeat.example.test/check")
    assert cli.main(args) == 0
    readiness = json.loads(capsys.readouterr().out)
    assert readiness["state"] == "SCHEDULER_READY"
    assert readiness["heartbeat"] == {"state": "NOT_EVALUATED", "kind": None}

    result = run_once(store, Client(), clock=clock, heartbeat_transport=lambda *_: 503)
    assert result["state"] == "SUCCESS" and result["heartbeat"]["state"] == "FAILED"
    assert cli.main(args) == 2
    assert json.loads(capsys.readouterr().out) == {
        "state": "NOT_READY", "error_type": "CanonicalError", "error_code": "HEARTBEAT_FAILED",
    }

    store.halt({"code": "SYNTHETIC_HALT"})
    assert cli.main(args) == 2
    assert json.loads(capsys.readouterr().out) == {
        "state": "NOT_READY", "error_type": "CanonicalError", "error_code": "LIVE_BUG_GUARD_HALT",
    }


def test_fetch_failure_preserves_business_state_and_is_visible(setup):
    store,clock,_,_=setup
    notifications=[]
    result=run_once(store,Client(error=CanonicalError("PROVIDER_AUTH_OR_ENTITLEMENT_403")),clock=clock,
                    notifier=lambda code:notifications.append(code))
    assert result["exit_code"]==1 and notifications==["PROVIDER_AUTH_OR_ENTITLEMENT_403"]
    assert store.db.execute("SELECT COUNT(*) FROM fixtures").fetchone()[0]==0


@pytest.mark.parametrize("status, reason", [
    ("IN_PLAY", "DUE_FIXTURE_UNRESOLVED"), ("SUSPENDED", "DUE_FIXTURE_UNRESOLVED"),
    ("UNKNOWN", "UNKNOWN_FIXTURE_STATUS"), ("FINISHED", "FINISHED_SCORE_MISSING_OR_INVALID"),
])
def test_forecast_gate_preserves_valid_settlement_and_alerts_without_forecasting(setup, status, reason):
    store, clock, _, _ = setup
    records = [
        fixture(1, clock.at - timedelta(days=1), "FINISHED", home_goals=0, away_goals=1),
        fixture(2, clock.at - service.RESULT_SETTLE_MARGIN, status),
        fixture(3, clock.at + timedelta(hours=9)),
    ]
    alerts = []
    result = run_once(store, Client(records), clock=clock, notifier=alerts.append)
    assert result["state"] == "SETTLEMENT_ONLY"
    assert result["error_code"] == service.COMPLETENESS_ERROR
    assert result["counts"]["results_added"] == 1
    assert not store.forecasts()
    run = store.last_run()
    assert run["id"] == result["run_id"]
    receipt = json.loads(run["counts_json"])
    assert (receipt["run_id"], receipt["waiver_sha256"]) == (result["run_id"], None)
    assert store.db.execute("SELECT COUNT(*) FROM forecasts WHERE run_id=?", (result["run_id"],)).fetchone()[0] == 0
    affected = sorted(LEAGUES) if status in {"UNKNOWN", "FINISHED"} else ["PL"]
    assert alerts == ["_".join([service.COMPLETENESS_ERROR, *affected])]
    status = health(store, at=clock.at)
    assert status["state"] == "RED" and service.COMPLETENESS_ERROR in status["reasons"]
    assert status["forecast_completeness"]["issues"][0]["reason"] == reason


@pytest.mark.parametrize("setup", ["all_leagues"], indirect=True)
@pytest.mark.parametrize("affected", [("SA",), LEAGUES], ids=["one-league", "all-leagues"])
def test_completeness_withholds_only_affected_leagues_and_recovers(setup, affected):
    store, clock, _, _ = setup
    upcoming = [fixture(100+i, clock.at + timedelta(hours=9), league=league)
                for i, league in enumerate(LEAGUES)]
    finished = [fixture(200+i, clock.at - timedelta(days=2), "FINISHED", league=league, home_goals=2, away_goals=0)
                for i, league in enumerate(LEAGUES)]
    due = [fixture(300+i, clock.at - timedelta(hours=4), "TIMED", league=league)
           for i, league in enumerate(affected)]
    alerts = []

    def notify(code):
        alerts.append((code, store.last_run()["status"], store.db.execute("SELECT COUNT(*) FROM results").fetchone()[0]))

    result = run_once(store, Client(upcoming + finished + due), clock=clock, notifier=notify)
    assert result["state"] == ("SETTLEMENT_ONLY" if len(affected) == 5 else "PARTIAL_FORECASTS")
    assert result["exit_code"] == 1 and result["error_code"] == service.COMPLETENESS_ERROR
    assert alerts == [("_".join([service.COMPLETENESS_ERROR, *sorted(affected)]), "FAILED", 5)]
    coverage = result["counts"]["forecast_completeness"]
    assert coverage["state"] == ("BLOCKED" if len(affected) == 5 else "PARTIAL")
    assert not coverage["global_block"] and coverage["withheld_leagues"] == sorted(affected)
    saved = {row["fixture_id"]: row for row in store.forecasts()}
    assert saved.keys() == {row["id"] for row in upcoming if row["league"] not in affected}
    status = health(store, at=clock.at)
    assert status["state"] == "RED" and status["forecast_completeness"] == coverage
    after_kickoff = health(store, at=clock.at + timedelta(hours=10))
    assert after_kickoff["missed_forecast_count"] == status["missed_forecast_count"] + len(affected)
    for row in upcoming:
        league = row["league"]
        withheld = league in affected
        assert coverage["leagues"][league] == {
            "state": "WITHHELD" if withheld else "WRITTEN",
            "forecasts_written": int(not withheld),
            "withhold_fixture_ids": [x["id"] for x in due if x["league"] == league],
            "withheld_forecasts": [{"fixture_id": row["id"], "kickoff_utc": utc(row["kickoff_utc"])}] if withheld else [],
        }

    clock.at += timedelta(minutes=1)
    resolved = [{**row, "status": "FINISHED", "home_goals": 1, "away_goals": 0} for row in due]
    retry = run_once(store, Client(upcoming + finished + resolved), clock=clock, notifier=notify)
    assert retry["state"] == "SUCCESS" and retry["run_id"] != result["run_id"]
    assert retry["counts"]["forecasts_frozen"] == len(affected)
    rows = {row["fixture_id"]: row for row in store.forecasts()}
    assert {key: rows[key] for key in saved} == saved
    assert all(row["run_id"] == retry["run_id"] for key, row in rows.items() if key not in saved)
    assert len(alerts) == 1


def test_due_fixture_receipt_uses_raw_payload_metadata(setup):
    store, _, _, _ = setup
    provider_clock = provider_fixtures.Clock()
    kickoff = "2026-07-31T19:00:00+01:00"
    last_updated = "2026-07-31T21:03:00Z"
    responses = []
    for i, league in enumerate(LEAGUES):
        if league == "PL":
            row = {**provider_fixtures.match("TIMED"), "utcDate": kickoff, "lastUpdated": last_updated}
        else:
            row = {**provider_fixtures.match("CANCELLED"), "id": 100+i, "utcDate": "2026-08-03T00:00:00Z"}
        responses.append(provider_fixtures.response(body=provider_fixtures.payload([row], league)))
    client = FootballData(provider_fixtures.KEY, clock=provider_clock,
                          transport=provider_fixtures.Transport(responses, provider_clock))
    result = run_once(store, client, clock=provider_clock)
    coverage = result["counts"]["forecast_completeness"]
    assert coverage["issues"] == [{
        "reason": "DUE_FIXTURE_UNRESOLVED", "fixture_id": 1, "status": "TIMED", "league": "PL",
        "season": 2026, "kickoff_utc": utc(kickoff), "utcDate": kickoff, "lastUpdated": last_updated,
        "fetch_completed_utc": utc("2026-08-01T00:00:00Z"),
    }]
    assert coverage["withheld_leagues"] == ["PL"]
    assert coverage["leagues"]["SA"]["state"] == "NO_OP"


@pytest.mark.parametrize("condition, reason", [
    ("missing", "COMPETITION_PAYLOAD_MISSING_OR_DUPLICATE"), ("empty", "COMPETITION_PAYLOAD_EMPTY"),
])
def test_forecast_gate_requires_a_nonempty_payload_from_each_competition(setup, condition, reason):
    store, clock, _, _ = setup
    result = run_once(store, Client([fixture(1, clock.at + timedelta(hours=9))], **{condition: ["SA"]}),
                      clock=clock)
    assert result["error_code"] == service.COMPLETENESS_ERROR
    assert result["counts"]["forecast_completeness"]["global_block"]
    assert result["counts"]["forecast_completeness"]["issues"] == [
        {"league": "SA", "season": 2026, "reason": reason}
    ]
    assert not store.forecasts()


@pytest.mark.parametrize("status, disposition", [
    ("POSTPONED", "deferred_postponed"), ("AWARDED", "excluded_awarded"), ("CANCELLED", "excluded_cancelled"),
])
def test_forecast_gate_accounts_for_due_non_played_statuses(setup, status, disposition):
    store, clock, _, _ = setup
    result = run_once(store, Client([fixture(1, clock.at - timedelta(days=1), status),
                                    fixture(2, clock.at + timedelta(hours=9))]), clock=clock)
    assert result["state"] == "SUCCESS"
    assert result["counts"]["forecast_completeness"][disposition] == 1
    assert result["counts"]["forecast_completeness"]["non_played_fixtures"] == [
        {"fixture_id": 1, "league": "PL", "season": 2026, "status": status, "disposition": disposition}
    ]
    assert result["counts"]["results_added"] == 0
    assert [row["fixture_id"] for row in store.forecasts()] == [2]


def test_forecast_gate_uses_payload_fetch_time_not_later_run_time(setup):
    store, clock, _, _ = setup
    fetched_at = clock.at
    in_play = fixture(1, fetched_at - service.RESULT_SETTLE_MARGIN + timedelta(seconds=1), "IN_PLAY")
    upcoming = fixture(2, fetched_at + timedelta(hours=9))

    class DelayedClient(Client):
        def fetch(self, pairs):
            result = super().fetch(pairs)
            clock.at += timedelta(minutes=1)
            return result

    result = run_once(store, DelayedClient([in_play, upcoming], fetched_at=fetched_at), clock=clock)
    assert result["state"] == "SUCCESS"
    assert result["counts"]["forecast_completeness"]["due"] == 0
    assert [row["fixture_id"] for row in store.forecasts()] == [2]


def test_forecast_gate_blocks_duplicate_fixture_ids(setup):
    store, clock, _, _ = setup
    match = fixture(1, clock.at - timedelta(days=1), "FINISHED", home_goals=1, away_goals=2)
    result = run_once(store, Client([match, {**match}, fixture(2, clock.at + timedelta(hours=9))]), clock=clock)
    assert result["error_code"] == service.COMPLETENESS_ERROR
    assert result["counts"]["forecast_completeness"]["global_block"]
    assert result["counts"]["forecast_completeness"]["issues"] == [
        {"reason": "DUPLICATE_FIXTURE_ID", "fixture_id": 1}
    ]
    assert result["counts"]["results_added"] == 0 and not store.forecasts()


@pytest.mark.parametrize("present", [True, False], ids=["regression", "missing-known-fixture"])
def test_forecast_gate_detects_regressed_or_omitted_result_on_later_run(setup, present):
    store, clock, _, _ = setup
    finished = fixture(1, clock.at - timedelta(days=1), "FINISHED", home_goals=1, away_goals=0)
    first = run_once(store, Client([finished]), clock=clock)
    assert first["state"] == "SUCCESS" and first["counts"]["forecast_completeness"]["applied_finished"] == 1
    clock.at += timedelta(hours=1)
    upcoming = fixture(2, clock.at + timedelta(hours=9))
    records = [{**finished, "status": "AWARDED"}, upcoming] if present else [upcoming]
    second = run_once(store, Client(records), clock=clock)
    assert second["error_code"] == service.COMPLETENESS_ERROR
    issue = second["counts"]["forecast_completeness"]["issues"][0]
    assert second["counts"]["forecast_completeness"]["global_block"] is present
    assert issue["reason"] == ("FINISHED_STATUS_REGRESSION" if present else "KNOWN_DUE_FIXTURE_MISSING")
    assert issue["fixture_id"] == 1 and not store.forecasts()


def test_forecast_gate_blocks_due_finished_result_rejected_by_mapping(setup):
    store, clock, _, _ = setup
    unmapped = fixture(1, clock.at - timedelta(days=1), "FINISHED", home_id=999, home_goals=2, away_goals=1)
    upcoming = fixture(2, clock.at + timedelta(hours=9), league="PD", home_id=2, away_id=3)
    result = run_once(store, Client([unmapped, upcoming]), clock=clock)
    assert result["error_code"] == service.COMPLETENESS_ERROR and result["counts"]["results_added"] == 1
    coverage = result["counts"]["forecast_completeness"]
    assert coverage["withheld_leagues"] == ["PL"]
    assert [issue["reason"] for issue in coverage["issues"]] == ["FINISHED_RESULT_NOT_APPLIED"]
    assert [row["fixture_id"] for row in store.forecasts()] == [2]
    assert health(store, at=clock.at)["state"] == "RED"


@pytest.mark.parametrize("fault", [None, "pin", "expired", "kickoff", "fixture", "status",
                                   "second-fixture", "other-league-fixture", "regression"])
def test_pinned_waiver_accounts_for_one_due_fixture_and_is_logged(setup, tmp_path, fault):
    store, clock, _, _ = setup
    stuck = fixture(1, clock.at - timedelta(days=2), "SUSPENDED")
    upcoming = fixture(2, clock.at + timedelta(hours=9))
    other_league = fixture(4, clock.at + timedelta(hours=10), league="PD", home_id=2, away_id=3)
    records = [stuck, upcoming, other_league]
    if fault == "regression":
        with store.transaction():
            store.ingest([{**stuck, "status": "FINISHED", "home_goals": 1, "away_goals": 0}],
                         clock.at - timedelta(days=1))
    if fault == "second-fixture":
        records.append(fixture(3, clock.at - timedelta(days=1), "IN_PLAY"))
    if fault == "other-league-fixture":
        records.append(fixture(3, clock.at - timedelta(days=1), "IN_PLAY", league="PD", home_id=2, away_id=3))

    waiver = {
        "schema_version": 1, "status": "APPROVED", "league": "PL", "season": 2026,
        "fixture_id": 777 if fault == "fixture" else stuck["id"],
        "kickoff_utc": utc(stuck["kickoff_utc"] + timedelta(minutes=(1 if fault == "kickoff" else 0))),
        "provider_status": "IN_PLAY" if fault == "status" else "SUSPENDED", "approved_by": "Synthetic operator",
        "reason": "Competition confirmed play is suspended",
        "approved_utc": utc(clock.at - timedelta(hours=2)),
        "expires_utc": utc(clock.at + timedelta(hours=(-1 if fault == "expired" else 1))),
    }
    path = tmp_path / "fixture-waiver.json"
    raw = json.dumps(waiver).encode()
    path.write_bytes(raw)
    pin = "0" * 64 if fault == "pin" else hashlib.sha256(raw).hexdigest()
    result = run_once(store, Client(records), clock=clock, fixture_waiver=(path, pin))
    if fault is None:
        assert result["state"] == "SUCCESS"
        waived_rows = {row["fixture_id"]: row for row in store.forecasts()}
        assert waived_rows.keys() == {2, 4}
        assert {(row["run_id"], row["waiver_sha256"]) for row in waived_rows.values()} == {
            (result["run_id"], pin)
        }
        assert store.history(clock.at) == []
        receipt = json.loads(store.last_run()["counts_json"])
        assert (receipt["run_id"], receipt["waiver_sha256"]) == (result["run_id"], pin)
        assert receipt["forecast_completeness"]["waiver"] == {
            **waiver, "sha256": pin, "fetch_completed_utc": utc(clock.at),
        }
        status = health(store, at=clock.at)
        assert status["state"] == "YELLOW" and "FORECAST_HISTORY_WAIVER_USED" in status["warnings"]

        clean_records = [{**stuck, "status": "POSTPONED"}, upcoming, other_league,
                         fixture(5, clock.at + timedelta(hours=11))]
        clean = run_once(store, Client(clean_records), clock=clock)
        assert clean["state"] == "SUCCESS" and clean["run_id"] != result["run_id"]
        clean_row, = store.db.execute("SELECT * FROM forecasts WHERE run_id=?", (clean["run_id"],))
        assert clean_row["fixture_id"] == 5 and clean_row["waiver_sha256"] is None
        assert json.loads(store.last_run()["counts_json"])["waiver_sha256"] is None
        persisted = {row["fixture_id"]: row for row in store.forecasts()}
        assert {key: persisted[key] for key in waived_rows} == waived_rows
    else:
        expected = "FORECAST_WAIVER_PIN_OR_READ_FAILED" if fault == "pin" else service.COMPLETENESS_ERROR
        assert result["error_code"] == expected
        forecasts = store.forecasts()
        if fault in ("pin", "regression"):
            assert result["state"] == "SETTLEMENT_ONLY" and not forecasts
        else:
            assert result["state"] == "PARTIAL_FORECASTS"
            coverage = result["counts"]["forecast_completeness"]
            assert coverage["withheld_leagues"] == (["PD"] if fault == "other-league-fixture" else ["PL"])
            expected_id = 2 if fault == "other-league-fixture" else 4
            expected_pin = pin if fault in ("second-fixture", "other-league-fixture") else None
            assert [(r["fixture_id"], r["run_id"], r["waiver_sha256"]) for r in forecasts] == [
                (expected_id, result["run_id"], expected_pin)
            ]
        assert health(store, at=clock.at)["state"] == "RED"


@pytest.mark.parametrize("failure", ["integrity-error", "interface-error"])
def test_store_failure_rolls_back_forecast_batch_and_preserves_settlement(setup, monkeypatch, failure):
    store, clock, _, model = setup
    prior = fixture(1, clock.at - timedelta(days=1))
    frozen_at = clock.at - timedelta(days=2)
    with store.transaction():
        store.ingest([prior], frozen_at)
        assert store.freeze(prior, frozen_at, 1.25, 1., model.model_id, MARKET_VERSION, "x" * 64, None,
                            run_id=store.start_run(frozen_at))
    official = dict(store.db.execute("SELECT * FROM forecasts").fetchone())
    records = [
        {**prior, "status": "FINISHED", "home_goals": 2, "away_goals": 0},
        fixture(2, clock.at + timedelta(hours=4)),
        fixture(3, clock.at + timedelta(hours=5)),
    ]
    notified = []
    freeze = store.freeze

    def fail_interface(row, *args, **kwargs):
        if row["id"] == 3:
            raise sqlite3.InterfaceError("synthetic driver failure")
        return freeze(row, *args, **kwargs)

    with monkeypatch.context() as fault:
        if failure == "interface-error":
            fault.setattr(store, "freeze", fail_interface)
            expected = "RUN_FAILED_INTERFACEERROR"
        else:
            # ROLLBACK aborts the entire forecast transaction inside SQLite.
            store.db.execute("""CREATE TEMP TRIGGER fail_second_freeze BEFORE INSERT ON forecasts
                WHEN NEW.fixture_id = 3 BEGIN SELECT RAISE(ROLLBACK, 'synthetic write failure'); END""")
            expected = "RUN_FAILED_INTEGRITYERROR"
        failed = run_once(store, Client(records), clock=clock, notifier=notified.append)
    if failure == "integrity-error":
        store.db.execute("DROP TRIGGER fail_second_freeze")

    assert failed["state"] == "SETTLEMENT_ONLY" and failed["exit_code"] == 1
    assert failed["error_code"] == expected and notified == [expected]
    assert failed["counts"]["results_added"] == 1 and failed["counts"]["forecasts_frozen"] == 0
    assert failed["counts"]["forecast_errors"] == 0
    with Store(store.path, readonly=True) as reader:
        settled, = reader.forecasts()
        assert (settled["fixture_id"], settled["settlement"], settled["home_goals"]) == (1, "SETTLED", 2)
        assert dict(reader.db.execute("SELECT * FROM forecasts").fetchone()) == official
        assert reader.last_success() is None
        assert health(reader, at=clock.at)["reasons"] == ["NO_SUCCESSFUL_RUN", expected]

    retry = run_once(store, Client(records), clock=clock)
    assert retry["state"] == "SUCCESS" and retry["counts"]["forecasts_frozen"] == 2
    assert retry["counts"]["results_added"] == 0


@pytest.mark.parametrize("stage", ["features", "prediction", "markets", "freeze"])
def test_bad_forecast_fixture_isolated_and_retry_freezes_it_once(setup, monkeypatch, stage):
    store, clock, _, _ = setup
    preserved = fixture(1, clock.at + timedelta(hours=1))
    assert run_once(store, Client([preserved]), clock=clock)["counts"]["forecasts_frozen"] == 1
    original = dict(store.db.execute("SELECT * FROM forecasts WHERE fixture_id = 1").fetchone())
    fixtures = [preserved, *(fixture(i, clock.at + timedelta(hours=i)) for i in range(2, 5))]
    active_fixture = None
    vector, predict, markets, freeze = service.FeatureEngine.vector, Model.predict, service.probabilities, store.freeze

    def fail_at(point):
        if active_fixture == 3 and stage == point:
            raise RuntimeError("synthetic secret must never appear in the traceback log")

    def build_features(engine, row):
        nonlocal active_fixture
        active_fixture = row["id"]
        fail_at("features")
        return vector(engine, row)

    def predict_fixture(model, features):
        fail_at("prediction")
        return predict(model, features)

    def validate_markets(*args, **kwargs):
        fail_at("markets")
        return markets(*args, **kwargs)

    def freeze_then_fail(*args, **kwargs):
        inserted = freeze(*args, **kwargs)
        fail_at("freeze")
        return inserted

    with monkeypatch.context() as fault:
        fault.setattr(service.FeatureEngine, "vector", build_features)
        fault.setattr(Model, "predict", predict_fixture)
        fault.setattr(service, "probabilities", validate_markets)
        fault.setattr(store, "freeze", freeze_then_fail)
        result = run_once(store, Client(fixtures), clock=clock)
    assert result["state"] == "SUCCESS" and result["counts"]["forecasts_frozen"] == 2
    assert result["counts"]["forecast_errors"] == 1
    assert result["counts"]["forecast_error_details"] == [{"fixture_id": 3, "error_code": "FORECAST_FAILED_RUNTIMEERROR"}]
    assert {row["fixture_id"] for row in store.forecasts()} == {1, 2, 4}
    partial_rows = {row["fixture_id"]: row for row in store.forecasts() if row["run_id"] == result["run_id"]}
    assert partial_rows.keys() == {2, 4}
    assert all(row["waiver_sha256"] is None for row in partial_rows.values())
    assert json.loads(store.last_run()["counts_json"])["run_id"] == result["run_id"]
    status = health(store, at=clock.at)
    assert status["state"] == "YELLOW" and status["warnings"] == ["HEARTBEAT_NOT_CONFIGURED", "FORECAST_ERRORS"]
    trace = store.path.with_name("forecast-errors.log").read_text(encoding="utf-8")
    assert "fixture_id=3" in trace and "Traceback" in trace and "in fail_at" in trace
    assert "synthetic secret" not in trace

    retry = run_once(store, Client(fixtures), clock=clock)
    assert retry["counts"]["forecasts_frozen"] == 1 and retry["counts"]["forecast_errors"] == 0
    assert retry["run_id"] != result["run_id"]
    inserted, = store.db.execute("SELECT fixture_id FROM forecasts WHERE run_id=?", (retry["run_id"],))
    assert inserted["fixture_id"] == 3
    assert {row["fixture_id"]: row for row in store.forecasts() if row["run_id"] == result["run_id"]} == partial_rows
    repeat = run_once(store, Client(fixtures), clock=clock)
    assert repeat["counts"]["forecasts_frozen"] == 0
    assert {row["fixture_id"] for row in store.forecasts()} == {1, 2, 3, 4}
    assert dict(store.db.execute("SELECT * FROM forecasts WHERE fixture_id = 1").fetchone()) == original


def test_all_forecast_fixtures_failed_is_red_and_caps_error_details(setup, monkeypatch):
    store, clock, _, _ = setup

    def broken_features(*args):
        raise CanonicalError("INVALID_FEATURE_VECTOR")

    monkeypatch.setattr(service.FeatureEngine, "vector", broken_features)
    count = service.FORECAST_ERROR_DETAIL_LIMIT + 2
    fixtures = [fixture(i, clock.at + timedelta(hours=1)) for i in range(1, count + 1)]
    notifications = []
    result = run_once(store, Client(fixtures), clock=clock, notifier=notifications.append)
    assert result["state"] == "SETTLEMENT_ONLY" and result["error_code"] == "ALL_FORECASTS_FAILED"
    assert result["exit_code"] == 1 and notifications == ["ALL_FORECASTS_FAILED"]
    assert result["counts"]["forecast_errors"] == count and store.forecasts() == []
    assert len(result["counts"]["forecast_error_details"]) == service.FORECAST_ERROR_DETAIL_LIMIT
    assert store.last_run()["status"] == "FAILED"
    status = health(store, at=clock.at)
    assert status["state"] == "RED" and "ALL_FORECASTS_FAILED" in status["reasons"]
    assert status["warnings"] == ["HEARTBEAT_NOT_CONFIGURED", "FORECAST_ERRORS"]


def test_multiple_forecast_errors_notify_once_and_clean_retry_does_not_notify(setup, monkeypatch):
    store, clock, _, _ = setup
    vector = service.FeatureEngine.vector
    fixtures = [fixture(i, clock.at + timedelta(hours=1)) for i in range(1, 4)]
    notifications = []
    receipt = {"attempted": True, "delivery_requested": True}

    def fail_for_two(engine, row):
        if row["id"] in (1, 2):
            raise CanonicalError("INVALID_FEATURE_VECTOR")
        return vector(engine, row)

    def notifier(code):
        notifications.append((code, store.last_run()["status"]))
        return receipt

    with monkeypatch.context() as fault:
        fault.setattr(service.FeatureEngine, "vector", fail_for_two)
        result = run_once(store, Client(fixtures), clock=clock, notifier=notifier)
    assert result["state"] == "SUCCESS"
    assert result["counts"]["forecast_errors"] == 2 and result["counts"]["forecasts_frozen"] == 1
    assert result["notification"] == receipt
    assert notifications == [("FORECAST_ERRORS", "SUCCESS")]
    assert health(store, at=clock.at)["state"] == "YELLOW"

    retry = run_once(store, Client(fixtures), clock=clock, notifier=notifier)
    assert retry["counts"]["forecast_errors"] == 0 and retry["counts"]["forecasts_frozen"] == 2
    assert "notification" not in retry and len(notifications) == 1


@pytest.mark.parametrize("returncode, output, confirmed", [
    (0, b"BALLOON_SHOWN\r\n", True),
    (0, b"", False),
    (1, b"BALLOON_SHOWN\r\n", False),
])
def test_local_notification_requires_windows_display_callback(monkeypatch, returncode, output, confirmed):
    monkeypatch.setattr(desktop, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(desktop.subprocess, "run", lambda *a, **kw: SimpleNamespace(
        returncode=returncode, stdout=output))
    receipt = desktop.notify_local("FORECAST_ERRORS")
    assert receipt["delivery_requested"] == (returncode == 0)
    assert receipt["display_confirmed"] == confirmed


def test_activation_change_between_settlement_and_forecasting_preserves_results(setup, monkeypatch):
    store, clock, _, model = setup
    transaction = store.transaction

    @contextmanager
    def switch_after_settlement():
        with transaction():
            yield
        with Store(store.path) as other:
            other.activate_model(model.model_id, clock.at, "SYNTHETIC_REACTIVATION")

    monkeypatch.setattr(store, "transaction", switch_after_settlement)
    result = run_once(store, Client([
        fixture(1, clock.at - timedelta(days=1), "FINISHED", home_goals=1, away_goals=0),
        fixture(2, clock.at + timedelta(hours=3)),
    ]), clock=clock)
    assert result["state"] == "SETTLEMENT_ONLY" and result["error_code"] == "ACTIVE_MODEL_CHANGED_DURING_RUN"
    assert result["counts"]["results_added"] == 1 and store.forecasts() == []
    assert store.db.execute("SELECT COUNT(*) FROM results").fetchone()[0] == 1


def test_model_tamper_stops_before_network(setup):
    store,clock,path,_=setup;path.write_text("{}")
    client=Client()
    result=run_once(store,client,clock=clock)
    assert result["error_code"]=="MODEL_ARTIFACT_HASH_MISMATCH" and client.calls==0


def test_cli_run_and_status_use_mapping_and_keep_rejected_results(setup, monkeypatch, capsys):
    store, clock, _, model = setup
    now = clock.at
    upcoming = now + timedelta(hours=1)
    records = [
        fixture(1, upcoming),
        fixture(2, upcoming + timedelta(days=1), home_id=999, home_name="Home"),
        fixture(3, now - timedelta(hours=2), "FINISHED", home_id=999, home_goals=3, away_goals=1),
        fixture(4, upcoming + timedelta(hours=36), league="PD", home_id=81, away_id=2),
        fixture(5, upcoming + timedelta(hours=1), league="PD", home_id=2, away_id=3),
    ]
    previously_frozen = {**records[2], "status": "TIMED"}
    frozen_at = now - timedelta(days=2)
    with store.transaction():
        store.ingest([previously_frozen], frozen_at)
        assert store.freeze(previously_frozen, frozen_at, 1.25, 1., model.model_id, MARKET_VERSION, "x" * 64, None,
                            run_id=store.start_run(frozen_at))
    client = Client(records)
    client.clock = clock
    monkeypatch.setattr(service, "RealClock", lambda: clock)
    monkeypatch.setattr(cli, "health", lambda store: health(store, at=clock.at))
    monkeypatch.setenv("FOOTBALL_DATA_API_KEY", "SYNTHETIC_OFFLINE_TEST_KEY")
    monkeypatch.setattr(cli, "FootballData", lambda key: client)
    monkeypatch.setattr(cli, "notify_local", lambda code: pytest.fail(code))
    args = ["--database", str(store.path)]

    assert cli.main([*args, "run"]) == 0
    run = json.loads(capsys.readouterr().out)
    mapping = run["counts"]["team_mapping"]
    assert mapping == {"state": "DEGRADED", "resolved": 2, "rejected": 2,
                       "rejection_reasons": {"TEAM_STATE_AFTER_SEED_WINDOW": 1, "UNKNOWN_TEAM_ID": 1},
                       "withheld_leagues": {"PL": ["UNKNOWN_TEAM_ID"]},
                       "history": {"state": "DEGRADED", "resolved": 0, "rejected": 1,
                                   "rejection_reasons": {"UNKNOWN_TEAM_ID": 1}}}
    forecast, settled = store.forecasts()
    assert forecast["fixture_id"] == 5
    assert forecast["history_known_utc"] is None
    assert (settled["fixture_id"], settled["settlement"], settled["home_xg"]) == (3, "SETTLED", 1.25)
    assert store.db.execute("SELECT COUNT(*) FROM results").fetchone()[0] == 1
    assert client.calls == 1

    assert cli.main([*args, "status"]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["team_mapping"] == mapping and status["state"] == "YELLOW"
    assert status["warnings"] == ["HEARTBEAT_NOT_CONFIGURED", "TEAM_MAPPING_FIXTURE_REJECTIONS",
                                  "TEAM_MAPPING_HISTORY_REJECTIONS"]
    assert client.calls == 1


def test_seed_window_rejection_turns_health_yellow_without_withholding_league(setup):
    store, clock, _, _ = setup
    records = [fixture(1, "2026-08-02T15:00:00Z", league="PD", home_id=81, away_id=2)]
    result = run_once(store, Client(records), clock=clock)
    assert result["counts"]["forecasts_frozen"] == 0
    assert result["counts"]["team_mapping"]["rejection_reasons"] == {"TEAM_STATE_AFTER_SEED_WINDOW": 1}
    assert result["counts"]["team_mapping"]["withheld_leagues"] == {}
    status = health(store, at=clock.at)
    assert status["state"] == "YELLOW"
    assert status["warnings"] == ["HEARTBEAT_NOT_CONFIGURED", "TEAM_MAPPING_FIXTURE_REJECTIONS"]


def test_barcelona_history_cannot_produce_a_partial_feature_vector(setup, monkeypatch):
    store, clock, _, _ = setup
    records = [
        fixture(1, "2026-07-31T22:00:00Z", "FINISHED", league="PD",
                home_id=2, away_id=81, home_goals=0, away_goals=4),
        fixture(2, "2026-08-02T15:00:00Z", league="PD", home_id=2, away_id=3),
    ]
    seen = []
    predict = Model.predict

    def capture_features(model, vector):
        seen.append(dict(zip(FEATURE_NAMES, vector, strict=True)))
        return predict(model, vector)

    monkeypatch.setattr(Model, "predict", capture_features)
    result = run_once(store, Client(records), clock=clock)
    history = result["counts"]["team_mapping"]["history"]
    assert history["rejection_reasons"] == {"TEAM_STATE_AFTER_SEED_WINDOW": 1}

    # If a forecast reaches the model, the vector must include Team Two's 0-4 loss.
    if seen:
        home_win_expectation = 1 / (1 + 10 ** (-75 / 400))
        complete = {"home_rolling_goals_scored_3": 1.0, "home_rolling_goals_conceded_3": 2.5,
                    "home_form_rating": 2.7, "home_elo": 1510 - 20 * 1.875 * home_win_expectation}
        assert {key: seen[0][key] for key in complete} == pytest.approx(complete)

    assert seen == [] and store.forecasts() == []
    assert result["state"] == "SUCCESS" and result["counts"]["results_added"] == 1
    withheld = {"PD": ["TEAM_STATE_AFTER_SEED_WINDOW"]}
    assert result["counts"]["team_mapping"]["withheld_leagues"] == withheld
    status = health(store, at=clock.at)
    assert status["state"] == "YELLOW"
    assert status["warnings"] == ["HEARTBEAT_NOT_CONFIGURED", "TEAM_MAPPING_HISTORY_REJECTIONS"]
    assert status["team_mapping"]["withheld_leagues"] == withheld


@pytest.mark.parametrize("reason", ["UNKNOWN_TEAM_ID", "DUPLICATE_PROVIDER_FIXTURE"])
def test_unexpected_history_rejection_withholds_league_and_turns_health_yellow(setup, monkeypatch, reason):
    store, clock, _, _ = setup
    history = fixture(1, "2026-07-31T22:00:00Z", "FINISHED", home_goals=2, away_goals=1)
    if reason == "UNKNOWN_TEAM_ID":
        history["away_id"] = 999
        rejected = 1
    else:
        read_history = store.history

        # SQLite exposes one revision per fixture. Inject a duplicate at the read boundary.
        def duplicate_history(before):
            rows = read_history(before)
            return rows + rows

        monkeypatch.setattr(store, "history", duplicate_history)
        rejected = 2

    result = run_once(store, Client([history, fixture(2, "2026-08-02T15:00:00Z")]), clock=clock)
    assert result["state"] == "SUCCESS" and result["counts"]["results_added"] == 1
    mapping = result["counts"]["team_mapping"]
    assert mapping["resolved"] == 1 and mapping["rejected"] == 0
    assert mapping["history"]["rejection_reasons"] == {reason: rejected}
    assert mapping["withheld_leagues"] == {"PL": [reason]}
    assert store.forecasts() == []
    status = health(store, at=clock.at)
    assert status["state"] == "YELLOW" and status["team_mapping"] == mapping
    warnings = ["HEARTBEAT_NOT_CONFIGURED", "TEAM_MAPPING_HISTORY_REJECTIONS"]
    if reason == "DUPLICATE_PROVIDER_FIXTURE":
        warnings.append("MISSED_FORECASTS")
    assert status["warnings"] == warnings


def test_empty_history_withholds_no_leagues(setup):
    store, clock, _, _ = setup
    records = [fixture(1, "2026-08-02T15:00:00Z"),
               fixture(2, "2026-08-02T16:00:00Z", league="PD", home_id=2, away_id=3)]
    result = run_once(store, Client(records), clock=clock)
    mapping = result["counts"]["team_mapping"]
    assert mapping["history"]["resolved"] == mapping["history"]["rejected"] == 0
    assert mapping["withheld_leagues"] == {}
    assert {forecast["fixture_id"] for forecast in store.forecasts()} == {1, 2}
    status = health(store, at=clock.at)
    assert status["state"] == "YELLOW" and status["team_mapping"] == mapping
    assert status["warnings"] == ["HEARTBEAT_NOT_CONFIGURED"]


def test_history_rejections_withhold_two_leagues_in_one_run(setup):
    store, clock, _, _ = setup
    records = [
        fixture(1, "2026-07-31T22:00:00Z", "FINISHED", home_id=999, home_goals=2, away_goals=1),
        fixture(2, "2026-07-31T22:00:00Z", "FINISHED", league="PD",
                home_id=2, away_id=81, home_goals=0, away_goals=4),
        fixture(3, "2026-08-02T15:00:00Z"),
        fixture(4, "2026-08-02T16:00:00Z", league="PD", home_id=2, away_id=3),
    ]
    result = run_once(store, Client(records), clock=clock)
    assert result["state"] == "SUCCESS" and result["counts"]["results_added"] == 2
    assert result["counts"]["forecasts_frozen"] == 0 and store.forecasts() == []
    withheld = {"PD": ["TEAM_STATE_AFTER_SEED_WINDOW"], "PL": ["UNKNOWN_TEAM_ID"]}
    assert result["counts"]["team_mapping"]["withheld_leagues"] == withheld
    status = health(store, at=clock.at)
    assert status["state"] == "YELLOW" and status["team_mapping"]["withheld_leagues"] == withheld


def test_clean_league_forecasts_alongside_withheld_league_and_rerun_is_idempotent(setup):
    store, clock, _, _ = setup
    records = [
        fixture(1, "2026-07-30T15:00:00Z", "FINISHED", home_goals=2, away_goals=1),
        fixture(2, "2026-07-31T22:00:00Z", "FINISHED", league="PD",
                home_id=2, away_id=81, home_goals=0, away_goals=4),
        fixture(3, "2026-08-02T15:00:00Z"),
        fixture(4, "2026-08-02T16:00:00Z", league="PD", home_id=2, away_id=3),
    ]
    first = run_once(store, Client(records), clock=clock)
    assert first["counts"]["forecasts_frozen"] == 1
    forecasts = store.forecasts()
    assert [row["fixture_id"] for row in forecasts] == [3]
    assert forecasts[0]["home_xg"] > 1.105170918  # PL's own history still updates its rating.
    withheld = {"PD": ["TEAM_STATE_AFTER_SEED_WINDOW"]}
    assert first["counts"]["team_mapping"]["withheld_leagues"] == withheld

    second = run_once(store, Client(records), clock=clock)
    for count in ("fixtures_changed", "results_added", "results_corrected", "forecasts_frozen"):
        assert second["counts"][count] == 0
    assert second["counts"]["team_mapping"]["withheld_leagues"] == withheld
    assert store.forecasts() == forecasts
    assert health(store, at=clock.at)["state"] == "YELLOW"


@pytest.mark.parametrize("setup", ["bridged"], indirect=True)
def test_bridged_barcelona_and_other_pd_teams_forecast_after_single_history_update(setup, monkeypatch):
    store, clock, _, model = setup
    store.activate_model(model.model_id, clock.at, "SYNTHETIC_LIVE_START")
    records = [
        fixture(1, "2026-07-30T15:00:00Z", "FINISHED", league="PD",
                home_id=81, away_id=2, home_goals=4, away_goals=0),
        fixture(2, "2026-08-01T15:00:00Z", league="PD", home_id=81, away_id=3),
        fixture(3, "2026-08-02T15:00:00Z", league="PD", home_id=2, away_id=3),
    ]
    seen = []
    predict = Model.predict

    def capture_features(model, vector):
        seen.append(dict(zip(FEATURE_NAMES, vector, strict=True)))
        return predict(model, vector)

    monkeypatch.setattr(Model, "predict", capture_features)
    result = run_once(store, Client(records), clock=clock)
    mapping = result["counts"]["team_mapping"]
    assert mapping["history"]["resolved"] == 1 and mapping["history"]["rejected"] == 0
    assert mapping["withheld_leagues"] == {} and mapping["rejected"] == 0
    assert result["counts"]["forecasts_frozen"] == 2 and result["counts"]["results_added"] == 1
    forecasts = store.forecasts()
    assert {(row["fixture_id"], row["home_id"]) for row in forecasts} == {(2, 81), (3, 2)}

    home_win_expectation = 1 / (1 + 10 ** (-75 / 400))
    delta = 20 * 1.875 * (1 - home_win_expectation)
    fields = ("home_rolling_goals_scored_3", "home_rolling_goals_conceded_3", "home_form_rating", "home_elo")
    assert tuple(seen[0][key] for key in fields) == pytest.approx((3., .5, 3., 1510 + delta))
    assert tuple(seen[1][key] for key in fields) == pytest.approx((1., 2.5, 2.7, 1510 - delta))
    rerun = run_once(store, Client(records), clock=clock)
    assert rerun["counts"]["forecasts_frozen"] == rerun["counts"]["results_added"] == 0
    assert len(seen) == 2 and store.forecasts() == forecasts
    status = health(store, at=clock.at)
    assert status["state"] == "YELLOW" and status["warnings"] == ["HEARTBEAT_NOT_CONFIGURED"]


@pytest.mark.parametrize("failure", ["pin", "load", "resolution", "malformed-output"])
def test_mapping_failure_still_settles_existing_forecasts(setup, publication, monkeypatch, failure):
    store, clock, _, model = setup
    prior = fixture(1, "2026-07-31T15:00:00Z")
    frozen_at = "2026-07-30T15:00:00Z"
    with store.transaction():
        store.ingest([prior], frozen_at)
        assert store.freeze(prior, frozen_at, 1.25, 1., model.model_id, MARKET_VERSION, "x" * 64, None,
                            run_id=store.start_run(frozen_at))
    mapping = publication[0]
    if failure == "pin":
        mapping.write_bytes(mapping.read_bytes() + b" ")
        expected = "TEAM_MAPPING_PIN_OR_READ_FAILED"
    elif failure == "malformed-output":
        monkeypatch.setattr(service.TeamMappingConsumer, "resolve", lambda *args: {})
        expected = "TEAM_MAPPING_RUNTIME_KEYERROR"
    else:
        def crash_translator(*args):
            try:
                raise KeyError("TEST_PASSWORD_IN_EXCEPTION_MESSAGE")
            except KeyError as cause:
                raise ValueError("TEST_KEY_IN_EXCEPTION_MESSAGE") from cause
        method = "for_model" if failure == "load" else "resolve"
        monkeypatch.setattr(service.TeamMappingConsumer, method, crash_translator)
        expected = "TEAM_MAPPING_RUNTIME_VALUEERROR"
    client = Client([{**prior, "status": "FINISHED", "home_goals": 2, "away_goals": 0},
                     fixture(2, "2026-08-02T15:00:00Z")])
    notifications = []
    result = run_once(store, client, clock=clock, notifier=notifications.append)
    assert result["state"] == "SETTLEMENT_ONLY" and result["exit_code"] == 1
    assert result["error_code"] == expected and notifications == [expected]
    assert result["counts"]["forecasts_frozen"] == 0 and result["counts"]["results_added"] == 1
    assert client.calls == 1 and store.last_success() is None
    settled, = store.forecasts()
    assert settled["fixture_id"] == 1 and settled["settlement"] == "SETTLED"
    status = health(store, at=clock.at)
    assert status["state"] == "RED"
    assert status["team_mapping"] == {"state": "BLOCKED", "error_code": expected}
    assert status["missed_forecast_count"] is None
    trace_path = store.path.with_name("translation-errors.log")
    if failure == "pin":
        assert not trace_path.exists()
    else:
        trace = trace_path.read_text(encoding="utf-8")
        stage = "load" if failure == "load" else "resolve"
        assert f"run_id={result['run_id']} stage={stage}" in trace
        assert 'File "service.py", line ' in trace and "KeyError [message omitted]" in trace
        if failure != "malformed-output":
            assert "in crash_translator" in trace and "ValueError [message omitted]" in trace
        assert "TEST_PASSWORD_IN_EXCEPTION_MESSAGE" not in trace and "TEST_KEY_IN_EXCEPTION_MESSAGE" not in trace
        assert trace.count("run_id=") == 1


@pytest.mark.parametrize("kickoff", [None, "random text", "2026-08-02T18:00:00"],
                         ids=["none", "random-text", "timezone-less"])
def test_invalid_provider_timestamp_blocks_forecasts_and_reports_rejection(setup, kickoff):
    store, clock, _, _ = setup
    with pytest.raises(CanonicalError):
        utc(kickoff)

    bad = {**provider_fixtures.match(), "id": 2, "utcDate": kickoff}
    good = {**provider_fixtures.match(), "utcDate": utc(clock.at + timedelta(hours=5))}
    replies = [provider_fixtures.response(body=provider_fixtures.payload(
        [good, bad] if league == "PL" else [
            {**provider_fixtures.match("CANCELLED"), "id": 100 + i, "utcDate": "2026-07-01T00:00:00Z"}
        ], league)) for i, league in enumerate(LEAGUES)]
    # The real fetcher receives only synthetic provider responses.
    provider_clock = provider_fixtures.Clock()
    transport = provider_fixtures.Transport(replies, provider_clock)
    client = FootballData(provider_fixtures.KEY, clock=provider_clock, transport=transport)
    result = run_once(store, client, clock=provider_clock)

    assert result["state"] == "SETTLEMENT_ONLY" and result["error_code"] == service.COMPLETENESS_ERROR
    assert result["counts"]["forecast_completeness"]["withheld_leagues"] == sorted(LEAGUES)
    assert result["counts"]["forecast_completeness"]["global_block"]
    assert result["counts"]["invalid_provider_fixtures"] == 1
    assert result["counts"]["request_receipts"][0]["fixture_rejections"] == [
        {"id": 2, "reason": "INVALID_PROVIDER_FIXTURE"},
    ]
    assert store.forecasts() == []
    status = health(store, at=provider_clock.utcnow())
    assert status["state"] == "RED" and status["invalid_provider_fixtures"] == 1
    assert status["warnings"] == ["HEARTBEAT_NOT_CONFIGURED", "INVALID_PROVIDER_FIXTURE"]
    assert not store.path.with_name("translation-errors.log").exists()


def test_malformed_provider_payload_prevents_all_forecasts(setup):
    store, _, _, _ = setup
    provider_clock = provider_fixtures.Clock()
    good = {**provider_fixtures.match(), "utcDate": "2026-08-01T05:00:00Z"}
    replies = [provider_fixtures.response(body=provider_fixtures.payload([good], "PL")),
               provider_fixtures.response(body={"matches": None})]
    client = FootballData(provider_fixtures.KEY, clock=provider_clock,
                          transport=provider_fixtures.Transport(replies, provider_clock))
    result = run_once(store, client, clock=provider_clock)
    assert result["state"] == "FAILED" and result["error_code"] == "MALFORMED_PROVIDER_ROOT"
    assert not store.forecasts()
    assert health(store, at=provider_clock.utcnow())["state"] == "RED"


def test_previous_model_without_mapping_contract_requires_setup(setup):
    store, clock, path, model = setup
    payload = {key: value for key, value in model.payload.items() if key != "serving"}
    previous = Model(payload)
    previous_path = path.with_name("previous.json")
    raw = json.dumps({"model": payload, "validation": {
        "status": "PASS_LOCKED_REPLAY", "model_id": previous.model_id,
    }}).encode("utf-8")
    previous_path.write_bytes(raw)
    store.register_model(previous.model_id, previous_path, hashlib.sha256(raw).hexdigest(), clock.at)
    store.activate_model(previous.model_id, clock.at, "SYNTHETIC_LEGACY_TEST")

    client = Client()
    result = run_once(store, client, clock=clock)
    assert result["state"] == "SETTLEMENT_ONLY" and client.calls == 1
    assert result["error_code"] == "TEAM_MAPPING_CONFIGURATION_REQUIRED"
    assert health(store, at=clock.at)["team_mapping"] == {
        "state": "BLOCKED", "error_code": "TEAM_MAPPING_CONFIGURATION_REQUIRED",
    }


def test_started_fixture_never_gets_forecast(setup):
    store,clock,_,_=setup
    for status in ("IN_PLAY","SCHEDULED","POSTPONED"):
        result=run_once(store,Client([fixture(1,"2026-08-02T15:00:00Z",status)]),clock=clock)
        assert result["counts"]["forecasts_frozen"]==0
    assert not store.forecasts()


def test_kickoff_during_prediction_is_skipped_not_backdated(setup):
    store,_,_,_=setup
    class MovingClock:
        def __init__(self):self.n=0
        def utcnow(self):
            values=[datetime(2026,8,1,14,59,50,tzinfo=timezone.utc),
                    datetime(2026,8,1,14,59,55,tzinfo=timezone.utc),
                    datetime(2026,8,1,15,0,1,tzinfo=timezone.utc)]
            value=values[min(self.n,2)];self.n+=1;return value
    result=run_once(store,Client([fixture(1,"2026-08-01T15:00:00Z")],
                                 fetched_at="2026-08-01T14:59:55Z"),clock=MovingClock())
    assert result["state"]=="SUCCESS" and result["counts"]["skipped_kickoff_during_run"]==1
    assert not store.forecasts()


@pytest.mark.parametrize(("offset", "selected"), [
    pytest.param(timedelta(hours=47, minutes=59), 1, id="now-plus-47h59m"),
    pytest.param(timedelta(hours=48), 1, id="exactly-now-plus-48h"),
    pytest.param(timedelta(hours=48, minutes=1), 0, id="now-plus-48h01m"),
    pytest.param(timedelta(0), 0, id="kickoff-equals-now"),
])
def test_first_forecast_kickoff_boundaries(setup, offset, selected):
    store, clock, _, _ = setup
    kickoff = clock.at + offset

    class DelayedClient(Client):
        def fetch(self, pairs):
            clock.at += timedelta(minutes=2)
            return super().fetch(pairs)

    result = run_once(store, DelayedClient([fixture(1, kickoff)]), clock=clock)

    assert result["state"] == "SUCCESS"
    assert result["counts"]["forecasts_frozen"] == selected
    assert [row["fixture_id"] for row in store.forecasts()] == ([1] if selected else [])


def test_first_seen_inside_window_gets_a_forecast(setup):
    store, clock, _, _ = setup
    kickoff = clock.at + timedelta(hours=36)
    assert run_once(store, Client(), clock=clock)["counts"]["forecasts_frozen"] == 0

    clock.at += timedelta(days=1)
    result = run_once(store, Client([fixture(1, kickoff)]), clock=clock)

    assert result["counts"]["forecasts_frozen"] == 1
    forecast, = store.forecasts()
    assert forecast["frozen_utc"] == utc(clock.at)
    assert forecast["kickoff_utc"] == utc(kickoff)


def test_reschedule_after_freeze_keeps_existing_forecast(setup):
    store, clock, _, _ = setup
    original = fixture(1, clock.at + timedelta(hours=12))
    assert run_once(store, Client([original]), clock=clock)["counts"]["forecasts_frozen"] == 1
    frozen = dict(store.db.execute("SELECT * FROM forecasts").fetchone())

    new_kickoff = clock.at + timedelta(hours=72)
    clock.at += timedelta(hours=24)
    result = run_once(store, Client([dict(original, kickoff_utc=new_kickoff)]), clock=clock)

    assert result["counts"]["fixtures_changed"] == 1
    assert result["counts"]["forecasts_frozen"] == 0
    assert dict(store.db.execute("SELECT * FROM forecasts").fetchone()) == frozen
    assert store.forecasts()[0]["current_kickoff_utc"] == utc(new_kickoff)


def test_healthy_payloads_with_empty_post_cutoff_window_pass(setup):
    store, clock, _, _ = setup
    result = run_once(store, Client(), clock=clock)

    assert result["state"] == "SUCCESS"
    assert result["counts"]["forecast_completeness"]["state"] == "READY"
    assert result["counts"]["forecast_completeness"]["due"] == 0
    assert result["counts"]["forecasts_frozen"] == 0
    assert store.forecasts() == []
    assert health(store, at=clock.at)["missed_forecast_count"] == 0


def test_two_runs_at_same_instant_preserve_all_existing_rows(setup):
    store, clock, _, _ = setup
    client = Client([fixture(1, clock.at + timedelta(hours=6)),
                     fixture(2, clock.at - timedelta(days=1), "FINISHED", home_goals=2, away_goals=1)])
    first = run_once(store, client, clock=clock)
    assert first["counts"]["forecasts_frozen"] == 1
    tables = ("fixtures", "fixture_revisions", "forecasts", "results", "runs")
    before = {table: store.db.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall() for table in tables}

    second = run_once(store, client, clock=clock)

    assert second["state"] == "SUCCESS" and second["run_id"] != first["run_id"]
    assert all(second["counts"][key] == 0
               for key in ("forecasts_frozen", "fixtures_changed", "results_added", "results_corrected"))
    for table, existing_rows in before.items():
        after = store.db.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
        if table == "runs":
            appended = after.pop()
            assert appended["id"] == second["run_id"]
        assert after == existing_rows


@pytest.mark.parametrize(("skipped_runs", "missed"), [
    pytest.param(set(), set(), id="all-runs-succeed"),
    pytest.param({1}, set(), id="one-missed-run"),
    pytest.param({1, 2}, {3, 4}, id="two-consecutive-missed-runs"),
])
def test_daily_18h_sast_week_forecasts_and_counts_missed_fixtures(setup, skipped_runs, missed):
    store, clock, _, _ = setup
    start = datetime(2026, 8, 1, 16, tzinfo=timezone.utc)
    offsets = [(2, 0), (23, 59), (48, 1), (72, 0), (94, 0), (119, 0), (145, 0), (166, 0)]
    kickoffs = {i: start + timedelta(hours=hours, minutes=minutes)
                for i, (hours, minutes) in enumerate(offsets, start=1)}
    for day in range(7):
        clock.at = start + timedelta(days=day)
        if day in skipped_runs:
            continue
        records = [
            fixture(i, kickoff, "FINISHED", home_goals=1, away_goals=0) if kickoff < clock.at
            else fixture(i, kickoff, "IN_PLAY" if kickoff == clock.at else "TIMED")
            for i, kickoff in kickoffs.items()
        ]
        assert run_once(store, Client(records), clock=clock)["state"] == "SUCCESS"

    frozen = store.forecasts()
    assert len(frozen) == len(kickoffs) - len(missed)
    assert {row["fixture_id"] for row in frozen} == kickoffs.keys() - missed
    assert all(row["frozen_utc"] < row["kickoff_utc"] for row in frozen)
    with Store(store.path, readonly=True) as reader:
        assert health(reader, at=start + timedelta(days=7))["missed_forecast_count"] == len(missed)


def test_missed_forecast_count_excludes_mapping_rejections_and_unconfirmed_kickoffs(setup):
    store, clock, _, _ = setup
    past = clock.at - timedelta(hours=1)
    records = [
        fixture(1, past, "IN_PLAY"),
        fixture(2, past, "IN_PLAY", home_id=999),
        fixture(3, past, "IN_PLAY", league="PD", home_id=81, away_id=2),
        fixture(4, past, "POSTPONED"),
        fixture(5, past, "SCHEDULED"),
        fixture(6, past, "CANCELLED"),
        fixture(7, past, "AWARDED"),
        fixture(8, clock.at + timedelta(hours=60)),
    ]
    assert run_once(store, Client(records), clock=clock)["counts"]["forecasts_frozen"] == 0
    with Store(store.path, readonly=True) as reader:
        status = health(reader, at=clock.at)
        assert status["missed_forecast_count"] == 1 and status["state"] == "YELLOW"
        assert status["warnings"] == ["HEARTBEAT_NOT_CONFIGURED", "MISSED_FORECASTS"]


def test_missed_forecast_count_starts_strictly_after_latest_activation(setup):
    store, clock, _, model = setup
    activated = clock.at - timedelta(hours=1)
    store.activate_model(model.model_id, activated, "SYNTHETIC_LIVE_START")
    backfill = [
        fixture(1, activated - timedelta(days=2), "FINISHED", home_goals=2, away_goals=0),
        fixture(2, activated, "FINISHED", home_goals=1, away_goals=0),
    ]
    assert run_once(store, Client(backfill), clock=clock)["counts"]["results_added"] == 2
    first_health = health(store, at=clock.at)
    assert first_health["missed_forecast_count"] == 0 and first_health["state"] == "YELLOW"
    assert first_health["warnings"] == ["HEARTBEAT_NOT_CONFIGURED"]

    missing = fixture(3, activated + timedelta(microseconds=1), "IN_PLAY")
    assert run_once(store, Client([*backfill, missing]), clock=clock)["state"] == "SUCCESS"
    status = health(store, at=clock.at)
    assert status["missed_forecast_count"] == 1 and status["state"] == "YELLOW"
    assert status["warnings"] == ["HEARTBEAT_NOT_CONFIGURED", "MISSED_FORECASTS"]


def test_stale_health_is_red_even_after_previous_success(setup):
    store,clock,_,_=setup
    run_once(store,Client(),clock=clock)
    clock.at+=timedelta(hours=37)
    assert "STALE_DATA" in health(store,at=clock.at)["reasons"]


def seed_bad_forecasts(store, model, first_id, count, *, frozen_at, kickoff, observed_at):
    # Storage-only synthetic replay: all rates deliberately wrong for home winners.
    with store.transaction():
        run_id = store.start_run(frozen_at)
        for offset in range(count):
            day = kickoff + timedelta(days=offset)
            row = fixture(first_id + offset, day.isoformat())
            store.ingest([row], frozen_at)
            assert store.freeze(row, frozen_at, .1, 4., model.model_id, MARKET_VERSION, "x" * 64, None, run_id=run_id)
            store.ingest([dict(row, status="FINISHED", home_goals=1, away_goals=0)], observed_at)


def test_guard_waits_for_300_eligible_forecasts_and_halt_persists(setup):
    store,clock,_,model=setup
    kickoff = datetime(2025, 1, 3, tzinfo=timezone.utc)
    times = {"frozen_at": "2025-01-01T00:00:00Z", "observed_at": "2026-07-31T00:00:00Z"}
    seed_bad_forecasts(store, model, 1, 299, kickoff=kickoff, **times)
    assert guardrail(store)["state"] == "COLLECTING"
    seed_bad_forecasts(store, model, 300, 1, kickoff=kickoff + timedelta(days=299), **times)
    assert guardrail(store)["state"]=="HALT"
    result=run_once(store,Client([fixture(301,"2026-08-02T15:00:00Z")]),clock=clock)
    assert result["state"]=="HALTED" and result["exit_code"]==3 and store.halted()
    assert result["counts"]["forecasts_frozen"]==0
    # A subsequent empty successful fetch must not auto-clear the halt.
    assert run_once(store,Client(),clock=clock)["state"]=="HALTED"
    assert health(store, at=clock.at)["guardrail"] == {"state": "HALTED"}

    resumed_at = resume(store, "Corrected the cause and reviewed the halt", clock.at)
    clock.at += timedelta(minutes=1)
    result = run_once(store, Client([fixture(302, "2026-08-02T18:00:00Z")]), clock=clock)
    assert result["state"] == "SUCCESS"
    assert result["counts"]["guardrail"] == {
        "state": "COLLECTING", "n": 0, "required": 300,
        "window_started_after_utc": resumed_at["window_started_after_utc"],
    }
    assert result["counts"]["forecasts_frozen"] == 2
    assert store.halted() is None
    status = health(store, at=clock.at)
    assert status["state"] == "YELLOW"
    assert status["warnings"] == ["GUARD_REARMING", "HEARTBEAT_NOT_CONFIGURED"]
    assert status["guardrail"] == {**result["counts"]["guardrail"], "state": "GUARD_REARMING"}


def test_guard_rearms_after_300_settled_forecasts_frozen_strictly_after_resume(setup):
    store, clock, _, model = setup
    store.halt({"code": "LIVE_BUG_GUARD", "observed_utc": "2026-08-01T00:00:00Z"})
    resume(store, "Reviewed the previous halt", clock.at)
    observed = datetime(2027, 8, 1, tzinfo=timezone.utc)
    kickoff = datetime(2026, 8, 3, tzinfo=timezone.utc)
    seed_bad_forecasts(store, model, 1, 1, frozen_at=clock.at,
                       kickoff=kickoff, observed_at=observed)
    clock.at += timedelta(seconds=1)
    seed_bad_forecasts(store, model, 2, 299, frozen_at=clock.at,
                       kickoff=kickoff, observed_at=observed)
    clock.at = observed
    returned = [fixture(i, kickoff + timedelta(days=max(0, i - 2)), "FINISHED", home_goals=1, away_goals=0)
                for i in range(1, 301)]
    result = run_once(store, Client(returned), clock=clock)
    assert result["state"] == "SUCCESS", result
    assert result["counts"]["guardrail"]["state"] == "COLLECTING"
    assert result["counts"]["guardrail"]["n"] == 299

    seed_bad_forecasts(store, model, 301, 1, frozen_at="2026-08-02T00:00:00Z",
                       kickoff=kickoff + timedelta(days=299), observed_at=observed)
    status = health(store, at=clock.at)
    assert status["guardrail"]["state"] == "ARMED"
    assert status["guardrail"]["n"] == 300
    assert "GUARD_REARMING" not in status["warnings"]
    result = run_once(store, Client(), clock=clock)
    assert result["state"] == "HALTED"
    assert result["counts"]["guardrail"]["n"] == 300


def test_guard_settlement_selection_matches_store_forecasts(setup):
    store, clock, _, model = setup
    before = clock.at
    after = before + timedelta(seconds=1)
    kickoff = before + timedelta(days=1)
    observed = kickoff + timedelta(days=1)
    with store.transaction():
        for fixture_id in range(1, 9):
            row = fixture(fixture_id, kickoff)
            frozen = before if fixture_id == 1 else after
            store.ingest([row], frozen)
            assert store.freeze(row, frozen, .1, 4., model.model_id, MARKET_VERSION, "x" * 64, None,
                                run_id=store.start_run(frozen))
            result = dict(row, status="FINISHED", home_goals=1, away_goals=0)
            if fixture_id in (1, 2, 6, 7):
                if fixture_id == 6:
                    result["kickoff_utc"] = before
                store.ingest([result], observed)
            if fixture_id in (4, 5, 7):
                status = {4: "CANCELLED", 5: "AWARDED", 7: "TIMED"}[fixture_id]
                store.ingest([dict(row, status=status)], observed + timedelta(seconds=1))
            if fixture_id == 8:
                # Current fixture status can be present without a result row.
                store.db.execute("UPDATE fixtures SET status = 'FINISHED' WHERE id = ?", (fixture_id,))

    forecasts = store.forecasts()
    assert [entry["settlement"] for entry in forecasts] == [
        "SETTLED", "SETTLED", "PENDING", "VOID", "VOID",
        "INVALID_RESCHEDULED_BEFORE_FREEZE", "PENDING", "PENDING",
    ]
    for cutoff, expected in ((None, [1, 2]), (utc(before), [2]), (utc(after), [])):
        settled = [entry["fixture_id"] for entry in forecasts
                   if entry["settlement"] == "SETTLED" and (cutoff is None or entry["frozen_utc"] > cutoff)]
        assert [row["fixture_id"] for row in _settled_guard_rows(store, cutoff)] == settled == expected

