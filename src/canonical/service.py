"""Fetch and settle before freezing forecasts; never trains or promotes models."""
from collections import Counter
from datetime import date,datetime,timedelta,timezone
from functools import cached_property
import json
import logging
import math
from pathlib import Path
import sqlite3
from traceback import walk_tb

from . import heartbeat
from .clock import CanonicalError,canonical,utc,digest
from .features import FeatureEngine
from .lifecycle import last_resume_utc
from .markets import probabilities,MARKET_VERSION
from .model import read_artifact
from .provider import FootballData, HTTPSnapshotTransport, LEAGUES, STATE_UPDATING_STATUS, STATUSES, RealClock
from .team_mapping_consumer import TeamMappingConsumer, _has_sha256, _object, _read_pinned

GUARD_WINDOW=300
GUARD_MARGIN=.03
STALE_HOURS=36
FORECAST_HORIZON_HOURS = 48
FORECAST_ERROR_DETAIL_LIMIT = 20
# Allow three hours from kickoff for regulation, stoppages and the provider's result update.
RESULT_SETTLE_MARGIN = timedelta(hours=3)
# A waiver is a reviewed exception for a run within one day of the decision.
MAX_FORECAST_WAIVER_AGE = timedelta(hours=24)
NON_PLAYED_DISPOSITIONS = {
    "AWARDED": "excluded_awarded", "CANCELLED": "excluded_cancelled", "POSTPONED": "deferred_postponed",
}
COMPLETENESS_ERROR = "FORECAST_HISTORY_INCOMPLETE"
ROOT = Path(__file__).resolve().parents[2]
logger = logging.getLogger(__name__)


class MappingResolutionError(CanonicalError):
    pass


def _mapping_error_code(error):
    return error.code if isinstance(error, CanonicalError) else "TEAM_MAPPING_RUNTIME_" + type(error).__name__.upper()


def _record_run_bug(store, run_id, started, stage, error, *, fixture_id=None):
    """Persist every frame and chained exception type without formatting payloads."""
    lines = [f"{started} run_id={run_id} stage={stage}"]
    filename = "translation-errors.log"
    if fixture_id is not None:
        lines[0] += f" fixture_id={fixture_id}"
        filename = "forecast-errors.log"
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        lines.append("Traceback (most recent call last):")
        for frame, line in walk_tb(error.__traceback__):
            source = Path(frame.f_code.co_filename).name
            lines.append(f'  File "{source}", line {line}, in {frame.f_code.co_name}')
        lines.append(f"{type(error).__name__} [message omitted]")
        error = error.__cause__ if error.__cause__ is not None else error.__context__
    trace = "\n".join(lines) + "\n"
    try:
        with store.path.with_name(filename).open("a", encoding="utf-8") as stream:
            stream.write(trace)
    except OSError:
        logger.error("Could not write %s\n%s", filename, trace)


def _read_model(reference):
    model, _, validation = read_artifact(reference["artifact_path"], reference["artifact_sha256"])
    if model.model_id!=reference["id"]:
        raise CanonicalError("MODEL_ID_REFERENCE_MISMATCH")
    return model, validation


def _settled_guard_rows(store, resumed_at):
    return store.db.execute(
        """SELECT p.*, f.league, r.home_goals, r.away_goals
           FROM forecasts p JOIN fixtures f ON f.id = p.fixture_id
           JOIN latest_results r ON r.fixture_id = p.fixture_id
           WHERE f.status = ? AND f.kickoff_utc > p.frozen_utc
             AND (? IS NULL OR p.frozen_utc > ?)
           ORDER BY p.kickoff_utc DESC, p.fixture_id ASC
           LIMIT ?""",
        (STATE_UPDATING_STATUS, resumed_at, resumed_at, GUARD_WINDOW),
    ).fetchall()


def guardrail(store):
    """Bug guard only; not a statistical model-promotion test."""
    resumed_at = last_resume_utc(store)
    settled = _settled_guard_rows(store, resumed_at)
    if len(settled)<GUARD_WINDOW:
        return {"state":"COLLECTING","n":len(settled),"required":GUARD_WINDOW,
                "window_started_after_utc":resumed_at}
    cache={};model_loss=0.;baseline_loss=0.;leagues={}
    for row in settled:
        mid=row["model_id"]
        if mid not in cache:
            reference=store.db.execute("SELECT * FROM models WHERE id=?",(mid,)).fetchone()
            if reference is None:raise CanonicalError("FROZEN_MODEL_REFERENCE_MISSING")
            cache[mid]=_read_model(reference)[0]
        baseline=cache[mid].payload["baseline_1x2"].get(row["league"])
        if baseline is None:raise CanonicalError("FROZEN_BASELINE_MISSING")
        values=probabilities(row["home_xg"],row["away_xg"],version=row["market_version"])
        actual=0 if row["home_goals"]>row["away_goals"] else 2 if row["home_goals"]<row["away_goals"] else 1
        p=[values["home_win"],values["draw"],values["away_win"]][actual]
        model_loss-=math.log(max(p,1e-15))
        baseline_loss-=math.log(max(baseline[actual],1e-15))
        leagues[row["league"]]=leagues.get(row["league"],0)+1
    delta=(model_loss-baseline_loss)/GUARD_WINDOW
    return {"state":"HALT" if delta>GUARD_MARGIN else "WITHIN_BUG_GUARD",
            "n":GUARD_WINDOW,"model_log_loss":model_loss/GUARD_WINDOW,
            "baseline_log_loss":baseline_loss/GUARD_WINDOW,"delta":delta,
            "margin":GUARD_MARGIN,"per_league_n":leagues,
            "window_order":"original frozen kickoff descending, provider ID ascending",
            "window_started_after_utc":resumed_at,
            "interpretation":"Code-bug guard; not a high-precision quality test"}


def _safe_notify(notifier,code):
    if notifier is None:return {"attempted":False}
    try:return notifier(code)
    except Exception as exc:
        return {"attempted":True,"delivered":False,"error_type":type(exc).__name__}


def _rejection_summary(status):
    return {
        "state": "DEGRADED" if status["rejected"] else "READY",
        "resolved": status["resolved"],
        "rejected": status["rejected"],
        "rejection_reasons": status["rejection_reasons"],
    }


def _mapping_summary(upcoming, history, withheld_leagues):
    status = _rejection_summary(upcoming)
    status["history"] = _rejection_summary(history)
    status["withheld_leagues"] = withheld_leagues
    if status["history"]["rejected"] or withheld_leagues:
        status["state"] = "DEGRADED"
    return status


def _season_start(day):
    return day.year - (day.month < 7)


def _history_cutoff(seed):
    last_day = seed.get("last_day")
    return date.fromisoformat(last_day) + timedelta(days=1) if last_day else None


def _valid_result(row):
    return all(type(row.get(key)) is int and row[key] >= 0 for key in ("home_goals", "away_goals"))


class _ObservedResponse:
    def __init__(self, response, metadata, max_body_bytes):
        self.status, self.headers = response.status, response.headers
        self._response, self._metadata, self._max_body_bytes = response, metadata, max_body_bytes

    @cached_property
    def body(self):
        # Observe bodies the fetch controller actually reads, including after a retry.
        raw = self._response.body
        if len(raw) > self._max_body_bytes:
            return raw
        try:
            matches = json.loads(raw).get("matches")
        except (ValueError, AttributeError, RecursionError):
            return raw  # The provider retains responsibility for validating the payload.
        if isinstance(matches, list):
            for row in matches:
                if isinstance(row, dict) and type(row.get("id")) is int:
                    metadata = {}
                    for name in ("utcDate", "lastUpdated"):
                        value = row.get(name)
                        if isinstance(value, str) and len(value) <= 40:
                            try:
                                utc(value)
                            except (CanonicalError, OverflowError):
                                continue
                            metadata[name] = value
                    self._metadata[row["id"]] = metadata
        return raw


class _CompletenessTransport:
    def __init__(self, transport, metadata):
        self._transport, self._metadata = transport, metadata

    def request(self, path, key, timeout, max_body_bytes):
        response = self._transport.request(path, key, timeout, max_body_bytes)
        if response.status == 200:
            return _ObservedResponse(response, self._metadata, max_body_bytes)
        return response

    def abort(self):
        return self._transport.abort()


def _fetch_with_fixture_metadata(client, pairs):
    metadata = {}
    if not isinstance(client, FootballData):
        records, receipts = client.fetch(pairs)
        return records, receipts, metadata
    original = client.transport
    client.transport = _CompletenessTransport(original or HTTPSnapshotTransport(allowed_pairs=pairs), metadata)
    try:
        records, receipts = client.fetch(pairs)
    finally:
        client.transport = original
    return records, receipts, metadata


def _load_forecast_waiver(reference):
    path, sha256 = reference
    code = "FORECAST_WAIVER_INVALID"
    if not _has_sha256(sha256):
        raise CanonicalError(code)
    waiver = _object(_read_pinned(path, sha256, "FORECAST_WAIVER"), code)
    fields = {"schema_version", "status", "league", "season", "fixture_id", "kickoff_utc",
              "provider_status", "approved_by", "reason", "approved_utc", "expires_utc"}
    if (waiver.keys() != fields or type(waiver["schema_version"]) is not int or waiver["schema_version"] != 1
            or waiver["status"] != "APPROVED" or waiver["league"] not in LEAGUES
            or type(waiver["season"]) is not int or type(waiver["fixture_id"]) is not int
            or waiver["fixture_id"] <= 0 or not isinstance(waiver["provider_status"], str)
            or waiver["provider_status"] not in STATUSES or waiver["provider_status"] == STATE_UPDATING_STATUS
            or waiver["provider_status"] in NON_PLAYED_DISPOSITIONS
            or not all(isinstance(waiver[key], str) and waiver[key].strip() for key in ("reason", "approved_by"))):
        raise CanonicalError(code)
    try:
        for key in ("kickoff_utc", "approved_utc", "expires_utc"):
            waiver[key] = utc(waiver[key])
        issued = datetime.fromisoformat(waiver["approved_utc"])
        expires = datetime.fromisoformat(waiver["expires_utc"])
    except CanonicalError as exc:
        raise CanonicalError(code) from exc
    if not timedelta(0) < expires - issued <= MAX_FORECAST_WAIVER_AGE:
        raise CanonicalError(code)
    return {**waiver, "sha256": sha256}


def _forecast_completeness(store, records, receipts, pairs, cutoff, *, started, observed, history, metadata, waiver=None):
    """Compare due fixtures from each fetch with the results the forecast will use."""
    integrity, unresolved, fetch_times, deadlines = [], [], {}, {}
    dispositions = Counter()
    non_played = []
    used_waiver = None
    pair_counts = Counter((row["league"], row["season"]) for row in records)
    ids = Counter(row["id"] for row in records)
    applied = {row["id"]: row for row in history}
    for pair in pairs:
        league, season = pair
        matches = [r for r in receipts if r["http_status"] == 200 and (r["league"], r["season"]) == pair]
        if len(matches) != 1:
            integrity.append({"reason": "COMPETITION_PAYLOAD_MISSING_OR_DUPLICATE", "league": league, "season": season})
            continue
        capture = matches[0]
        if not pair_counts[pair]:
            integrity.append({"reason": "COMPETITION_PAYLOAD_EMPTY", "league": league, "season": season})
        if capture.get("fixture_rejections"):
            integrity.append({"reason": "COMPETITION_FIXTURE_REJECTED", "league": league, "season": season,
                              "fixture_ids": sorted(r["id"] for r in capture["fixture_rejections"])})
        try:
            fetched = utc(capture["completed_utc"])
        except (CanonicalError, KeyError):
            fetched = ""
        if not started <= fetched <= observed:
            integrity.append({"reason": "COMPETITION_FETCH_TIME_INVALID", "league": league, "season": season})
        else:
            fetch_times[pair] = fetched
            deadlines[pair] = utc(datetime.fromisoformat(fetched) - RESULT_SETTLE_MARGIN)
    if cutoff is None:
        integrity.append({"reason": "SEED_CUTOFF_MISSING"})
    cutoff_utc = cutoff.isoformat() + "T00:00:00.000000Z" if cutoff else ""
    previously_finished = {row[0] for row in store.db.execute(
        """SELECT DISTINCT r.fixture_id FROM results r JOIN fixtures f ON f.id = r.fixture_id
           WHERE r.kickoff_utc >= ? OR f.kickoff_utc >= ?""", (cutoff_utc, cutoff_utc)
    )}
    for row in store.db.execute("SELECT id, league, season, kickoff_utc FROM fixtures WHERE kickoff_utc >= ?",
                                (cutoff_utc,)):
        pair = row["league"], row["season"]
        deadline = deadlines.get(pair)
        if deadline and row["kickoff_utc"] <= deadline and row["id"] not in ids:
            unresolved.append({"reason": "KNOWN_DUE_FIXTURE_MISSING", "fixture_id": row["id"],
                               "league": pair[0], "season": pair[1], "kickoff_utc": row["kickoff_utc"],
                               "status": None, "utcDate": None, "lastUpdated": None,
                               "fetch_completed_utc": fetch_times[pair]})
    integrity.extend({"reason": "DUPLICATE_FIXTURE_ID", "fixture_id": fixture_id}
                     for fixture_id, count in ids.items() if count > 1)
    for row in records:
        fixture_id, status = row["id"], row["status"]
        pair = row["league"], row["season"]
        deadline = deadlines.get(pair)
        kickoff = utc(row["kickoff_utc"])
        if ids[fixture_id] > 1:
            continue
        if status not in STATUSES:
            integrity.append({"reason": "UNKNOWN_FIXTURE_STATUS", "fixture_id": fixture_id})
        elif status != STATE_UPDATING_STATUS and fixture_id in previously_finished:
            integrity.append({"reason": "FINISHED_STATUS_REGRESSION", "fixture_id": fixture_id, "status": status})
        elif status == STATE_UPDATING_STATUS and not _valid_result(row):
            integrity.append({"reason": "FINISHED_SCORE_MISSING_OR_INVALID", "fixture_id": fixture_id})
        elif deadline and cutoff_utc <= kickoff <= deadline:
            dispositions["due"] += 1
            if status == STATE_UPDATING_STATUS:
                result = applied.get(fixture_id, {})
                fields = ("league", "home_id", "away_id", "home_goals", "away_goals")
                if (result.get("kickoff_utc") == kickoff
                        and all(result.get(key) == row[key] for key in fields)):
                    dispositions["applied_finished"] += 1
                    continue
                reason = "FINISHED_RESULT_NOT_APPLIED"
            elif status in NON_PLAYED_DISPOSITIONS:
                disposition = NON_PLAYED_DISPOSITIONS[status]
                dispositions[disposition] += 1
                non_played.append({"fixture_id": fixture_id, "league": row["league"], "season": row["season"],
                                   "status": status, "disposition": disposition})
                continue
            elif (waiver and (waiver["league"], waiver["season"]) == pair
                  and waiver["fixture_id"] == fixture_id and waiver["provider_status"] == status
                  and waiver["kickoff_utc"] == kickoff
                  and waiver["approved_utc"] <= fetch_times[pair] < waiver["expires_utc"]):
                used_waiver = {**waiver, "fetch_completed_utc": fetch_times[pair]}
                continue
            else:
                reason = "DUE_FIXTURE_UNRESOLVED"
            raw = metadata.get(fixture_id, {})
            unresolved.append({"reason": reason, "fixture_id": fixture_id, "status": status, "league": pair[0],
                               "season": pair[1], "kickoff_utc": kickoff, "utcDate": raw.get("utcDate"),
                               "lastUpdated": raw.get("lastUpdated"), "fetch_completed_utc": fetch_times[pair]})
    withheld = set(LEAGUES) if integrity else {issue["league"] for issue in unresolved}
    state = "BLOCKED" if len(withheld) == len(LEAGUES) else "PARTIAL" if withheld else "READY"
    return {"state": state, "cutoff_utc": cutoff_utc or None,
            "settle_margin_minutes": int(RESULT_SETTLE_MARGIN.total_seconds() // 60),
            **{name: dispositions[name] for name in ("due", "applied_finished", *NON_PLAYED_DISPOSITIONS.values())},
            "non_played_fixtures": sorted(non_played, key=lambda row: row["fixture_id"]),
            "waived_due": int(used_waiver is not None),
            "waiver": used_waiver,
            "global_block": bool(integrity),
            "withheld_leagues": sorted(withheld),
            "issues": sorted([*integrity, *unresolved], key=lambda issue: (issue["reason"], issue.get("league", ""),
                                                                         issue.get("season", 0), issue.get("fixture_id", 0)))}


def _feature_inputs(store, model, mapping, *, started, observed, end):
    # Rescheduling preserves the existing frozen forecast for that fixture ID.
    fixtures = store.upcoming(started, end)
    # The seed already covers its cutoff day. The provider can return that
    # history again on the first run or after a subsequent model change.
    cutoff = model.seed.get("last_day") or ""
    history = [dict(row, id=row["fixture_id"]) for row in store.history(observed)
               if row["kickoff_utc"][:10] > cutoff]
    try:
        upcoming = mapping.resolve(fixtures)
        known = mapping.resolve(history)
        withheld = {}
        for row, decision in zip(history, known["fixtures"], strict=True):
            if "reason" in decision:
                withheld.setdefault(row["league"], set()).add(decision["reason"])

        # Do not build or serve a partial result history for a withheld league.
        states = {}
        accepted = []
        for rows, batch in ((fixtures, upcoming), (history, known)):
            accepted_rows = []
            for row, decision in zip(rows, batch["fixtures"], strict=True):
                if "reason" in decision or row["league"] in withheld:
                    continue
                accepted_rows.append(row)
                for side in ("home", "away"):
                    team = decision[side]
                    states[row["league"], team["provider_id"]] = team["state"]
            accepted.append(accepted_rows)
        status = _mapping_summary(upcoming["status"], known["status"],
                                  {league: sorted(withheld[league]) for league in sorted(withheld)})
    except Exception as error:
        raise MappingResolutionError(_mapping_error_code(error)) from error

    seed = {**model.seed, "states": [
        {"league": league, "team_id": team_id, "state": state}
        for (league, team_id), state in sorted(states.items())
    ]}
    eligible_fixtures, eligible_history = accepted
    return eligible_fixtures, FeatureEngine(seed, eligible_history, asof=observed), status, eligible_history


def _forecast_fixtures(store, fixtures, engine, model, clock, forecast_asof, counts, run_id, started):
    written = Counter()
    for fixture in fixtures:
        store.db.execute("SAVEPOINT forecast_fixture")
        try:
            vector, provenance = engine.vector(fixture)
            home, away = model.predict(vector)
            # Validate publishable markets before committing an immutable forecast.
            probabilities(home, away, version=MARKET_VERSION)
            frozen = utc(clock.utcnow())
            if frozen >= fixture["kickoff_utc"]:
                counts["skipped_kickoff_during_run"] += 1
            elif frozen < forecast_asof:
                raise CanonicalError("UTC_CLOCK_MOVED_BACKWARDS")
            elif store.freeze(fixture, frozen, home, away, model.model_id, MARKET_VERSION,
                              provenance["feature_hash"], provenance["history_known_utc"],
                              run_id=run_id, waiver_sha256=counts["waiver_sha256"]):
                counts["forecasts_frozen"] += 1
                written[fixture["league"]] += 1
        except (sqlite3.Error, OSError):
            # Storage failures invalidate the forecast batch. SQLite may already have aborted it.
            raise
        except Exception as error:
            code = error.code if isinstance(error, CanonicalError) else "FORECAST_FAILED_" + type(error).__name__.upper()
            if code == "UTC_CLOCK_MOVED_BACKWARDS":
                raise
            store.db.execute("ROLLBACK TO forecast_fixture")
            counts["forecast_errors"] += 1
            details = counts["forecast_error_details"]
            if len(details) < FORECAST_ERROR_DETAIL_LIMIT:
                details.append({"fixture_id": fixture["id"], "error_code": code})
            _record_run_bug(store, run_id, started, "forecast", error, fixture_id=fixture["id"])
        store.db.execute("RELEASE forecast_fixture")
    return written


def _league_forecast_receipt(coverage, fixtures, written):
    leagues = {}
    for league in sorted(LEAGUES):
        withheld = league in coverage["withheld_leagues"]
        issues = [issue for issue in coverage["issues"] if coverage["global_block"] or issue.get("league") == league]
        fixture_ids = {fixture_id for issue in issues for fixture_id in issue.get("fixture_ids", ())}
        fixture_ids.update(issue["fixture_id"] for issue in issues if "fixture_id" in issue)
        leagues[league] = {
            "state": "WITHHELD" if withheld else "WRITTEN" if written[league] else "NO_OP",
            "forecasts_written": written[league],
            "withhold_fixture_ids": sorted(fixture_ids) if withheld else [],
            "withheld_forecasts": [{"fixture_id": row["id"], "kickoff_utc": row["kickoff_utc"]}
                                   for row in fixtures if withheld and row["league"] == league],
        }
    return leagues


def run_once(store,client,*,clock=None,notifier=None,fixture_waiver=None,heartbeat_transport=None,heartbeat_attempt=None):
    attempt = heartbeat.Attempt(heartbeat_transport) if heartbeat_attempt is None else heartbeat_attempt
    run_id=None
    counts = {}
    settled_counts = None
    try:
        clock=clock or RealClock()
        started=utc(clock.utcnow())
        run_id=store.start_run(started)
        attempt.run_id = run_id
        with store.transaction():
            active=store.active_model()
            model, validation = _read_model(active)
            if utc(active["registered_utc"])>started or utc(active["activated_utc"])>started:
                raise CanonicalError("MODEL_FROM_FUTURE")
            current=datetime.fromisoformat(started.replace("Z","+00:00"))
            season = _season_start(current)
            cutoff = _history_cutoff(model.seed)
            first_season = min(season, _season_start(cutoff)) if cutoff else season
            required_pairs = [(league, year) for year in range(first_season, season + 1) for league in LEAGUES]
            pairs = [*required_pairs, *sorted(set(store.pending_seasons()) - set(required_pairs))]
            records, receipts, metadata = _fetch_with_fixture_metadata(client, pairs)
            observed=utc(clock.utcnow())
            if observed<started:raise CanonicalError("UTC_CLOCK_MOVED_BACKWARDS")
            ids = Counter(row["id"] for row in records)
            # The provider normally rejects corrupt batches before returning them. Preserve unambiguous
            # results if a client returns row-level defects and let the forecast gate report the defects.
            counts=store.ingest((row for row in records if ids[row["id"]] == 1 and row["status"] in STATUSES
                                 and (row["status"] != STATE_UPDATING_STATUS or _valid_result(row))), observed)
            counts.update(model_id=model.model_id, run_id=run_id, waiver_sha256=None,
                          forecasts_frozen=0,skipped_kickoff_during_run=0,
                          forecast_errors=0, forecast_error_details=[],
                          request_receipts=receipts,
                          empty_response_batches=sum(not any(r["league"]==l and r["season"]==s for r in records) for l,s in pairs))
            rejected_fixtures = sum(len(receipt.get("fixture_rejections", ())) for receipt in receipts)
            if rejected_fixtures:
                counts["invalid_provider_fixtures"] = rejected_fixtures

        # Official results must survive a broken forecast batch, even if SQLite rolls that batch back.
        settled_counts = counts.copy()
        mapping_error = None
        forecast_error = None
        with store.transaction():
            if store.active_model() != active:
                raise CanonicalError("ACTIVE_MODEL_CHANGED_DURING_RUN")
            # Another run may have committed fixtures while this run was between transactions.
            forecast_asof = utc(clock.utcnow())
            if forecast_asof < observed:
                raise CanonicalError("UTC_CLOCK_MOVED_BACKWARDS")
            try:
                mapping = TeamMappingConsumer.for_model(ROOT, model, validation)
            except Exception as error:
                mapping = None
                mapping_error = _mapping_error_code(error)
                if not isinstance(error, CanonicalError):
                    _record_run_bug(store, run_id, started, "load", error)
            if mapping is not None:
                guard=guardrail(store)
                if guard["state"]=="HALT" and store.halted() is None:
                    store.halt({"code":"LIVE_BUG_GUARD","observed_utc":forecast_asof,**guard})
                counts["guardrail"] = guard
                if not store.halted():
                    try:
                        fixtures, engine, counts["team_mapping"], history = _feature_inputs(
                            store, model, mapping, started=started, observed=forecast_asof,
                            end=utc(current + timedelta(hours=FORECAST_HORIZON_HOURS)),
                        )
                    except MappingResolutionError as error:
                        mapping_error = error.code
                        if not isinstance(error.__cause__, CanonicalError):
                            _record_run_bug(store, run_id, started, "resolve", error)
                    else:
                        waiver = _load_forecast_waiver(fixture_waiver) if fixture_waiver is not None else None
                        coverage = _forecast_completeness(store, records, receipts, pairs, cutoff,
                                                          started=started, observed=observed, history=history,
                                                          metadata=metadata, waiver=waiver)
                        counts["forecast_completeness"] = coverage
                        if coverage["waiver"] is not None:
                            # Debt: mark the entire run conservatively; evaluation can narrow by league.
                            counts["waiver_sha256"] = coverage["waiver"]["sha256"]
                        withheld = set(coverage["withheld_leagues"])
                        if withheld:
                            forecast_error = COMPLETENESS_ERROR
                        eligible = [row for row in fixtures if row["league"] not in withheld]
                        if withheld and eligible:
                            engine = FeatureEngine(
                                {**engine.seed, "states": [row for row in engine.seed["states"]
                                                            if row["league"] not in withheld]},
                                [row for row in history if row["league"] not in withheld], asof=forecast_asof)
                        written = _forecast_fixtures(store, eligible, engine, model, clock, forecast_asof,
                                                     counts, run_id, started)
                        coverage["leagues"] = _league_forecast_receipt(coverage, fixtures, written)
                        if not withheld and eligible and counts["forecast_errors"] == len(eligible):
                            forecast_error = "ALL_FORECASTS_FAILED"
            counts["halted"] = bool(store.halted())
            ended=utc(clock.utcnow())
            if mapping_error:
                counts["team_mapping"] = {"state": "BLOCKED", "error_code": mapping_error}
            run_error = mapping_error or forecast_error
            store.finish_run(run_id,ended,"FAILED" if run_error else "SUCCESS",
                             counts=counts,error_code=run_error)
        if run_error:
            alert = run_error
            if run_error == COMPLETENESS_ERROR:
                alert = "_".join([run_error, *counts["forecast_completeness"]["withheld_leagues"]])
            result = {"state": "PARTIAL_FORECASTS" if counts["forecasts_frozen"] else "SETTLEMENT_ONLY",
                      "exit_code": 1, "run_id": run_id,
                      "error_code": run_error, "counts": counts,
                      "notification": _safe_notify(notifier, alert)}
        elif counts["halted"]:
            result = {"state": "HALTED", "exit_code": 3, "run_id": run_id, "counts": counts,
                      "notification": _safe_notify(notifier, "LIVE_BUG_GUARD")}
        else:
            result = {"state": "SUCCESS", "exit_code": 0, "run_id": run_id, "counts": counts}
            if counts["forecast_errors"]:
                result["notification"] = _safe_notify(notifier, "FORECAST_ERRORS")
    except Exception as exc:
        code=exc.code if isinstance(exc,CanonicalError) else "DATABASE_BUSY" if isinstance(exc,sqlite3.OperationalError) and "locked" in str(exc).lower() else "RUN_FAILED_"+type(exc).__name__.upper()
        failure={"error_code":code,"request_receipts":getattr(client,"receipts",[])}
        state = "FAILED"
        if settled_counts is not None:
            state = "SETTLEMENT_ONLY"
            settled_counts.update(forecast_errors=counts["forecast_errors"],
                                  forecast_error_details=counts["forecast_error_details"])
            failure["counts"] = settled_counts
        if run_id is not None:
            try:store.finish_run(run_id,utc(clock.utcnow()),"FAILED",counts=settled_counts or failure,error_code=code)
            except Exception as logging_error:failure["failure_record_error_type"]=type(logging_error).__name__
        notification=_safe_notify(notifier,code)
        result = {"state":state,"exit_code":1,"run_id":run_id,**failure,"notification":notification}

    coverage = counts.get("forecast_completeness", {})
    mapping = counts.get("team_mapping", {})
    withheld = sorted(set(coverage.get("withheld_leagues", ())) | set(mapping.get("withheld_leagues", ())))
    delivery = attempt.report(result["state"], withheld, blocked=coverage.get("global_block", False))
    result["heartbeat"] = delivery
    if "counts" in result:
        result["counts"]["heartbeat"] = delivery
    if run_id is not None:
        try:
            # Enrich the committed receipt without reopening the forecasting transaction.
            store.db.execute("UPDATE runs SET counts_json=json_set(counts_json,'$.heartbeat',json(?)) "
                             "WHERE id=? AND counts_json IS NOT NULL",
                             (canonical(delivery), run_id))
        except Exception as exc:
            result["heartbeat_record_error_type"] = type(exc).__name__
    return result


def health(store,*,at=None):
    at=utc(at or datetime.now(timezone.utc))
    last=store.last_success();run=store.last_run();reasons=[];warnings=[]
    if last is None:reasons.append("NO_SUCCESSFUL_RUN")
    else:
        age=(datetime.fromisoformat(at.replace("Z","+00:00"))-datetime.fromisoformat(last.replace("Z","+00:00"))).total_seconds()
        if age<0:reasons.append("UTC_CLOCK_ANOMALY")
        elif age>STALE_HOURS*3600:reasons.append("STALE_DATA")
    if run and run["status"]=="FAILED":reasons.append(run["error_code"] or "LAST_RUN_FAILED")
    if run and run["status"]=="RUNNING":
        age=(datetime.fromisoformat(at.replace("Z","+00:00"))-datetime.fromisoformat(run["started_utc"].replace("Z","+00:00"))).total_seconds()
        if age>15*60:reasons.append("RUN_DID_NOT_FINISH_WITHIN_BOUND")
        else:warnings.append("RUN_IN_PROGRESS")
    halted=store.halted()
    if halted:reasons.append("LIVE_BUG_GUARD_HALT")
    resumed_at = last_resume_utc(store)
    guard_state = None
    if halted:
        guard_state = {"state": "HALTED"}
    elif resumed_at is not None:
        count = len(_settled_guard_rows(store, resumed_at))
        rearming = count < GUARD_WINDOW
        if rearming:
            warnings.append("GUARD_REARMING")
        guard_state = {"state": "GUARD_REARMING" if rearming else "ARMED",
                       "n": count, "required": GUARD_WINDOW,
                       "window_started_after_utc": resumed_at}
    mapping_status = {"state": "NOT_EVALUATED"}
    missed_forecast_count = None
    counts = json.loads(run["counts_json"]) if run and run["counts_json"] else {}
    heartbeat_status = heartbeat.status(counts.get("heartbeat"))
    if heartbeat_status["state"] in heartbeat.WARNINGS:
        warnings.append(heartbeat.WARNINGS[heartbeat_status["state"]])
    invalid_provider_fixtures = counts.get("invalid_provider_fixtures", 0)
    forecast_errors = counts.get("forecast_errors", 0)
    if forecast_errors:
        warnings.append("FORECAST_ERRORS")
    if invalid_provider_fixtures:
        warnings.append("INVALID_PROVIDER_FIXTURE")
    if counts.get("forecast_completeness", {}).get("waiver"):
        warnings.append("FORECAST_HISTORY_WAIVER_USED")
    try:
        active = store.active_model()
        model, validation = _read_model(active)
        model_id = model.model_id
    except (CanonicalError,OSError,ValueError,KeyError):
        reasons.append("MODEL_UNAVAILABLE_OR_UNVALIDATED");model_id=None
    else:
        try:
            mapping = TeamMappingConsumer.for_model(ROOT, model, validation)
        except Exception as error:
            code = _mapping_error_code(error)
            mapping_status = {"state": "BLOCKED", "error_code": code}
            if code not in reasons:
                reasons.append(code)
        else:
            # Ignore the historical season backfill. A confirmed kickoff can pass between refreshes.
            past_unforecast = [dict(row) for row in store.db.execute("""
                SELECT f.id, f.league, f.home_id, f.away_id, f.kickoff_utc
                FROM fixtures f
                WHERE f.kickoff_utc > ? AND f.kickoff_utc <= ?
                  AND f.status IN ('TIMED', 'IN_PLAY', 'PAUSED', 'SUSPENDED', 'FINISHED')
                  AND NOT EXISTS (SELECT 1 FROM forecasts p WHERE p.fixture_id = f.id)
                """, (active["activated_utc"], at))]
            try:
                missed_forecast_count = mapping.resolve(past_unforecast)["status"]["resolved"]
            except Exception:
                # An unavailable mapping cannot establish which fixtures were eligible.
                pass
            if counts.get("model_id") == model_id and "team_mapping" in counts:
                saved = counts["team_mapping"]
                mapping_status = saved
                if saved["state"] != "BLOCKED":
                    mapping_status = _mapping_summary(saved, saved["history"], saved.get("withheld_leagues", {}))
                    if mapping_status["rejected"]:
                        warnings.append("TEAM_MAPPING_FIXTURE_REJECTIONS")
                    if mapping_status["history"]["rejected"] or mapping_status["withheld_leagues"]:
                        warnings.append("TEAM_MAPPING_HISTORY_REJECTIONS")
    if missed_forecast_count:
        warnings.append("MISSED_FORECASTS")
    cutoff=utc(datetime.fromisoformat(at.replace("Z","+00:00"))-timedelta(hours=72))
    missing=store.db.execute("""SELECT COUNT(*) FROM fixtures f JOIN forecasts p ON p.fixture_id=f.id
     WHERE f.kickoff_utc<? AND f.status NOT IN ('FINISHED','CANCELLED','AWARDED','POSTPONED')""",(cutoff,)).fetchone()[0]
    if missing:warnings.append("RESULTS_MISSING_OVER_72H")
    return {"state":"RED" if reasons else "YELLOW" if warnings else "GREEN",
            "reasons":reasons,"warnings":warnings,"last_success_utc":last,"last_run":run,
            "heartbeat":heartbeat_status,
            "model_id":model_id,"halt":halted,"guardrail":guard_state,"team_mapping":mapping_status,
            "forecast_completeness":counts.get("forecast_completeness", {"state": "NOT_EVALUATED"}),
            "missing_results":missing,"missed_forecast_count":missed_forecast_count,
            "invalid_provider_fixtures":invalid_provider_fixtures,"forecast_errors":forecast_errors,"stale_after_hours":STALE_HOURS,
            "fixture_count":store.db.execute("SELECT COUNT(*) FROM fixtures").fetchone()[0],
            "forecast_count":store.db.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0]}

