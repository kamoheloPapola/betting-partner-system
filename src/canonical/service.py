"""One transactional fetch/freeze/settle loop; never trains or promotes models."""
from datetime import datetime,timedelta,timezone
import math
import sqlite3

from .clock import CanonicalError,utc,digest
from .features import FeatureEngine
from .lifecycle import last_resume_utc
from .markets import probabilities,MARKET_VERSION
from .model import read_artifact
from .provider import LEAGUES,RealClock

GUARD_WINDOW=300
GUARD_MARGIN=.03
STALE_HOURS=36


def _model(store,reference):
    model=read_artifact(reference["artifact_path"],reference["artifact_sha256"])[0]
    if model.model_id!=reference["id"]:
        raise CanonicalError("MODEL_ID_REFERENCE_MISMATCH")
    return model


def _settled_guard_rows(store, resumed_at):
    return store.db.execute(
        """SELECT p.*, f.league, r.home_goals, r.away_goals
           FROM forecasts p JOIN fixtures f ON f.id = p.fixture_id
           JOIN latest_results r ON r.fixture_id = p.fixture_id
           WHERE f.status = 'FINISHED' AND f.kickoff_utc > p.frozen_utc
             AND (? IS NULL OR p.frozen_utc > ?)
           ORDER BY p.kickoff_utc DESC, p.fixture_id ASC
           LIMIT ?""",
        (resumed_at, resumed_at, GUARD_WINDOW),
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
            cache[mid]=_model(store,reference)
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


def run_once(store,client,*,clock=None,notifier=None,horizon_days=7):
    clock=clock or RealClock()
    started=utc(clock.utcnow())
    run_id=None
    try:
        run_id=store.start_run(started)
        with store.transaction():
            active=store.active_model()
            model=_model(store,active)
            if utc(active["registered_utc"])>started or utc(active["activated_utc"])>started:
                raise CanonicalError("MODEL_FROM_FUTURE")
            current=datetime.fromisoformat(started.replace("Z","+00:00"))
            season=current.year if current.month>=7 else current.year-1
            pairs=[(league,season) for league in LEAGUES]
            pairs.extend(sorted(set(store.pending_seasons())-set(pairs)))
            records,receipts=client.fetch(pairs)
            observed=utc(clock.utcnow())
            if observed<started:raise CanonicalError("UTC_CLOCK_MOVED_BACKWARDS")
            counts=store.ingest(records,observed)
            guard=guardrail(store)
            if guard["state"]=="HALT" and store.halted() is None:
                store.halt({"code":"LIVE_BUG_GUARD","observed_utc":observed,**guard})
            halted=store.halted()
            counts.update(forecasts_frozen=0,skipped_kickoff_during_run=0,
                          guardrail=guard,halted=bool(halted),request_receipts=receipts,
                          empty_response_batches=sum(not any(r["league"]==l and r["season"]==s for r in records) for l,s in pairs))
            if not halted:
                engine=FeatureEngine(model.seed,store.history(observed),asof=observed)
                end=current+timedelta(days=horizon_days)
                for fixture in store.upcoming(observed,utc(end)):
                    vector,provenance=engine.vector(fixture)
                    home,away=model.predict(vector)
                    # Validate publishable markets before committing an immutable forecast.
                    probabilities(home,away,version=MARKET_VERSION)
                    frozen=utc(clock.utcnow())
                    if frozen>=fixture["kickoff_utc"]:
                        counts["skipped_kickoff_during_run"]+=1
                        continue
                    if frozen<observed:raise CanonicalError("UTC_CLOCK_MOVED_BACKWARDS")
                    if store.freeze(fixture,frozen,home,away,model.model_id,MARKET_VERSION,
                                    provenance["feature_hash"],provenance["history_known_utc"]):
                        counts["forecasts_frozen"]+=1
            ended=utc(clock.utcnow())
            store.finish_run(run_id,ended,"SUCCESS",counts=counts)
        if counts["halted"]:
            notification=_safe_notify(notifier,"LIVE_BUG_GUARD")
            return {"state":"HALTED","exit_code":3,"run_id":run_id,"counts":counts,"notification":notification}
        return {"state":"SUCCESS","exit_code":0,"run_id":run_id,"counts":counts}
    except Exception as exc:
        code=exc.code if isinstance(exc,CanonicalError) else "DATABASE_BUSY" if isinstance(exc,sqlite3.OperationalError) and "locked" in str(exc).lower() else "RUN_FAILED_"+type(exc).__name__.upper()
        failure={"error_code":code,"request_receipts":getattr(client,"receipts",[])}
        if run_id is not None:
            try:store.finish_run(run_id,utc(clock.utcnow()),"FAILED",counts=failure,error_code=code)
            except Exception as logging_error:failure["failure_record_error_type"]=type(logging_error).__name__
        notification=_safe_notify(notifier,code)
        return {"state":"FAILED","exit_code":1,"run_id":run_id,**failure,"notification":notification}


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
    try:
        ref=store.active_model();model=_model(store,ref);model_id=model.model_id
    except (CanonicalError,OSError,ValueError,KeyError):
        reasons.append("MODEL_UNAVAILABLE_OR_UNVALIDATED");model_id=None
    cutoff=utc(datetime.fromisoformat(at.replace("Z","+00:00"))-timedelta(hours=72))
    missing=store.db.execute("""SELECT COUNT(*) FROM fixtures f JOIN forecasts p ON p.fixture_id=f.id
     WHERE f.kickoff_utc<? AND f.status NOT IN ('FINISHED','CANCELLED','AWARDED','POSTPONED')""",(cutoff,)).fetchone()[0]
    if missing:warnings.append("RESULTS_MISSING_OVER_72H")
    return {"state":"RED" if reasons else "YELLOW" if warnings else "GREEN",
            "reasons":reasons,"warnings":warnings,"last_success_utc":last,"last_run":run,
            "model_id":model_id,"halt":halted,"guardrail":guard_state,
            "missing_results":missing,"stale_after_hours":STALE_HOURS,
            "fixture_count":store.db.execute("SELECT COUNT(*) FROM fixtures").fetchone()[0],
            "forecast_count":store.db.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0]}

