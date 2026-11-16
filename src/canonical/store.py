"""Single SQLite authority: fixtures, immutable forecasts and append-only results."""
from contextlib import contextmanager
from pathlib import Path
import json
import math
import sqlite3
import time

from .clock import CanonicalError, canonical, digest, utc

SCHEMA = """
BEGIN IMMEDIATE;
CREATE TABLE fixtures(
 id INTEGER PRIMARY KEY CHECK(id>0), league TEXT NOT NULL, season INTEGER NOT NULL,
 home_id INTEGER NOT NULL CHECK(home_id>0), away_id INTEGER NOT NULL CHECK(away_id>0),
 home_name TEXT NOT NULL, away_name TEXT NOT NULL, kickoff_utc TEXT NOT NULL,
 status TEXT NOT NULL, observed_utc TEXT NOT NULL, payload_hash TEXT NOT NULL,
 CHECK(home_id<>away_id));
CREATE TABLE fixture_revisions(
 fixture_id INTEGER NOT NULL REFERENCES fixtures(id), revision INTEGER NOT NULL,
 observed_utc TEXT NOT NULL, metadata_json TEXT NOT NULL,
 PRIMARY KEY(fixture_id,revision));
CREATE TABLE models(
 id TEXT PRIMARY KEY, artifact_path TEXT NOT NULL, artifact_sha256 TEXT NOT NULL,
 registered_utc TEXT NOT NULL);
CREATE TABLE model_activations(
 sequence INTEGER PRIMARY KEY AUTOINCREMENT, model_id TEXT NOT NULL REFERENCES models(id),
 activated_utc TEXT NOT NULL, reason TEXT NOT NULL);
CREATE TRIGGER no_model_update BEFORE UPDATE ON models
 BEGIN SELECT RAISE(ABORT,'Model references are immutable'); END;
CREATE TRIGGER no_model_delete BEFORE DELETE ON models
 BEGIN SELECT RAISE(ABORT,'Model references are immutable'); END;
CREATE TRIGGER no_activation_update BEFORE UPDATE ON model_activations
 BEGIN SELECT RAISE(ABORT,'Model activations are append-only'); END;
CREATE TRIGGER no_activation_delete BEFORE DELETE ON model_activations
 BEGIN SELECT RAISE(ABORT,'Model activations are append-only'); END;
CREATE TABLE forecasts(
 fixture_id INTEGER PRIMARY KEY REFERENCES fixtures(id),
 frozen_utc TEXT NOT NULL, kickoff_utc TEXT NOT NULL,
 home_xg REAL NOT NULL CHECK(home_xg>0 AND home_xg<=1.7976931348623157e308),
 away_xg REAL NOT NULL CHECK(away_xg>0 AND away_xg<=1.7976931348623157e308),
 model_id TEXT NOT NULL REFERENCES models(id), market_version TEXT NOT NULL,
 feature_hash TEXT NOT NULL, history_known_utc TEXT,
 CHECK(frozen_utc<kickoff_utc),
 CHECK(history_known_utc IS NULL OR history_known_utc<=frozen_utc));
CREATE TRIGGER no_forecast_update BEFORE UPDATE ON forecasts
 BEGIN SELECT RAISE(ABORT,'Frozen forecast is immutable'); END;
CREATE TRIGGER no_forecast_delete BEFORE DELETE ON forecasts
 BEGIN SELECT RAISE(ABORT,'Frozen forecast is immutable'); END;
CREATE TRIGGER no_forecast_replace BEFORE INSERT ON forecasts
 WHEN EXISTS(SELECT 1 FROM forecasts WHERE fixture_id=NEW.fixture_id)
 BEGIN SELECT RAISE(ABORT,'Frozen forecast already exists'); END;
CREATE TRIGGER forecast_eligible BEFORE INSERT ON forecasts
 WHEN NOT EXISTS(SELECT 1 FROM fixtures WHERE id=NEW.fixture_id AND status='TIMED'
   AND kickoff_utc=NEW.kickoff_utc AND kickoff_utc>NEW.frozen_utc)
 BEGIN SELECT RAISE(ABORT,'Forecast requires a confirmed future fixture'); END;
CREATE TRIGGER forecast_model_eligible BEFORE INSERT ON forecasts
 WHEN NOT EXISTS(SELECT 1 FROM models m JOIN model_activations a ON a.model_id=m.id
   WHERE m.id=NEW.model_id AND m.registered_utc<=NEW.frozen_utc
   AND a.sequence=(SELECT MAX(sequence) FROM model_activations WHERE activated_utc<=NEW.frozen_utc))
 BEGIN SELECT RAISE(ABORT,'Forecast requires the model adopted before freeze'); END;
CREATE TABLE results(
 fixture_id INTEGER NOT NULL REFERENCES fixtures(id), revision INTEGER NOT NULL,
 home_goals INTEGER NOT NULL CHECK(typeof(home_goals)='integer' AND home_goals>=0),
 away_goals INTEGER NOT NULL CHECK(typeof(away_goals)='integer' AND away_goals>=0),
 observed_utc TEXT NOT NULL, kickoff_utc TEXT NOT NULL,
 home_id INTEGER NOT NULL, away_id INTEGER NOT NULL, payload_hash TEXT NOT NULL,
 PRIMARY KEY(fixture_id,revision));
CREATE TRIGGER no_result_update BEFORE UPDATE ON results
 BEGIN SELECT RAISE(ABORT,'Result revisions are append-only'); END;
CREATE TRIGGER no_result_delete BEFORE DELETE ON results
 BEGIN SELECT RAISE(ABORT,'Result revisions are append-only'); END;
CREATE TRIGGER no_result_replace BEFORE INSERT ON results
 WHEN EXISTS(SELECT 1 FROM results WHERE fixture_id=NEW.fixture_id AND revision=NEW.revision)
 BEGIN SELECT RAISE(ABORT,'Result revision already exists'); END;
CREATE TRIGGER result_finished_only BEFORE INSERT ON results
 WHEN NOT EXISTS(SELECT 1 FROM fixtures WHERE id=NEW.fixture_id AND status='FINISHED')
 BEGIN SELECT RAISE(ABORT,'Only FINISHED fixtures can be settled'); END;
CREATE VIEW latest_results AS SELECT r.* FROM results r
 WHERE r.revision=(SELECT MAX(x.revision) FROM results x WHERE x.fixture_id=r.fixture_id);
CREATE TABLE runs(
 id INTEGER PRIMARY KEY AUTOINCREMENT, started_utc TEXT NOT NULL, ended_utc TEXT,
 status TEXT NOT NULL, error_code TEXT, counts_json TEXT);
CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
CREATE TABLE operator_decisions(
 sequence INTEGER PRIMARY KEY AUTOINCREMENT, observed_utc TEXT NOT NULL,
 action TEXT NOT NULL, reason TEXT NOT NULL, prior_json TEXT NOT NULL);
CREATE TRIGGER no_decision_update BEFORE UPDATE ON operator_decisions
 BEGIN SELECT RAISE(ABORT,'Operator decisions are append-only'); END;
CREATE TRIGGER no_decision_delete BEFORE DELETE ON operator_decisions
 BEGIN SELECT RAISE(ABORT,'Operator decisions are append-only'); END;
CREATE TRIGGER no_fixture_revision_update BEFORE UPDATE ON fixture_revisions
 BEGIN SELECT RAISE(ABORT,'Fixture history is append-only'); END;
CREATE TRIGGER no_fixture_revision_delete BEFORE DELETE ON fixture_revisions
 BEGIN SELECT RAISE(ABORT,'Fixture history is append-only'); END;
CREATE INDEX result_observation ON results(observed_utc);
CREATE INDEX upcoming ON fixtures(status,kickoff_utc);
PRAGMA user_version=1;
COMMIT;
"""


class Store:
    def __init__(self, path, *, readonly=False):
        self.path = Path(path).resolve()
        if readonly:
            self.db = sqlite3.connect(self.path.as_uri()+"?mode=ro", uri=True, timeout=5, isolation_level=None)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.db = sqlite3.connect(self.path, timeout=5, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA recursive_triggers=ON")
        version = self.db.execute("PRAGMA user_version").fetchone()[0]
        if version == 0 and not readonly:
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.executescript(SCHEMA)
        elif version != 1:
            self.db.close()
            raise CanonicalError("UNSUPPORTED_DATABASE_SCHEMA")

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @contextmanager
    def transaction(self):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield self
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def start_run(self, at):
        return self.db.execute("INSERT INTO runs(started_utc,status) VALUES(?,'RUNNING')", (utc(at),)).lastrowid

    def finish_run(self, run_id, at, status, *, counts=None, error_code=None):
        if status not in ("SUCCESS", "FAILED"):
            raise CanonicalError("INVALID_RUN_STATE")
        self.db.execute("UPDATE runs SET ended_utc=?,status=?,error_code=?,counts_json=? WHERE id=?",
                        (utc(at), status, error_code, canonical(counts or {}), run_id))

    def register_model(self, model_id, path, sha256, at):
        existing = self.db.execute("SELECT * FROM models WHERE id=?", (model_id,)).fetchone()
        values = (model_id, str(Path(path).resolve()), sha256, utc(at))
        if existing:
            if (existing["artifact_path"], existing["artifact_sha256"]) != values[1:3]:
                raise CanonicalError("MODEL_ID_REBOUND")
            return
        self.db.execute("INSERT INTO models VALUES(?,?,?,?)", values)

    def activate_model(self, model_id, at, reason):
        if not self.db.execute("SELECT 1 FROM models WHERE id=?", (model_id,)).fetchone():
            raise CanonicalError("MODEL_NOT_REGISTERED")
        self.db.execute("INSERT INTO model_activations(model_id,activated_utc,reason) VALUES(?,?,?)",
                        (model_id, utc(at), reason))

    def active_model(self):
        row = self.db.execute("""SELECT m.*,a.activated_utc FROM models m JOIN model_activations a ON a.model_id=m.id
                              ORDER BY a.sequence DESC LIMIT 1""").fetchone()
        if row is None:
            raise CanonicalError("NO_VALIDATED_MODEL")
        return dict(row)

    def ingest(self, records, at):
        """Caller validates the complete response and holds one transaction across a run."""
        at = utc(at)
        counts = {"fixtures_changed": 0, "results_added": 0, "results_corrected": 0}
        for item in records:
            payload = {k:item[k] for k in ("id","league","season","home_id","away_id","home_name","away_name","kickoff_utc","status")}
            payload["kickoff_utc"] = utc(payload["kickoff_utc"])
            phash = digest(payload)
            old = self.db.execute("SELECT * FROM fixtures WHERE id=?", (item["id"],)).fetchone()
            if old:
                if (old["league"],old["season"]) != (item["league"],item["season"]):
                    raise CanonicalError("FIXTURE_COMPETITION_REBOUND")
                if (old["home_id"],old["away_id"]) != (item["home_id"],item["away_id"]):
                    protected = self.db.execute("SELECT 1 FROM forecasts WHERE fixture_id=? UNION SELECT 1 FROM results WHERE fixture_id=?", (item["id"],item["id"])).fetchone()
                    if protected:
                        raise CanonicalError("PROTECTED_FIXTURE_TEAMS_CHANGED")
            if not old or old["payload_hash"] != phash:
                values = tuple(payload[k] for k in ("id","league","season","home_id","away_id","home_name","away_name","kickoff_utc","status"))+(at,phash)
                self.db.execute("""INSERT INTO fixtures VALUES(?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET home_id=excluded.home_id,away_id=excluded.away_id,
                   home_name=excluded.home_name,away_name=excluded.away_name,kickoff_utc=excluded.kickoff_utc,
                   status=excluded.status,observed_utc=excluded.observed_utc,payload_hash=excluded.payload_hash""",values)
                rev = self.db.execute("SELECT COALESCE(MAX(revision),0)+1 FROM fixture_revisions WHERE fixture_id=?", (item["id"],)).fetchone()[0]
                self.db.execute("INSERT INTO fixture_revisions VALUES(?,?,?,?)", (item["id"],rev,at,canonical(payload)))
                counts["fixtures_changed"] += 1
            if item["status"] != "FINISHED":
                continue
            h,a = item.get("home_goals"),item.get("away_goals")
            if any(isinstance(x,bool) or not isinstance(x,int) or x<0 for x in (h,a)):
                raise CanonicalError("FINISHED_SCORE_MISSING_OR_INVALID")
            if payload["kickoff_utc"] >= at:
                raise CanonicalError("FINISHED_FIXTURE_IN_FUTURE")
            rhash = digest([h,a])
            previous = self.db.execute("SELECT * FROM latest_results WHERE fixture_id=?", (item["id"],)).fetchone()
            if previous and previous["payload_hash"] == rhash:
                continue
            rev = previous["revision"]+1 if previous else 1
            self.db.execute("INSERT INTO results VALUES(?,?,?,?,?,?,?,?,?)",
                            (item["id"],rev,h,a,at,payload["kickoff_utc"],item["home_id"],item["away_id"],rhash))
            counts["results_corrected" if previous else "results_added"] += 1
        return counts

    def freeze(self, fixture, at, home_xg, away_xg, model_id, market_version, feature_hash, history_known_utc):
        at = utc(at)
        for x in (home_xg, away_xg):
            if isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or x<=0:
                raise CanonicalError("INVALID_EXPECTED_GOALS")
        if self.db.execute("SELECT 1 FROM forecasts WHERE fixture_id=?", (fixture["id"],)).fetchone():
            return False
        self.db.execute("INSERT INTO forecasts VALUES(?,?,?,?,?,?,?,?,?)",
                        (fixture["id"],at,utc(fixture["kickoff_utc"]),home_xg,away_xg,model_id,
                         market_version,feature_hash,utc(history_known_utc) if history_known_utc else None))
        return True

    def history(self, before):
        """As-known result revision, never a correction observed after the forecast."""
        before=utc(before)
        rows=self.db.execute("""SELECT r.*,f.league FROM results r JOIN fixtures f ON f.id=r.fixture_id
          WHERE r.observed_utc<=? AND r.revision=(SELECT MAX(x.revision) FROM results x
          WHERE x.fixture_id=r.fixture_id AND x.observed_utc<=?)
          ORDER BY r.kickoff_utc,r.fixture_id""",(before,before)).fetchall()
        revisions=self.db.execute("""SELECT fixture_id,metadata_json FROM fixture_revisions r
          WHERE observed_utc<=? AND revision=(SELECT MAX(x.revision) FROM fixture_revisions x
          WHERE x.fixture_id=r.fixture_id AND x.observed_utc<=?)""",(before,before)).fetchall()
        known={r["fixture_id"]:json.loads(r["metadata_json"])["status"] for r in revisions}
        return [dict(r) for r in rows if known.get(r["fixture_id"])=="FINISHED"]

    def upcoming(self, at, end):
        return [dict(r) for r in self.db.execute("""SELECT * FROM fixtures f WHERE status='TIMED'
          AND kickoff_utc>? AND kickoff_utc<=? AND NOT EXISTS(SELECT 1 FROM forecasts p WHERE p.fixture_id=f.id)
          ORDER BY kickoff_utc,id""",(utc(at),utc(end)))]

    def forecasts(self, league=None, limit=200):
        sql="""SELECT p.*,f.league,f.home_name,f.away_name,f.home_id,f.away_id,
          f.status AS fixture_status,f.kickoff_utc AS current_kickoff_utc,
          r.home_goals,r.away_goals,r.revision AS result_revision,
          CASE WHEN f.status IN ('CANCELLED','AWARDED') THEN 'VOID'
               WHEN f.kickoff_utc<=p.frozen_utc THEN 'INVALID_RESCHEDULED_BEFORE_FREEZE'
               WHEN r.fixture_id IS NOT NULL AND f.status='FINISHED' THEN 'SETTLED'
               ELSE 'PENDING' END AS settlement
          FROM forecasts p JOIN fixtures f ON f.id=p.fixture_id
          LEFT JOIN latest_results r ON r.fixture_id=p.fixture_id"""
        args=[]
        if league:
            sql+=" WHERE f.league=?";args.append(league)
        sql+=" ORDER BY p.kickoff_utc DESC,p.fixture_id LIMIT ?";args.append(min(max(int(limit),1),10000))
        return [dict(r) for r in self.db.execute(sql,args)]

    def pending_seasons(self):
        return [(r[0],r[1]) for r in self.db.execute("""SELECT DISTINCT f.league,f.season FROM fixtures f
          JOIN forecasts p ON p.fixture_id=f.id WHERE f.status NOT IN ('FINISHED','CANCELLED','AWARDED')""")]

    def last_run(self):
        row=self.db.execute("SELECT * FROM runs ORDER BY id DESC LIMIT 1").fetchone()
        return dict(row) if row else None

    def last_success(self):
        row=self.db.execute("SELECT MAX(ended_utc) FROM runs WHERE status='SUCCESS'").fetchone()
        return row[0]

    def halted(self):
        row=self.db.execute("SELECT value FROM metadata WHERE key='halt'").fetchone()
        return json.loads(row[0]) if row else None

    def halt(self, value):
        self.db.execute("INSERT INTO metadata VALUES('halt',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (canonical(value),))

    def backup(self, destination):
        destination=Path(destination)
        destination.parent.mkdir(parents=True,exist_ok=True)
        try:
            with destination.open("xb"):pass
        except FileExistsError as exc:
            raise CanonicalError("BACKUP_DESTINATION_EXISTS") from exc
        deadline=time.monotonic()+15
        def progress(status,remaining,total):
            if time.monotonic()>deadline:raise CanonicalError("BACKUP_TIMEOUT_PARTIAL_RETAINED")
        with sqlite3.connect(destination) as target:
            self.db.backup(target,pages=128,progress=progress,sleep=.1)
            if target.execute("PRAGMA integrity_check").fetchone()[0]!="ok":
                raise CanonicalError("BACKUP_INTEGRITY_FAILED")
        return destination

