"""football-data.org v4 only: bounded, paced requests and minimal match-data projection."""
from datetime import datetime
import json
import re

from src.ingest.provider_snapshots import (
    HTTPSnapshotTransport, RealClock, bounded_request, delay_seconds, rate_wait,
    check_secret, check_json_secret, selected_headers, SecretReflection, RequestTimeout, ControllerTimeout,
)
from .clock import CanonicalError, utc

LEAGUES=("PL","BL1","FL1","SA","PD")
STATUSES=frozenset(("SCHEDULED","TIMED","IN_PLAY","PAUSED","FINISHED","SUSPENDED","POSTPONED","CANCELLED","AWARDED"))
MAX_BODY=8_000_000
PACE=7.0
DEADLINE=600.0


def _id(value):
    if isinstance(value,bool) or not isinstance(value,int) or value<=0:
        raise CanonicalError("INVALID_PROVIDER_ID")
    return value


def project_matches(payload,league,season,observed):
    if not isinstance(payload,dict) or not isinstance(payload.get("matches"),list):
        raise CanonicalError("MALFORMED_PROVIDER_ROOT")
    competition=payload.get("competition")
    if not isinstance(competition,dict) or competition.get("code")!=league:
        raise CanonicalError("PROVIDER_COMPETITION_MISMATCH")
    result_set=payload.get("resultSet")
    if not isinstance(result_set,dict):
        raise CanonicalError("MALFORMED_PROVIDER_RESULT_SET")
    count=result_set.get("count")
    if isinstance(count,bool) or not isinstance(count,int) or count!=len(payload["matches"]):
        raise CanonicalError("PARTIAL_PROVIDER_RESPONSE")
    result=[];seen=set();observed=utc(observed)
    for source in payload["matches"]:
        try:
            mid=_id(source["id"])
            if mid in seen:raise CanonicalError("DUPLICATE_PROVIDER_FIXTURE")
            seen.add(mid)
            status=source["status"]
            if status not in STATUSES:raise CanonicalError("UNKNOWN_PROVIDER_STATUS")
            kickoff=utc(source["utcDate"])
            start=source["season"]["startDate"]
            if not isinstance(start,str) or int(start[:4])!=season:
                raise CanonicalError("PROVIDER_SEASON_MISMATCH")
            row={"id":mid,"league":league,"season":season,"kickoff_utc":kickoff,"status":status}
            for side in ("home","away"):
                team=source[side+"Team"]
                row[side+"_id"]=_id(team["id"])
                name=team.get("name")
                if not isinstance(name,str) or not name.strip() or len(name)>200:
                    raise CanonicalError("INVALID_PROVIDER_TEAM_NAME")
                row[side+"_name"]=name
            if row["home_id"]==row["away_id"]:raise CanonicalError("SELF_PROVIDER_FIXTURE")
            if status=="FINISHED":
                if kickoff>=observed:raise CanonicalError("FINISHED_FIXTURE_IN_FUTURE")
                score=source["score"]["fullTime"]
                for side in ("home","away"):
                    goal=score[side]
                    if isinstance(goal,bool) or not isinstance(goal,int) or goal<0:
                        raise CanonicalError("FINISHED_SCORE_MISSING_OR_INVALID")
                    row[side+"_goals"]=goal
            result.append(row)
        except CanonicalError:
            raise
        except (KeyError,TypeError,ValueError,OverflowError) as exc:
            raise CanonicalError("MALFORMED_PROVIDER_MATCH") from exc
    return result


class FootballData:
    def __init__(self,key,*,clock=None,transport=None):
        if not isinstance(key,str) or not re.fullmatch(r"[A-Za-z0-9._-]{16,256}",key):
            raise CanonicalError("MISSING_OR_INVALID_ENVIRONMENT_KEY")
        self._key=key
        self.clock=clock or RealClock()
        self.transport=transport
        self.receipts=[]

    def fetch(self,pairs):
        pairs=list(dict.fromkeys(pairs))
        if not pairs or len(pairs)>10 or any(l not in LEAGUES or isinstance(y,bool) or not isinstance(y,int) or not 2010<=y<=2100 for l,y in pairs):
            raise CanonicalError("INVALID_FETCH_PLAN")
        transport=self.transport or HTTPSnapshotTransport(allowed_pairs=pairs)
        start=self.clock.monotonic();last_start=None;not_before=start
        rows=[];self.receipts=[];fixture_ids=set()
        def remaining():
            value=DEADLINE-(self.clock.monotonic()-start)
            if value<=0:raise CanonicalError("FETCH_CONTROLLER_TIMEOUT")
            return value
        for league,season in pairs:
            for attempt in range(3):
                wait=max(0.0,not_before-self.clock.monotonic(),
                         (last_start+PACE-self.clock.monotonic()) if last_start is not None else 0.0)
                if wait>=remaining():raise CanonicalError("FETCH_CONTROLLER_TIMEOUT")
                if wait:self.clock.sleep(wait)
                remaining()
                last_start=self.clock.monotonic()
                path=f"/v4/competitions/{league}/matches?season={season}"
                state,response,error=bounded_request(transport,path,self._key,self.clock,remaining(),MAX_BODY)
                if error is not None:
                    self.receipts.append({"league":league,"season":season,"attempt":attempt+1,
                      "http_status":None,"error_type":type(error).__name__,"requested_utc":state.get("requested_utc"),
                      "completed_utc":utc(self.clock.utcnow())})
                    code="FETCH_"+type(error).__name__.upper()
                    if attempt<2 and not isinstance(error,ControllerTimeout):
                        not_before=self.clock.monotonic()+2**attempt
                        continue
                    raise CanonicalError(code)
                status=response.status
                # Only selected quota/retry headers are retained; no request headers or error bodies.
                headers=selected_headers(response.headers)
                try:
                    check_json_secret(headers,self._key)
                except SecretReflection as exc:
                    self.receipts.append({"league":league,"season":season,"attempt":attempt+1,
                      "http_status":status,"error_type":"HEADER_SECRET_REFLECTION",
                      "requested_utc":state.get("requested_utc"),"completed_utc":utc(self.clock.utcnow())})
                    raise CanonicalError("PROVIDER_SECRET_REFLECTION") from exc
                self.receipts.append({"league":league,"season":season,"attempt":attempt+1,
                                 "http_status":status,"rate_headers":headers,"requested_utc":state.get("requested_utc"),
                                 "completed_utc":utc(self.clock.utcnow())})
                if status in (401,403):
                    raise CanonicalError("PROVIDER_AUTH_OR_ENTITLEMENT_"+str(status))
                if status==429 or 500<=status<=599:
                    if attempt==2:raise CanonicalError("PROVIDER_HTTP_"+str(status))
                    delay=delay_seconds(headers.get("Retry-After"),self.clock,60.0 if status==429 else 2**attempt)
                    not_before=self.clock.monotonic()+max(PACE,delay)
                    continue
                if status!=200:raise CanonicalError("PROVIDER_HTTP_"+str(status))
                if len(response.body)>MAX_BODY:raise CanonicalError("PROVIDER_BODY_TOO_LARGE")
                try:
                    check_secret(response.body,self._key)
                    payload=json.loads(response.body)
                    check_json_secret(payload,self._key)
                except SecretReflection as exc:raise CanonicalError("PROVIDER_SECRET_REFLECTION") from exc
                except (ValueError,UnicodeError) as exc:raise CanonicalError("PROVIDER_JSON_INVALID") from exc
                batch=project_matches(payload,league,season,self.clock.utcnow())
                for row in batch:
                    if row["id"] in fixture_ids:raise CanonicalError("DUPLICATE_FIXTURE_ACROSS_RESPONSES")
                    fixture_ids.add(row["id"])
                rows.extend(batch)
                cooldown,_=rate_wait(headers,self.clock)
                not_before=self.clock.monotonic()+cooldown
                break
        remaining()
        return rows,self.receipts

