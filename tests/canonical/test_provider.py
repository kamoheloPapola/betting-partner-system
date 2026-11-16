import json
import time
from datetime import datetime,timezone,timedelta
import pytest
from src.canonical.provider import FootballData,project_matches
from src.canonical.clock import CanonicalError
from src.ingest import provider_snapshots as snapshot

KEY="SYNTHETIC_ONLY_API_KEY_123456789"


class Clock:
    def __init__(self):self.value=0.
    def monotonic(self):return self.value
    def utcnow(self):return datetime(2026,8,1,tzinfo=timezone.utc)+timedelta(seconds=self.value)
    def sleep(self,seconds):self.value+=seconds


def match(status="TIMED"):
    item={"id":1,"utcDate":"2026-08-03T15:00:00Z","status":status,"season":{"startDate":"2026-08-01"},
          "homeTeam":{"id":57,"name":"Home"},"awayTeam":{"id":61,"name":"Away"}}
    if status=="FINISHED":
        item["utcDate"]="2026-07-30T15:00:00Z"
        item["score"]={"fullTime":{"home":1,"away":0}}
    return item


def payload(items=None,league="PL"):
    items=[match()] if items is None else items
    return {"competition":{"code":league},"resultSet":{"count":len(items)},"matches":items}


class Transport:
    def __init__(self,responses,clock):self.responses=list(responses);self.calls=[];self.clock=clock;self.aborted=False
    def request(self,path,key,timeout,max_body_bytes):
        assert key==KEY and timeout<=15
        self.calls.append((path,self.clock.value))
        return self.responses.pop(0)
    def abort(self):self.aborted=True;return True


def response(status=200,body=None,headers=None):
    return snapshot.Response(status,headers or {},json.dumps(body if body is not None else payload()).encode())


def test_minimal_projection_drops_people_and_all_unneeded_fields():
    source=match();source["referees"]=[{"name":"PERSON_MUST_NOT_BE_STORED"}];source["lineups"]=[{"name":"PRIVATE"}]
    result=project_matches(payload([source]),"PL",2026,"2026-08-01T00:00:00Z")
    assert set(result[0])=={"id","league","season","kickoff_utc","status","home_id","away_id","home_name","away_name"}


@pytest.mark.parametrize("status",["SCHEDULED","TIMED","IN_PLAY","PAUSED","POSTPONED","SUSPENDED","CANCELLED","AWARDED"])
def test_only_finished_creates_result_fields(status):
    source=match(status);source["score"]={"fullTime":{"home":99,"away":99}}
    item=project_matches(payload([source]),"PL",2026,"2026-08-01T00:00:00Z")[0]
    assert "home_goals" not in item


@pytest.mark.parametrize("status",[401,403,404,302])
def test_auth_entitlement_and_redirect_fail_without_workaround(status):
    clock=Clock();transport=Transport([response(status)],clock);client=FootballData(KEY,clock=clock,transport=transport)
    with pytest.raises(CanonicalError):client.fetch([("PL",2026),("BL1",2026)])
    assert len(transport.calls)==1 and len(client.receipts)==1
    assert KEY not in json.dumps(client.receipts)


def test_rate_limit_bounded_retry_and_pacing():
    clock=Clock();transport=Transport([response(429,headers={"Retry-After":"7"}),response()],clock)
    client=FootballData(KEY,clock=clock,transport=transport)
    rows,receipts=client.fetch([("PL",2026)])
    assert len(rows)==1 and len(receipts)==2 and transport.calls[1][1]-transport.calls[0][1]>=7


def test_zero_remaining_header_is_case_insensitive():
    clock=Clock();transport=Transport([
        response(body=payload([]),headers={"x-requests-available-minute":"0","x-requestcounter-reset":"60"}),
        response(body=payload([],league="BL1"))],clock)
    FootballData(KEY,clock=clock,transport=transport).fetch([("PL",2026),("BL1",2026)])
    assert transport.calls[1][1]>=60


def test_repeated_rate_limit_stops_at_three_attempts():
    clock=Clock();transport=Transport([response(429,headers={"Retry-After":"7"}) for _ in range(3)],clock)
    client=FootballData(KEY,clock=clock,transport=transport)
    with pytest.raises(CanonicalError,match="429"):client.fetch([("PL",2026)])
    assert len(client.receipts)==3


def test_wait_cannot_exceed_controller_deadline():
    clock=Clock();transport=Transport([response(429,headers={"Retry-After":"1000"})],clock)
    with pytest.raises(CanonicalError,match="CONTROLLER"):
        FootballData(KEY,clock=clock,transport=transport).fetch([("PL",2026)])
    assert len(transport.calls)==1


@pytest.mark.parametrize("where",["body","escaped_body","header"])
def test_reflected_key_never_reaches_retained_metadata(where):
    data=payload()
    headers={}
    if where=="body":data["ignored"]=KEY
    elif where=="escaped_body":data["ignored"]=KEY
    else:headers={"X-RequestCounter-Reset":KEY}
    raw=json.dumps(data).encode()
    if where=="escaped_body":raw=raw.replace(KEY.encode(),b"".join(("\\u%04x"%ord(c)).encode() for c in KEY))
    clock=Clock();transport=Transport([snapshot.Response(200,headers,raw)],clock)
    client=FootballData(KEY,clock=clock,transport=transport)
    with pytest.raises(CanonicalError,match="SECRET_REFLECTION"):client.fetch([("PL",2026)])
    assert KEY not in json.dumps(client.receipts)


def test_hard_whole_request_timeout_and_abort(monkeypatch):
    class Stall:
        def request(self,*args):time.sleep(.2);return response()
        def abort(self):self.aborted=True;return True
    monkeypatch.setattr(snapshot,"REQUEST_TIMEOUT",.02)
    clock=Clock();transport=Stall()
    client=FootballData(KEY,clock=clock,transport=transport)
    with pytest.raises(CanonicalError,match="REQUESTTIMEOUT"):client.fetch([("PL",2026)])
    assert transport.aborted and len(client.receipts)==3


@pytest.mark.parametrize("change",["duplicate","partial","missing_score","future_finished","unknown_status","bad_team"])
def test_malformed_data_fails_before_any_store_write(change):
    data=payload()
    if change=="duplicate":data["matches"]*=2;data["resultSet"]["count"]=2
    elif change=="partial":data["resultSet"]["count"]=2
    elif change=="missing_score":data["matches"][0]["status"]="FINISHED";data["matches"][0]["utcDate"]="2026-07-01T15:00:00Z"
    elif change=="future_finished":data["matches"][0]=match("FINISHED");data["matches"][0]["utcDate"]="2026-08-05T15:00:00Z"
    elif change=="unknown_status":data["matches"][0]["status"]="GUESS"
    else:data["matches"][0]["awayTeam"]["id"]=57
    with pytest.raises(CanonicalError):project_matches(data,"PL",2026,"2026-08-01T00:00:00Z")


def test_empty_season_response_is_explicit_valid_empty_batch():
    assert project_matches(payload([]),"PL",2026,"2026-08-01T00:00:00Z")==[]

