import hashlib
import json
import socket
from datetime import datetime,timezone,timedelta
from pathlib import Path
import pytest
from src.ingest import provider_snapshots as p

KEY="OFFLINE_TEST_ONLY_NOT_A_REAL_CREDENTIAL_7e12"
BODY=b' { "matches": [], "message": "offline fixture" }\n'

class Clock:
    def __init__(self):
        self.seconds=0.0;self.waits=[]
    def utcnow(self):
        return datetime(2026,11,8,12,0,tzinfo=timezone.utc)+timedelta(seconds=self.seconds)
    def monotonic(self):
        return self.seconds
    def sleep(self,n):
        self.waits.append(n);self.seconds+=n

class Fake:
    def __init__(self,clock,answers=(),callback=None):
        self.clock=clock;self.answers=list(answers);self.calls=[];self.callback=callback;self.aborted=False
    def abort(self):
        self.aborted=True
    def request(self,path,key,timeout,max_body_bytes):
        assert key==KEY
        assert timeout<=15
        self.calls.append((path,self.clock.monotonic()))
        if self.callback:self.callback(len(self.calls),path)
        value=self.answers.pop(0) if self.answers else p.Response(200,{},BODY)
        if isinstance(value,Exception):raise value
        return value

def perform(tmp_path,answers=(),**kwargs):
    clock=Clock();fake=Fake(clock,answers)
    result=p.fetch_snapshots(KEY,transport=fake,clock=clock,output_root=tmp_path,**kwargs)
    manifest=json.loads(Path(result.manifest_path).read_text()) if result.manifest_path else None
    return result,manifest,fake,clock


def test_gap_entrypoint_has_exactly_five_2026_requests(tmp_path, monkeypatch, capsys):
    clock = Clock()
    fake = Fake(clock)
    pairs = tuple((league, 2026) for league in p.LEAGUES)
    fetch = p.fetch_snapshots
    monkeypatch.delenv("SSLKEYLOGFILE", raising=False)
    monkeypatch.setenv("FOOTBALL_DATA_API_KEY", KEY)
    monkeypatch.setattr(p, "RealClock", lambda: clock)
    monkeypatch.setattr(p, "HTTPSnapshotTransport", lambda allowed_pairs: fake if allowed_pairs == pairs else pytest.fail())
    monkeypatch.setattr(p, "fetch_snapshots", lambda *a, **kw: fetch(*a, output_root=tmp_path, **kw))

    assert p.main(["--gap-2026"]) == 0
    report = json.loads(capsys.readouterr().out)
    manifest = json.loads(Path(report["manifest"]).read_bytes())
    assert manifest["requests_made"] == manifest["request_limit"] == manifest["successful_files"] == 5
    assert [path for path, _ in fake.calls] == [f"/v4/competitions/{league}/matches?season=2026" for league in p.LEAGUES]


def test_gap_request_limit_does_not_retry_rate_limits(tmp_path):
    plan = tuple((league, 2026) for league in p.LEAGUES)
    result, manifest, fake, _ = perform(tmp_path, [p.Response(429, {"Retry-After": "10"}, b"")],
                                        plan=plan, request_limit=5, attempts=1)
    assert result.exit_code == 1 and manifest["requests_made"] == len(fake.calls) == 1
    assert manifest["stopped_reason"] == "HTTP_429"


def test_complete_raw_bytes_manifest_hashes_utc_and_pacing(tmp_path):
    result,manifest,fake,clock=perform(tmp_path)
    assert result.exit_code==0 and len(fake.calls)==16 and result.successful_files==16
    assert fake.calls[-1][0]=="/v4/competitions/PL/matches?season=2020"
    assert all(b[1]-a[1]>=7 for a,b in zip(fake.calls,fake.calls[1:]))
    assert manifest["request_limit"]==18
    for entry in manifest["entries"]:
        path=Path(entry["path"])
        assert path.read_bytes()==BODY
        assert entry["size_bytes"]==len(BODY)
        assert entry["sha256"]==hashlib.sha256(BODY).hexdigest()
        for name in ["requested_utc","received_utc"]:
            assert datetime.fromisoformat(entry[name]).utcoffset()==timedelta(0)
    for path in tmp_path.rglob("*"):
        if path.is_file():assert KEY.encode() not in path.read_bytes()

def test_retry_after_is_bounded_and_recorded(tmp_path):
    result,m,f,c=perform(tmp_path,[p.Response(429,{"Retry-After":"999"},b""),p.Response(200,{},BODY)])
    assert result.exit_code==0 and len(f.calls)==17
    assert f.calls[1][1]-f.calls[0][1]==60
    assert m["entries"][0]["attempts"][0]["error_type"]=="HTTP_429"
    assert m["entries"][0]["error_type"] is None

def test_repeated_429_stops(tmp_path):
    result,m,f,c=perform(tmp_path,[p.Response(429,{"Retry-After":"1"},b""),p.Response(429,{},b"")])
    assert result.exit_code==1 and len(f.calls)==2
    assert m["stopped_reason"]=="REPEATED_HTTP_429"
    assert all(x["path"] is None for x in m["entries"])

@pytest.mark.parametrize("status",[403,404,500,400,302])
def test_later_http_errors_record_and_continue_without_body(tmp_path,status):
    result,m,f,c=perform(tmp_path,[p.Response(200,{},BODY),p.Response(status,{},b"not stored")])
    assert result.exit_code==1 and len(f.calls)==16 and result.successful_files==15
    assert m["entries"][1]["http_status"]==status and m["entries"][1]["path"] is None

@pytest.mark.parametrize("status",[401,403])
def test_first_auth_or_entitlement_error_stops(tmp_path,status):
    result,m,f,c=perform(tmp_path,[p.Response(status,{},b"")])
    assert result.exit_code==1 and len(f.calls)==1 and result.successful_files==0
    assert m["stopped_reason"]=="HTTP_"+str(status)

def test_later_401_also_stops(tmp_path):
    result,m,f,c=perform(tmp_path,[p.Response(200,{},BODY),p.Response(401,{},b"")])
    assert result.exit_code==1 and len(f.calls)==2 and m["stopped_reason"]=="HTTP_401"

def test_correct_rate_header_names_and_reset(tmp_path):
    headers={"x-requests-available-minute":"0","x-requestcounter-reset":"20"}
    result,m,f,c=perform(tmp_path,[p.Response(200,headers,BODY)])
    assert result.exit_code==0 and f.calls[1][1]==20
    assert m["entries"][0]["rate_headers"]["X-Requests-Available-Minute"]=="0"
    assert m["entries"][0]["rate_headers"]["X-RequestCounter-Reset"]=="20"

def test_unknown_reset_uses_bounded_controller_safe_wait(tmp_path):
    result,m,f,c=perform(tmp_path,[p.Response(200,{"X-Requests-Available-Minute":"0","X-RequestCounter-Reset":"unparseable"},BODY)])
    assert result.exit_code==0 and f.calls[1][1]==60

def test_reset_iso_and_retry_http_date_parsing():
    c=Clock()
    assert p.delay_seconds("2026-11-08T12:00:12Z",c,60)==12
    assert p.delay_seconds("Sun, 08 Nov 2026 12:00:09 GMT",c,60)==9

def test_controller_timeout_stops_before_next_start(tmp_path):
    result,m,f,c=perform(tmp_path,controller_seconds=5)
    assert result.exit_code==1 and len(f.calls)==1
    assert m["stopped_reason"]=="CONTROLLER_TIMEOUT"

def test_request_total_elapsed_timeout(tmp_path):
    clock=Clock()
    def slow(n,path):clock.seconds+=16
    fake=Fake(clock,callback=slow)
    result=p.fetch_snapshots(KEY,transport=fake,clock=clock,output_root=tmp_path)
    m=json.loads(Path(result.manifest_path).read_text())
    assert result.exit_code==1 and len(fake.calls)==1
    assert m["entries"][0]["error_type"]=="REQUEST_TIMEOUT"

@pytest.mark.parametrize("value,kind",[
    (p.Response(200,{},b"<html>error</html>"),"NON_JSON_BODY"),
    (p.Response(200,{},b" "*33),"OVERSIZE_BODY"),
    (p.BodyTooLarge(),"OVERSIZE_BODY"),
    (socket.gaierror(),"NETWORK_OS_ERROR"),
    (ValueError(KEY),"TRANSPORT_ERROR"),
])
def test_failure_types_retained_without_exception_payload(tmp_path,value,kind):
    result,m,f,c=perform(tmp_path,[value],max_body_bytes=32 if kind=="OVERSIZE_BODY" else 8192)
    assert result.exit_code==1 and m["entries"][0]["error_type"]==kind
    assert m["entries"][0]["path"] is None
    assert KEY not in Path(result.manifest_path).read_text()

def test_request_budget_never_exceeds_18(tmp_path):
    answers=[]
    for _ in range(3):answers.extend([p.Response(429,{"Retry-After":"1"},b""),p.Response(200,{},BODY)])
    result,m,f,c=perform(tmp_path,answers)
    assert len(f.calls)==18 and result.exit_code==1
    assert m["stopped_reason"]=="REQUEST_BUDGET_EXCEEDED"

def test_exclusive_body_create_refusal_keeps_existing_bytes(tmp_path):
    c=Clock();run=tmp_path/c.utcnow().strftime("%Y%m%dT%H%M%S.%fZ")
    def create_existing(n,path):
        if n==1:(run/"PL_2026.json").write_bytes(b"owner sentinel")
    fake=Fake(c,callback=create_existing)
    result=p.fetch_snapshots(KEY,transport=fake,clock=c,output_root=tmp_path)
    m=json.loads(Path(result.manifest_path).read_text())
    assert result.exit_code==2 and len(fake.calls)==1
    assert (run/"PL_2026.json").read_bytes()==b"owner sentinel"
    assert m["entries"][0]["error_type"]=="BODY_EXISTS"

def test_existing_run_directory_never_modified(tmp_path):
    c=Clock();run=tmp_path/c.utcnow().strftime("%Y%m%dT%H%M%S.%fZ");run.mkdir()
    (run/"sentinel").write_bytes(b"keep")
    fake=Fake(c);result=p.fetch_snapshots(KEY,transport=fake,clock=c,output_root=tmp_path)
    assert result.exit_code==2 and not fake.calls
    assert [x.name for x in run.iterdir()]==["sentinel"]

@pytest.mark.parametrize("body,headers",[
    (json.dumps({"echo":KEY}).encode(),{}),
    (('{"echo":"\\u004f'+KEY[1:]+'"}').encode(),{}),
    (BODY,{"Retry-After":KEY}),
])
def test_secret_reflections_blocked_before_output(tmp_path,body,headers):
    result,m,f,c=perform(tmp_path,[p.Response(200,headers,body)])
    assert result.exit_code==2 and result.successful_files==0 and len(f.calls)==1
    assert m["entries"][0]["error_type"]=="SECRET_REFLECTION_BLOCKED"
    for path in tmp_path.rglob("*"):
        if path.is_file():assert KEY.encode() not in path.read_bytes()

def test_output_partial_is_retained_flagged_not_success(tmp_path,monkeypatch):
    original=p.write_exclusive
    def fail_body(path,body):
        if path.name=="PL_2026.json":
            path.write_bytes(b'{"incomplete":')
            raise OSError("synthetic output failure")
        original(path,body)
    monkeypatch.setattr(p,"write_exclusive",fail_body)
    result,m,f,c=perform(tmp_path)
    assert result.exit_code==2 and m["entries"][0]["error_type"]=="OUTPUT_WRITE_ERROR"
    assert m["entries"][0]["path"] is None
    assert Path(m["entries"][0]["incomplete_output_path"]).read_bytes()==b'{"incomplete":'

def test_missing_key_no_output_or_request(tmp_path):
    c=Clock();fake=Fake(c)
    result=p.fetch_snapshots("",transport=fake,clock=c,output_root=tmp_path)
    assert result.exit_code==2 and not fake.calls and not list(tmp_path.iterdir())

def test_real_transport_surface_uses_only_fake_https_and_refuses_redirect(monkeypatch):
    calls=[]
    class Reply:
        status=302
        def getheaders(self):return [("Location","https://unapproved.invalid/")]
        def read(self,*args):raise AssertionError("Redirect body must not be fetched")
    class Connection:
        def __init__(self,host,timeout,context):
            assert host==p.HOST and timeout==15
        def request(self,method,path,headers):
            assert method=="GET" and headers["X-Auth-Token"]==KEY
            assert set(headers)=={"X-Auth-Token","Accept"}
            assert KEY not in path
            calls.append(path)
        def getresponse(self):return Reply()
        def close(self):pass
    monkeypatch.setattr(p.http.client,"HTTPSConnection",Connection)
    result=p.HTTPSnapshotTransport().request("/v4/competitions/PL/matches?season=2024",KEY,15,8192)
    assert result.status==302 and len(calls)==1
    assert "Location" not in result.headers


def test_latest_season_first_canary_does_not_stop_on_old_season_403(tmp_path):
    clock=Clock()
    class LatestSeasonOnly(Fake):
        def request(self,path,key,timeout,max_body_bytes):
            assert key==KEY and timeout<=15
            self.calls.append((path,self.clock.monotonic()))
            return p.Response(200,{},BODY) if path.endswith("?season=2026") else p.Response(403,{},b"")
    fake=LatestSeasonOnly(clock)
    result=p.fetch_snapshots(KEY,transport=fake,clock=clock,output_root=tmp_path)
    manifest=json.loads(Path(result.manifest_path).read_text())
    expected=[f"/v4/competitions/{league}/matches?season={year}"
              for year in (2026,2025,2024) for league in p.LEAGUES]
    expected.append("/v4/competitions/PL/matches?season=2020")
    assert fake.calls[0][0]=="/v4/competitions/PL/matches?season=2026"
    assert [path for path,start in fake.calls]==expected
    assert len(fake.calls)==16 and result.successful_files==5 and result.exit_code==1
    assert manifest["requests_made"]==16 and manifest["stopped_reason"] is None
    assert sum(entry["path"] is not None for entry in manifest["entries"])==5
    for entry in manifest["entries"]:
        if entry["season"]==2026:
            assert entry["http_status"]==200 and entry["path"] is not None
            assert entry["error_type"] is None
        else:
            assert entry["http_status"]==403 and entry["error_type"]=="HTTP_403"
            assert entry["path"] is None
