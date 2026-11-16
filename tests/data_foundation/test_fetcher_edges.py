"""Extra termination/output tests remain fake-transport-only."""
import json
import queue
import threading
from pathlib import Path
from datetime import datetime,timezone,timedelta
import pytest
from src.ingest import provider_snapshots as p

KEY="OFFLINE_EXTRA_FAKE_TOKEN_2381"
class Clock:
    def __init__(self):self.n=0
    def monotonic(self):return self.n
    def sleep(self,n):self.n+=n
    def utcnow(self):return datetime(2026,11,8,12,tzinfo=timezone.utc)+timedelta(seconds=self.n)
class Fake:
    def __init__(self,answer=None):self.answer=answer;self.calls=0;self.aborted=False
    def request(self,*args):
        self.calls+=1
        return self.answer or p.Response(200,{},b'{"matches":[]}')
    def abort(self):self.aborted=True;return True

def test_real_wall_queue_timeout_branch_and_abort_retained(tmp_path,monkeypatch):
    original=queue.Queue
    class InstantTimeout(original):
        def get(self,*args,**kwargs):raise queue.Empty
    monkeypatch.setattr(p.queue,"Queue",InstantTimeout)
    class AbortFails(Fake):
        def abort(self):
            self.aborted=True
            raise RuntimeError(KEY)  # Payload must never escape into output.
    fake=AbortFails()
    result=p.fetch_snapshots(KEY,transport=fake,clock=Clock(),output_root=tmp_path)
    manifest=json.loads(Path(result.manifest_path).read_text())
    assert fake.aborted and result.exit_code==1
    attempt=manifest["entries"][0]["attempts"][0]
    assert attempt["error_type"]=="REQUEST_TIMEOUT" and attempt["abort_error_type"]=="ABORT_FAILED"
    assert KEY not in Path(result.manifest_path).read_text()

def test_first_logical_target_403_after_retry_still_stops(tmp_path):
    class Sequence(Fake):
        def request(self,*args):
            self.calls+=1
            return p.Response(429,{"Retry-After":"1"},b"") if self.calls==1 else p.Response(403,{},b"")
    fake=Sequence();result=p.fetch_snapshots(KEY,transport=fake,clock=Clock(),output_root=tmp_path)
    assert result.exit_code==1 and fake.calls==2
    assert json.loads(Path(result.manifest_path).read_text())["stopped_reason"]=="HTTP_403"

def test_existing_manifest_never_overwritten(tmp_path):
    clock=Clock();run=tmp_path/clock.utcnow().strftime("%Y%m%dT%H%M%S.%fZ")
    class Existing(Fake):
        def request(self,*args):
            self.calls+=1
            if self.calls==1:(run/"manifest.json").write_bytes(b"existing manifest sentinel")
            return p.Response(200,{},b"{}")
    fake=Existing();result=p.fetch_snapshots(KEY,transport=fake,clock=clock,output_root=tmp_path)
    assert result.exit_code==2 and result.manifest_path is None
    assert (run/"manifest.json").read_bytes()==b"existing manifest sentinel"

def test_success_manifest_is_last_output_written(tmp_path,monkeypatch):
    order=[];original=p.write_exclusive
    def record(path,body):order.append(path.name);original(path,body)
    monkeypatch.setattr(p,"write_exclusive",record)
    result=p.fetch_snapshots(KEY,transport=Fake(),clock=Clock(),output_root=tmp_path)
    assert result.exit_code==0 and len(order)==17 and order[-1]=="manifest.json"

def test_stream_body_cap_with_fake_https_only(monkeypatch):
    class Reply:
        status=200
        def getheaders(self):return []
        def read(self,n):return b"x"*n
    class Connection:
        def __init__(self,*args,**kwargs):pass
        def request(self,*args,**kwargs):pass
        def getresponse(self):return Reply()
        def close(self):pass
    monkeypatch.setattr(p.http.client,"HTTPSConnection",Connection)
    with pytest.raises(p.BodyTooLarge):
        p.HTTPSnapshotTransport().request("/v4/competitions/PL/matches?season=2024",KEY,15,32)


def test_scan_files_detects_key_straddling_65536_byte_boundary(tmp_path):
    encoded=KEY.encode("ascii")
    prefix=b"x"*(65536-len(encoded)//2)
    path=tmp_path/"cross-boundary.bin"
    path.write_bytes(prefix+encoded+b"trailer")
    assert len(prefix)<65536<len(prefix)+len(encoded)
    with pytest.raises(p.SecretReflection):
        p.scan_files(tmp_path,KEY)
