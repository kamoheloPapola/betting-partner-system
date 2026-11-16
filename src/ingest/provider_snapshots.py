"""football-data.org v4 immutable snapshots. Importing this module makes no request."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime,timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
import hashlib
import http.client
import json
import math
import os
import queue
import ssl
import threading
import time

HOST="api.football-data.org"
LEAGUES=("PL","BL1","FL1","SA","PD")
SEASONS=(2024,2025,2026)
PLAN=tuple((league,year) for year in reversed(SEASONS) for league in LEAGUES)+(("PL",2020),)
MAX_REQUESTS=18
REQUEST_TIMEOUT=15.0
MAX_BODY=8_000_000  # Decimal MB, conservatively within the owner's 8 MB ceiling.
CONTROLLER_TIMEOUT=600.0
PACE=7.0
RATE_NAMES=("X-Requests-Available-Minute","X-RequestCounter-Reset")
OUTPUT_ROOT=Path(__file__).resolve().parents[2]/"data/raw/football_data_org"

class ControllerTimeout(Exception):
    pass

class RequestTimeout(Exception):
    pass

class BodyTooLarge(Exception):
    pass

class SecretReflection(Exception):
    pass

class ClockError(Exception):
    pass

class RealClock:
    def monotonic(self):
        return time.monotonic()
    def utcnow(self):
        return datetime.now(timezone.utc)
    def sleep(self,seconds):
        time.sleep(seconds)

@dataclass
class Response:
    status:int
    headers:dict
    body:bytes

@dataclass
class RunResult:
    exit_code:int
    manifest_path:str|None
    successful_files:int
    error_type:str|None

def utc(clock):
    value=clock.utcnow()
    if value.tzinfo is None or value.utcoffset().total_seconds()!=0:
        raise ClockError("UTC clock required")
    return value.isoformat()

def selected_headers(headers):
    # Response headers only, never request headers or Location.
    lower={str(k).lower():(None if v is None else str(v)) for k,v in headers.items()}
    return {k:lower.get(k.lower()) for k in RATE_NAMES+("Retry-After",)}

class HTTPSnapshotTransport:
    """HTTPS to one hardcoded host; http.client never follows redirects."""
    def __init__(self,allowed_pairs=None):
        self._connection=None
        self._allowed_pairs=tuple(PLAN if allowed_pairs is None else allowed_pairs)
        if any(l not in LEAGUES or isinstance(y,bool) or not isinstance(y,int) or not 2010<=y<=2100 for l,y in self._allowed_pairs):
            raise ValueError("Invalid explicit endpoint allowlist")
    def abort(self):
        conn=self._connection
        if conn is not None:
            try:
                conn.close()
            except Exception:
                return False  # Controller records the secondary cleanup failure.
        return True
    def request(self,path,key,timeout,max_body_bytes):
        allowed={f"/v4/competitions/{l}/matches?season={y}" for l,y in self._allowed_pairs}
        if path not in allowed:
            raise ValueError("Unapproved endpoint")
        conn=http.client.HTTPSConnection(HOST,timeout=timeout,context=ssl.create_default_context())
        self._connection=conn
        try:
            conn.request("GET",path,headers={"X-Auth-Token":key,"Accept":"application/json"})
            response=conn.getresponse()
            headers=selected_headers(dict(response.getheaders()))
            if response.status!=200:
                return Response(response.status,headers,b"")
            chunks=[];total=0
            while total<=max_body_bytes:
                chunk=response.read(min(65536,max_body_bytes-total+1))
                if not chunk:
                    break
                total+=len(chunk)
                if total>max_body_bytes:
                    raise BodyTooLarge()
                chunks.append(chunk)
            return Response(response.status,headers,b"".join(chunks))
        finally:
            conn.close()
            self._connection=None

def bounded_request(transport,path,key,clock,controller_remaining,max_body_bytes):
    """Real wall-clock queue deadline bounds the ENTIRE transport call, not each read."""
    out=queue.Queue(maxsize=1);state={}
    def worker():
        try:
            state["requested_utc"]=utc(clock)
            state["started_mono"]=clock.monotonic()
            result=transport.request(path,key,min(REQUEST_TIMEOUT,controller_remaining),max_body_bytes)
            out.put((True,result))
        except Exception as exc:
            out.put((False,exc))
    thread=threading.Thread(target=worker,daemon=True,name="football-data-snapshot-request")
    thread.start()
    try:
        ok,value=out.get(timeout=min(REQUEST_TIMEOUT,controller_remaining))
    except queue.Empty:
        try:
            if transport.abort() is False:
                state["abort_error_type"]="ABORT_FAILED"
        except Exception:
            state["abort_error_type"]="ABORT_FAILED"
        exc=ControllerTimeout() if controller_remaining<=REQUEST_TIMEOUT else RequestTimeout()
        return state,None,exc
    elapsed=clock.monotonic()-state.get("started_mono",clock.monotonic())
    if elapsed>controller_remaining:
        return state,None,ControllerTimeout()
    if elapsed>REQUEST_TIMEOUT:
        return state,None,RequestTimeout()
    return state,value if ok else None,None if ok else value

def delay_seconds(value,clock,default):
    if value is None:
        return default
    try:
        number=float(value)
        if not math.isfinite(number) or number<0:
            return default
        if number>=1_000_000_000:
            return max(0.0,number-clock.utcnow().timestamp())
        return number
    except (TypeError,ValueError):
        pass
    try:
        stamp=datetime.fromisoformat(str(value).replace("Z","+00:00"))
        if stamp.tzinfo is None:
            raise ValueError()
        return max(0.0,(stamp-clock.utcnow()).total_seconds())
    except (TypeError,ValueError):
        try:
            stamp=parsedate_to_datetime(str(value))
            if stamp.tzinfo is None:
                raise ValueError()
            return max(0.0,(stamp-clock.utcnow()).total_seconds())
        except (TypeError,ValueError,OverflowError):
            return default

def rate_wait(headers,clock):
    value=headers.get(RATE_NAMES[0])
    try:
        remaining=int(value)
    except (TypeError,ValueError):
        return 0.0,"REMAINING_UNKNOWN_PACE_ONLY"
    if remaining==0:
        reset=headers.get(RATE_NAMES[1])
        return delay_seconds(reset,clock,60.0),"ZERO_REMAINING_RESET_OR_SAFE_60S"
    return 0.0,"REMAINING_POSITIVE" if remaining>0 else "INVALID_REMAINING_PACE_ONLY"

def check_secret(data,key):
    encoded=key.encode("ascii")
    if encoded in data:
        raise SecretReflection()

def check_json_secret(value,key):
    if isinstance(value,str):
        if key in value:
            raise SecretReflection()
    elif isinstance(value,dict):
        for name,child in value.items():
            check_json_secret(name,key);check_json_secret(child,key)
    elif isinstance(value,list):
        for child in value:
            check_json_secret(child,key)

def scan_files(directory,key):
    encoded=key.encode("ascii");overlap=max(0,len(encoded)-1)
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            tail=b""
            with path.open("rb") as stream:
                for chunk in iter(lambda:stream.read(65536),b""):
                    data=tail+chunk
                    if encoded in data:
                        raise SecretReflection()
                    tail=data[-overlap:] if overlap else b""

def encode_json(value,key):
    body=(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n").encode("utf-8")
    check_secret(body,key)
    return body

def write_exclusive(path,body):
    with path.open("xb") as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())

def error_type(exc):
    if isinstance(exc,(TimeoutError,RequestTimeout)):
        return "REQUEST_TIMEOUT"
    if isinstance(exc,ControllerTimeout):
        return "CONTROLLER_TIMEOUT"
    if isinstance(exc,BodyTooLarge):
        return "OVERSIZE_BODY"
    if isinstance(exc,SecretReflection):
        return "SECRET_REFLECTION_BLOCKED"
    if isinstance(exc,ClockError):
        return "ENVIRONMENT_CLOCK_ERROR"
    if isinstance(exc,OSError):
        return "NETWORK_OS_ERROR"
    return "TRANSPORT_ERROR"

def fetch_snapshots(key,*,transport,clock,output_root=OUTPUT_ROOT,
                    controller_seconds=CONTROLLER_TIMEOUT,max_body_bytes=MAX_BODY):
    """Injected dependencies are mandatory; production main supplies real HTTPS/clock."""
    if not isinstance(key,str) or not key or any(ord(c)<33 or ord(c)>126 for c in key):
        return RunResult(2,None,0,"MISSING_OR_INVALID_ENVIRONMENT_KEY")
    directory=None;entries=[];request_count=0;written=0;stop=None;exit_code=0
    try:
        started_utc=utc(clock);started_mono=clock.monotonic()
        run_id=clock.utcnow().strftime("%Y%m%dT%H%M%S.%fZ")
        directory=Path(output_root)/run_id
        if key in str(directory):
            raise SecretReflection()
        directory.mkdir(parents=True,exist_ok=False)
    except Exception:
        return RunResult(2,None,0,"ENVIRONMENT_OR_OUTPUT_SETUP")
    deadline=started_mono+controller_seconds;last_start=None;not_before=started_mono
    try:
        for index,(league,season) in enumerate(PLAN):
            entry={"league":league,"season":season,"path":None,"http_status":None,
                   "size_bytes":None,"sha256":None,"requested_utc":None,"received_utc":None,
                   "rate_headers":{},"error_type":None,"attempts":[]}
            entries.append(entry)
            if stop:
                entry["error_type"]="NOT_REQUESTED_STOPPED"
                continue
            for retry in range(2):
                current=clock.monotonic()
                target=max(not_before,current if last_start is None else last_start+PACE)
                if target>=deadline or current>=deadline:
                    entry["error_type"]="CONTROLLER_TIMEOUT";stop="CONTROLLER_TIMEOUT";exit_code=max(exit_code,1);break
                if target>current:
                    clock.sleep(target-current)
                current=clock.monotonic()
                if current+1e-6<target:
                    raise ClockError("Clock did not honor pacing")
                if current>=deadline:
                    entry["error_type"]="CONTROLLER_TIMEOUT";stop="CONTROLLER_TIMEOUT";exit_code=max(exit_code,1);break
                if request_count>=MAX_REQUESTS:
                    entry["error_type"]="REQUEST_BUDGET_EXCEEDED";stop="REQUEST_BUDGET_EXCEEDED";exit_code=max(exit_code,1);break
                path=f"/v4/competitions/{league}/matches?season={season}"
                request_count+=1
                state,response,error=bounded_request(transport,path,key,clock,deadline-current,max_body_bytes)
                last_start=state.get("started_mono",current)
                received=utc(clock)
                attempt={"requested_utc":state.get("requested_utc"),"received_utc":received,
                         "http_status":None,"rate_headers":{},"error_type":None}
                if state.get("abort_error_type"):
                    attempt["abort_error_type"]=state["abort_error_type"]
                entry["attempts"].append(attempt)
                entry["requested_utc"]=attempt["requested_utc"];entry["received_utc"]=received
                if error is not None:
                    kind=error_type(error);attempt["error_type"]=kind;entry["error_type"]=kind;exit_code=max(exit_code,1)
                    if kind in ["REQUEST_TIMEOUT","CONTROLLER_TIMEOUT"]:
                        stop=kind
                    if kind=="ENVIRONMENT_CLOCK_ERROR":
                        stop=kind;exit_code=2
                    break
                headers=selected_headers(response.headers)
                try:
                    check_json_secret(headers,key)
                except SecretReflection:
                    attempt["error_type"]="SECRET_REFLECTION_BLOCKED";entry["error_type"]=attempt["error_type"];stop=attempt["error_type"];exit_code=2;break
                code=int(response.status)
                attempt["http_status"]=code;entry["http_status"]=code
                attempt["rate_headers"]=headers;entry["rate_headers"]=headers
                wait,policy=rate_wait(headers,clock)
                attempt["rate_wait_policy"]=policy
                not_before=clock.monotonic()+wait
                if code==429:
                    attempt["error_type"]="HTTP_429";entry["error_type"]="HTTP_429"
                    if retry==1:
                        entry["error_type"]="REPEATED_HTTP_429";stop=entry["error_type"];exit_code=max(exit_code,1);break
                    retry_delay=min(60.0,delay_seconds(headers.get("Retry-After"),clock,60.0))
                    not_before=max(not_before,clock.monotonic()+retry_delay)
                    continue
                if code!=200:
                    kind=("HTTP_401" if code==401 else "HTTP_403" if code==403 else "HTTP_404" if code==404
                          else "REDIRECT_REFUSED" if 300<=code<400 else "HTTP_SERVER_ERROR" if code>=500 else "HTTP_CLIENT_ERROR")
                    attempt["error_type"]=kind;entry["error_type"]=kind;exit_code=max(exit_code,1)
                    if code==401 or (code==403 and index==0):
                        stop=kind
                    break
                body=response.body
                try:
                    if not isinstance(body,bytes):
                        raise TypeError()
                    if len(body)>max_body_bytes:
                        raise BodyTooLarge()
                    check_secret(body,key)
                    parsed=json.loads(body)
                    check_json_secret(parsed,key)
                except SecretReflection:
                    attempt["error_type"]="SECRET_REFLECTION_BLOCKED";entry["error_type"]=attempt["error_type"];stop=entry["error_type"];exit_code=2;break
                except BodyTooLarge:
                    attempt["error_type"]="OVERSIZE_BODY";entry["error_type"]=attempt["error_type"];exit_code=max(exit_code,1);break
                except (ValueError,UnicodeError,TypeError,RecursionError):
                    attempt["error_type"]="NON_JSON_BODY";entry["error_type"]=attempt["error_type"];exit_code=max(exit_code,1);break
                destination=directory/f"{league}_{season}.json"
                try:
                    write_exclusive(destination,body)
                except FileExistsError:
                    attempt["error_type"]="BODY_EXISTS";entry["error_type"]=attempt["error_type"];stop=entry["error_type"];exit_code=2;break
                except OSError:
                    entry["incomplete_output_path"]=str(destination)
                    attempt["error_type"]="OUTPUT_WRITE_ERROR";entry["error_type"]=attempt["error_type"];stop=entry["error_type"];exit_code=2;break
                entry.update(path=str(destination),size_bytes=len(body),sha256=hashlib.sha256(body).hexdigest(),error_type=None)
                written+=1
                break
    except (KeyboardInterrupt,ControllerTimeout):
        stop="INTERRUPTED_OR_CONTROLLER_TIMEOUT";exit_code=max(exit_code,1)
        try:
            abort_error=transport.abort() is False
        except Exception:
            abort_error=True
        if entries:
            entries[-1]["error_type"]=stop
            if entries[-1]["attempts"]:
                entries[-1]["attempts"][-1]["error_type"]=stop
                if abort_error:entries[-1]["attempts"][-1]["abort_error_type"]="ABORT_FAILED"
    except Exception:
        # Never stringify unknown exceptions: server/client exceptions may contain secrets.
        stop="ENVIRONMENT_OR_OUTPUT_ERROR";exit_code=2
        if entries and entries[-1]["error_type"] is None:
            entries[-1]["error_type"]=stop
            if entries[-1]["attempts"]:
                entries[-1]["attempts"][-1]["error_type"]=stop
    try:
        for league,season in PLAN[len(entries):]:
            entries.append({"league":league,"season":season,"path":None,"http_status":None,
                            "size_bytes":None,"sha256":None,"requested_utc":None,"received_utc":None,
                            "rate_headers":{},"error_type":"NOT_REQUESTED_STOPPED","attempts":[]})
        if written!=len(PLAN):exit_code=max(exit_code,1)
        manifest={"schema":"football-data-org.snapshots.v1","host":HOST,"run_id":run_id,"started_utc":started_utc,
                  "finished_utc":utc(clock),"requests_made":request_count,"request_limit":MAX_REQUESTS,
                  "planned_files":len(PLAN),"successful_files":written,"exit_code":exit_code,"stopped_reason":stop,
                  "entries":entries,"secret_self_check":"Pre-write body/header/manifest plus full-file scan; no request headers stored"}
        scan_files(directory,key)
        body=encode_json(manifest,key)
        destination=directory/"manifest.json"
        write_exclusive(destination,body)  # Last output written, never overwritten.
        scan_files(directory,key)
        return RunResult(exit_code,str(destination),written,stop)
    except Exception:
        return RunResult(2,None,written,"MANIFEST_OR_SELF_CHECK_OUTPUT_ERROR")

def main():
    # This is the only environment-key access. Tests never invoke this entrypoint.
    key=os.environ.get("FOOTBALL_DATA_API_KEY","")
    result=fetch_snapshots(key,transport=HTTPSnapshotTransport(),clock=RealClock())
    # No key, request header, response body or raw exception text enters console output.
    print(json.dumps({"exit_code":result.exit_code,"manifest":result.manifest_path,
                      "successful_files":result.successful_files,"error_type":result.error_type}))
    return result.exit_code
