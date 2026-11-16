"""NEXT-36 read-only root-cause audit; no repaired data or matcher substitution."""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
import ast
import csv
import gzip
import io
import json
import re
import unicodedata
from .audit import project_metadata, parse_day, load_metadata, MetadataError
from .snapshot_metadata import identity, verify_snapshots, project_snapshot, flatten_snapshot
from .repair_plan import prepare_aliases, simulation

LEAGUES=("PL","BL1","FL1","SA","PD")
PACKET_NAME="NEXT-36-root-cause-audit-01"
TARGET_LINES={179,182,295,296}
ABSORBED=(("BL1",28,"1. FC Union Berlin",16),("BL1",44,"1. FC Heidenheim 1846",16),
    ("FL1",529,"Stade Rennais FC 1901",1045),("SA",487,"AC Pisa 1909",98),
    ("PD",78,"Club Atlético de Madrid",86),("PD",90,"Real Betis Balompié",86),
    ("PD",92,"Real Sociedad de Fútbol",86),("PD",80,"RCD Espanyol de Barcelona",81))
EXPECTED_2024={"PL":760,"BL1":612,"FL1":306,"SA":761,"PD":760}

def utc_now():
    return datetime.now(timezone.utc).isoformat()

def write_new(repo,packet,leaf,body):
    repo,packet=Path(repo).resolve(),Path(packet).resolve()
    if packet.name!=PACKET_NAME or not packet.is_relative_to(repo/"tmp"):
        raise ValueError("Only the NEXT-36 tmp packet is writable")
    leaf=Path(leaf)
    if leaf.is_absolute() or ".." in leaf.parts:raise ValueError("Traversal refused")
    target=(packet/leaf).resolve()
    if target==packet or not target.is_relative_to(packet):raise ValueError("Packet escape")
    if target.exists() or target.is_symlink():raise FileExistsError("Immutable output exists")
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open("xb") as stream:stream.write(body if isinstance(body,bytes) else body.encode("utf-8"))
    return identity(target)

def emit(repo,packet,name,value):
    raw=json.dumps(value,ensure_ascii=False,separators=(",",":")).encode("utf-8")
    if len(raw)>25000:name+=".gz";raw=gzip.compress(raw,mtime=0)
    return write_new(repo,packet,name,raw)

def csv_text(rows,fields):
    text=io.StringIO(newline="")
    writer=csv.DictWriter(text,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    return text.getvalue()

def selected_cells(raw,indices,*,repo,path):
    allowed={"date","home","away"}
    if "kickoff" in indices.values():
        approved=Path(repo).resolve()/"data/processed/matches/PD_2025.csv"
        if Path(path).resolve()!=approved:raise ValueError("Kickoff cell refused outside approved PD2025 file")
        allowed.add("kickoff")
    if not set(indices.values()).issubset(allowed):
        raise ValueError("Nonwhitelisted score/odds/other cell requested")
    return project_metadata(raw,indices)

def date_label(day):
    value=date.fromisoformat(day)
    return value.year if value.month>=6 else value.year-1

def count_dates(repo,path,column):
    path=Path(path);before=identity(path);counts=Counter();errors=[];raw_count=0
    try:
        with path.open("rb") as stream:
            first=stream.readline(65536)
            if not first:return {"identity":before,"raw_rows":0,"valid_rows":0,"counts":{},"errors":["EMPTY_FILE"],"header_BOM":False,"header_EOL":"NONE"}
            fields=next(csv.reader([first.decode("utf-8-sig")]))
            if fields.count(column)!=1:raise ValueError("Date header missing/duplicated")
            index=fields.index(column)
            for line,raw in enumerate(stream,2):
                raw_count+=1
                try:
                    cells=selected_cells(raw,{index:"date"},repo=repo,path=path)
                    day=parse_day(cells.get("date",""));counts[date_label(day)]+=1
                except (MetadataError,ValueError,UnicodeError) as exc:
                    errors.append({"path":str(path),"line":line,"type":type(exc).__name__,"context":"Date-only projection; no values logged"})
    except (OSError,UnicodeError,csv.Error,ValueError) as exc:
        raise ValueError(f"Date-only audit failed file={path} column={column}: {type(exc).__name__}") from exc
    after=identity(path)
    if (before["size_bytes"],before["sha256"])!=(after["size_bytes"],after["sha256"]):
        raise ValueError(f"Input changed during date-only read: {path}")
    return {"identity":before,"raw_rows":raw_count,"valid_rows":sum(counts.values()),"counts":dict(counts),"errors":errors,
        "header_BOM":first.startswith(b"\xef\xbb\xbf"),"header_EOL":"CRLF" if first.endswith(b"\r\n") else "LF" if first.endswith(b"\n") else "NONE"}

def group_hashes(paths):
    groups=defaultdict(list)
    for path in paths:
        item=identity(path);groups[item["sha256"]].append(item)
    return [{"sha256":digest,"files":files} for digest,files in sorted(groups.items())]

def predict_counts(date_files,year):
    return sum(x["counts"].get(year,x["counts"].get(str(year),0)) for x in date_files)

def shared_tokens(left,right):
    # Diagnostic only: no canonical selection, score, substring rank or fuzzy result.
    def tokens(value):
        clean=unicodedata.normalize("NFKD",value).encode("ascii","ignore").decode("ascii").upper()
        return set(re.findall(r"[A-Z0-9]+",clean))
    return sorted(tokens(left)&tokens(right))

def naming_import_gate(source):
    tree=ast.parse(source);blocked=[]
    for node in tree.body:
        if isinstance(node,(ast.Assign,ast.AnnAssign)):
            value=node.value
            if isinstance(value,ast.Call) and isinstance(value.func,ast.Name) and value.func.id=="_load_season_maps":
                blocked.append({"line":node.lineno,"call":"_load_season_maps","reason":"Import executes data-file read"})
    return {"status":"NOT_RUN" if blocked else "UNKNOWN","blocking_calls":blocked,
        "reason":"Direct import-time file I/O" if blocked else "Full import closure still requires static approval"}

def kickoff_schema(repo,path):
    approved=Path(repo).resolve()/"data/processed/matches/PD_2025.csv"
    if Path(path).resolve()!=approved:raise ValueError("Kickoff schema outside approved file")
    try:
        with approved.open("rb") as stream:
            first=stream.readline(65536)
            if not first:return {"status":"NOT_RUN","column":None,"reason":"Empty PD2025 file"}
            fields=next(csv.reader([first.decode("utf-8-sig")]))
    except (OSError,UnicodeError,csv.Error) as exc:
        raise ValueError(f"H-M schema file={approved}: {type(exc).__name__}") from exc
    candidates=[f for f in fields if re.search(r"time|kick.?off",f,re.I)]
    if not candidates:return {"status":"NOT_RUN","column":None,"reason":"No kickoff-time column"}
    if len(candidates)!=1:return {"status":"UNKNOWN","column":None,"reason":"Multiple time columns; no column chosen","candidates":candidates}
    if not re.fullmatch(r"time|kickoff[_ ]?time|kick[_ ]?off[_ ]?time|kickoff[_ ]?utc",candidates[0],re.I):
        return {"status":"UNKNOWN","column":None,"reason":"Time-like column not explicitly kickoff-time","candidates":candidates}
    return {"status":"PASS","column":candidates[0],"reason":"Single explicit kickoff-time header"}

def parse_clock(value):
    if not isinstance(value,str) or not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d(?::00)?",value.strip()):
        return None
    hh,mm=map(int,value.strip().split(":")[:2])
    return None if hh==mm==0 else hh*60+mm

def calibrate_times(controls):
    offsets=defaultdict(list);exceptions=[];clocks=[];distribution=Counter()
    for row in controls:
        clock=parse_clock(row["csv_time"])
        if clock is None:
            exceptions.append({"line":row["line"],"reason":"MISSING_OR_PLACEHOLDER_CLOCK"});continue
        try:
            provider=datetime.fromisoformat(row["provider_utc"].replace("Z","+00:00"))
            if provider.tzinfo is None:raise ValueError("Naive provider time")
            provider=provider.astimezone(timezone.utc)
            local=datetime.fromisoformat(row["date"])+timedelta(minutes=clock)
            offset=(local.replace(tzinfo=timezone.utc)-provider).total_seconds()/60
            if offset!=int(offset) or abs(offset)>840:raise ValueError("Unsupported offset")
        except (ValueError,TypeError) as exc:
            exceptions.append({"line":row["line"],"reason":"INVALID_TIME_CONTEXT","error":type(exc).__name__});continue
        side="before" if row["date"]<"2026-03-29" else "after"
        offsets[side].append(int(offset));distribution[int(offset)]+=1;clocks.append(clock)
    rule={"status":"UNKNOWN","reason":"","n":len(controls),"n_valid":len(clocks),
        "offset_distribution":dict(distribution),"placeholders_or_exceptions":len(exceptions),"exceptions":exceptions,
        "side_counts":{k:len(offsets[k]) for k in ("before","after")},"boundary":"2026-03-29","offset_minutes":{}}
    if exceptions:rule["reason"]="Control placeholder/invalid values; no exclusion or tuning";return rule
    if not clocks:rule["reason"]="No controls";return rule
    if len(set(clocks))==1:rule["reason"]="Uniform clock may be a repeated default";return rule
    if any(len(set(v))>1 for v in offsets.values()):
        rule["reason"]="Unexplained mixed offsets within boundary side";return rule
    values=set(distribution)
    if len(values)==1:
        value=next(iter(values));local=value!=0
        if local and any(len(offsets[k])<30 for k in ("before","after")):
            rule["reason"]="Local offset needs30 controls on each side";return rule
        rule.update(status="PASS",reason="Observed constant offset; geographic timezone not inferred",
            mode="UTC" if value==0 else "CONSTANT_OFFSET",offset_minutes={"before":value,"after":value})
    elif len(values)==2 and all(offsets.get(k) for k in ("before","after")):
        a,b=offsets["before"][0],offsets["after"][0]
        if b-a!=60 or min(len(offsets["before"]),len(offsets["after"]))<30:
            rule["reason"]="Unexplained shift or insufficient local/DST controls";return rule
        rule.update(status="PASS",reason="One-hour offset transition supported on both sides; location not inferred",
            mode="DST_OFFSET",offset_minutes={"before":a,"after":b})
    else:rule["reason"]="More than one unexplained offset"
    return rule

def freeze_rule(repo,packet,rule,leaf="H-M-calibration-rule.json"):
    receipt=write_new(repo,packet,leaf,json.dumps(rule,sort_keys=True,separators=(",",":")))
    return {"identity":receipt,"utc":utc_now(),"target_read_started":False}

def verify_frozen(receipt):
    current=identity(receipt["identity"]["path"])
    if any(current[k]!=receipt["identity"][k] for k in ("size_bytes","sha256")):
        raise ValueError("Frozen H-M rule changed; target read/apply refused")
    return json.loads(Path(current["path"]).read_text(encoding="utf-8"))

def read_kickoff_cells(repo,path,column,lines,phase,receipt=None):
    lines=set(lines);schema=kickoff_schema(repo,path)
    if schema["status"]!="PASS" or schema["column"]!=column:raise ValueError("Approved kickoff schema unavailable")
    if phase=="controls":
        if lines&TARGET_LINES:raise ValueError("Target clocks forbidden before frozen rule")
    elif phase=="targets":
        if lines!=TARGET_LINES or receipt is None:raise ValueError("Target read requires all four lines and frozen rule")
        rule=verify_frozen(receipt)
        if rule["status"]!="PASS":raise ValueError("Calibration gate not PASS")
        if receipt["target_read_started"]:raise ValueError("Target time cells may be read only once")
        receipt["target_read_started"]=True
    else:raise ValueError("Unknown kickoff phase")
    output={}
    try:
        with Path(path).open("rb") as stream:
            fields=next(csv.reader([stream.readline().decode("utf-8-sig")]))
            index=fields.index(column)
            for number,raw in enumerate(stream,2):
                if number in lines:
                    output[number]=selected_cells(raw,{index:"kickoff"},repo=repo,path=path).get("kickoff","")
    except (OSError,UnicodeError,csv.Error,ValueError) as exc:
        raise ValueError(f"H-M {phase} file={path} column={column}: {type(exc).__name__}") from exc
    return output

def hm_verdict(rows):
    for group in ((179,182),(295,296)):
        selected=[r for r in rows if r["line"] in group]
        if len(selected)!=2:return "UNKNOWN"
        clocks=[parse_clock(r["csv_time"]) for r in selected]
        if None in clocks or clocks[0]==clocks[1]:return "UNKNOWN"
    if any(r.get("outside_candidate") for r in rows):return "FAIL"
    if any(len(r["matched_fixture_ids"])!=1 for r in rows):return "UNKNOWN"
    for group in ((179,182),(295,296)):
        ids=[r["matched_fixture_ids"][0] for r in rows if r["line"] in group]
        if len(set(ids))!=2:return "FAIL"
    return "PASS"

def apply_targets(rows,candidates,receipt):
    rule=verify_frozen(receipt)
    if not receipt["target_read_started"] or rule["status"]!="PASS":raise ValueError("Frozen target phase missing")
    output=[]
    for row in rows:
        clock=parse_clock(row["csv_time"]);matched=[]
        if clock is not None:
            side="before" if row["date"]<rule["boundary"] else "after"
            instant=(datetime.fromisoformat(row["date"])+timedelta(minutes=clock-rule["offset_minutes"][side])).replace(tzinfo=timezone.utc)
            for provider in candidates[row["line"]]:
                value=datetime.fromisoformat(provider["utcDate"].replace("Z","+00:00"))
                if value.tzinfo is None:raise ValueError(f"Naive target provider timestamp line={row['line']}")
                value=value.astimezone(timezone.utc)
                if value==instant:matched.append(provider["id"])
        output.append({**row,"matched_fixture_ids":matched,"outside_candidate":False,"authority":"UNREVIEWED"})
    return {"status":hm_verdict(output),"rows":output}

def load_reference(path):
    path=Path(path)
    raw=gzip.decompress(path.read_bytes()) if path.suffix==".gz" else path.read_bytes()
    return json.loads(raw)

def run(repo,packet):
    repo,packet=Path(repo).resolve(),Path(packet).resolve()
    prereg=load_reference(packet/"preregistration.json")["hypotheses"]
    if identity(packet/"HYPOTHESES.md")["sha256"]!=prereg["sha256"]:raise ValueError("Preregistration changed")
    prior=packet.parent/"NEXT-35-repair-plan-audit-01"
    gate=load_reference(packet/"static-gates.json")
    if gate["H-J1"]["status"]!="PASS":raise ValueError("H-J1 trace/dedupe gate requires stop")
    historical={};hash_groups={};source_dates=[];inputs=[]
    for league in LEAGUES:
        paths=sorted((repo/"data/historical"/league).glob("*.csv"))
        files=[count_dates(repo,q,"Date") for q in paths]
        for item in files:inputs.append(item["identity"]);item["league"]=league
        historical[league]=files;source_dates.extend(files);hash_groups[league]=group_hashes(paths)
    h2=[];h3=[];h4=[];observed={}
    prior_hh=load_reference(prior/"selected-final/results/H-H-2024.json.gz")
    for league in LEAGUES:
        q=repo/"data/processed/matches"/f"{league}_2024.csv";actual=count_dates(repo,q,"date");inputs.append(actual["identity"])
        prediction=predict_counts(historical[league],2024)
        profile=next(x["primary"] for x in prior_hh if x["league"]==league)
        ok=prediction==EXPECTED_2024[league]==actual["raw_rows"]==profile["raw_rows"] and not actual["errors"]
        h2.append({"league":league,"registered":EXPECTED_2024[league],"date_only_prediction":prediction,"observed_raw":actual["raw_rows"],
            "delta":actual["raw_rows"]-prediction,"status":"PASS" if ok else "FAIL",
            "reason":"Counts and inherited multiplicity profile agree" if ok else "Observed date/count difference; no cause inferred",
            "multiplicity_profile_reference":profile["multiplicity_histogram"],"profile_basis":"Prior name-key profile trusted only by unchanged whole-file pin"})
    if any(x["status"]=="FAIL" for x in h2):
        emit(repo,packet,"H-J2-replacement-hypothesis.json",{"utc":utc_now(),"hypothesis":"An unread numeric-score/date filter or a different writer may explain count differences; supporting date-only comparisons cannot distinguish them."})
    for q in sorted((repo/"data/processed/matches").glob("*.csv")):
        match=re.fullmatch(r"(PL|BL1|FL1|SA|PD)_(\d{4})\.csv",q.name)
        if not match:continue
        league,year=match.group(1),int(match.group(2))
        if 2012<=year<=2023 or year==2025:
            actual=count_dates(repo,q,"date");inputs.append(actual["identity"])
            comparison={"league":league,"year":year,"source":actual["identity"],"predicted_date_label_rows":predict_counts(historical[league],year),
                "observed_raw_rows":actual["raw_rows"],"observed_valid_dates":actual["valid_rows"],"errors":actual["errors"]}
            comparison["delta"]=comparison["observed_raw_rows"]-comparison["predicted_date_label_rows"]
            if year==2025:
                comparison.update(status="NOT_RUN",reason="Exploratory count observation only; no hypothesis verdict")
                comparison["contributions"]=[{"file":x["identity"],"date_label_2025_rows":x["counts"].get(2025,0)} for x in historical[league] if x["counts"].get(2025,0)]
                h4.append(comparison)
            else:
                comparison.update(status="UNKNOWN",reason="Importer filters numeric scores at101-104; those values are forbidden, so no effective-input count inferred")
                h3.append(comparison)
    snapshots_root=repo/"data/raw/football_data_org/20261108T193117.443560Z"
    manifest,verified=verify_snapshots(snapshots_root)
    pl=next(x for x in verified if x["league"]=="PL" and x["season"]==2025)
    raw=json.loads(Path(pl["identity"]["path"]).read_bytes());projected=project_snapshot(raw);del raw
    provider,bad=flatten_snapshot(projected,"PL")
    if bad or projected["metadata_errors"]:raise ValueError("PL snapshot metadata invalid")
    canonical,errors,pinned=load_metadata(repo/"data/processed/matches/PL_2025.csv","canonical","PL")
    if errors:raise ValueError("PL canonical metadata invalid")
    inputs.append(pinned)
    aliases,_,issues,_=prepare_aliases("PL",canonical,provider)
    hl=simulation("PL",canonical,provider,aliases,pinned)
    hl["status"]="PASS" if not issues and hl["complete"] and hl["zero_self_global"]=="PASS" and hl["per_club_global"]=="PASS" and hl["provider_membership_resolved"]=="PASS" else "FAIL"
    hl["reason"]="Unchanged helper supports PL exact closure" if hl["status"]=="PASS" else "Observed PL closure failure"
    if hl["status"]=="FAIL":emit(repo,packet,"H-L-replacement-hypothesis.json",{"utc":utc_now(),"hypothesis":"Candidate PL aliases or provider/date coverage differences prevent complete exact closure."})
    prior_sims=load_reference(prior/"selected-final/results/simulation.json.gz")
    label_names={};label_locations={}
    for league,label in sorted({(l,label) for l,_,_,label in ABSORBED}):
        sim=next(s for s in prior_sims if s["league"]==league);names=set()
        for plan in sim["plans"]:
            if plan["status"]=="UNIQUE":
                for side in ("home","away"):
                    if plan["proposed_"+side+"_id"]==label:names.add(plan["original_"+side])
        meta,errs,pin=load_metadata(repo/"data/processed/matches"/f"{league}_2025.csv","canonical",league)
        if errs:raise ValueError(f"H-K names metadata invalid league={league}")
        inputs.append(pin)
        observed_names={name for r in meta for name in (r.home,r.away)}
        if len(names)!=1 or not names.issubset(observed_names):raise ValueError(f"Cannot derive label string league={league} id={label}")
        label_names[(league,label)]=next(iter(names))
        label_locations[(league,label)]=[{"line":r.line,"date":r.day} for r in meta if r.home in names or r.away in names]
    hk={"status":"UNKNOWN","reason":"K2 NOT_RUN: naming.py import opens season_maps.json; no behavior or matcher mirror executed",
        "static_gate":naming_import_gate((repo/"src/utils/naming.py").read_text(encoding="utf-8")),
        "pairs":[{"league":l,"provider_id":tid,"absorbed_provider_name":name,"merged_label_id":label,
            "evidence_label_string":label_names[(l,label)],"shared_tokens_diagnostic_only":shared_tokens(name,label_names[(l,label)]),
            "label_locations":label_locations[(l,label)]} for l,tid,name,label in ABSORBED]}
    hm=kickoff_schema(repo,repo/"data/processed/matches/PD_2025.csv")
    if hm["status"]=="PASS":
        raise ValueError("Unexpected kickoff schema: do not proceed without the explicit control/freeze orchestration")
    hm.update({"target_time_cells_read":0,"controls_time_cells_read":0,"four_rows":"UNKNOWN","calibration":"NOT_RUN","PD_reclosure":"NOT_RUN"})
    with (prior/"alias-review-sheet.csv").open(encoding="utf-8",newline="") as stream:
        reader=csv.DictReader(stream);old_fields=reader.fieldnames;sheet=list(reader)
    if len(sheet)!=22:raise ValueError("Inherited alias row count changed")
    by_league={x["league"]:x for x in prior_sims};by_league["PL"]=hl
    for row in sheet:
        name=row["canonical_name"];sim=by_league[row["league"]]
        anchored=sum(p["status"]=="UNIQUE" and name in (p["original_home"],p["original_away"]) for p in sim["plans"])
        row.update(operation="LABEL_SPLIT" if row["league"]=="PD" and int(row["candidate_id"])==81 else "ALIAS_1_TO_1",
            sides_match=int(row["canonical_span_sides"])==int(row["provider_span_sides"]),
            anchored_unique_rows=anchored,evidence_tier="ANCHORED_SIMULATION" if anchored else "SIDE_COUNT_ONLY",
            decision_date="",absorbed_provider_id="80" if row["league"]=="PD" and int(row["candidate_id"])==81 else "")
        if row["status"]!="UNREVIEWED" or row["reviewer"] or row["decision"]:raise ValueError("Alias review state changed")
    fields=old_fields+["operation","sides_match","anchored_unique_rows","evidence_tier","decision_date","absorbed_provider_id"]
    write_new(repo,packet,"alias-review-sheet-v2.csv",csv_text(sheet,fields))
    excluded=[]
    for root,kind in [(repo/"data/historical","historical"),(repo/"data/processed/matches","processed")]:
        for q in sorted(root.rglob("*.csv")):
            direct_hist=q.parent.name in LEAGUES and q.parent.parent==repo/"data/historical"
            exact_primary=q.parent==repo/"data/processed/matches" and re.fullmatch(r"(PL|BL1|FL1|SA|PD)_\d{4}\.csv",q.name)
            if (kind=="historical" and not direct_hist) or (kind=="processed" and not exact_primary):
                excluded.append({"path":str(q),"reason":"Nested historical path outside direct glob" if kind=="historical" else "Outside exact primary L_YYYY comparison: backup/BACKFILL/upcoming"})
    results={"H-J":{"status":"PASS" if all(x["status"]=="PASS" for x in h2) else "FAIL",
        "reason":"J1 no-dedupe writer and all five J2 count/profile predictions match" if all(x["status"]=="PASS" for x in h2) else "A primary J2 prediction failed",
        "J1":gate["H-J1"],"J2":h2,"J3":h3,"J4":h4},
        "H-K":hk,"H-L":hl,"H-M":hm,"manifest":manifest,"snapshot_identities":verified,
        "no_repair_or_approval_applied":True,"score_or_odds_values_accessed":False,"network_requests":0}
    for name,value in {"historical-date-only.json":source_dates,"hash-groups.json":hash_groups,"H-J.json":results["H-J"],
        "H-K.json":hk,"H-L.json":hl,"H-M.json":hm,"input-identities.json":inputs,
        "excluded-paths.json":excluded,"summary.json":{k:({"status":results[k]["status"],"reason":results[k].get("reason","")} if k.startswith("H-") else results[k]) for k in results}}.items():
        emit(repo,packet,"results/"+name,value)
    return {k:results[k]["status"] for k in ("H-J","H-K","H-L","H-M")}

