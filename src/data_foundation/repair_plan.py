"""NEXT-35 metadata-only candidate plan; never writes repaired data."""
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
import csv
import gzip
import io
import json
import re
import subprocess
from .audit import load_metadata
from .snapshot_metadata import (
    normalize_name, identity, verify_snapshots, project_snapshot,
    flatten_snapshot, catalog, merge_diagnostic, double_check, label_year,
)

LEAGUES = ("PL", "BL1", "FL1", "SA", "PD")
CANDIDATES = (
    ("PL","NEWCASTLE UNITED",67),("PL","BRIGHTON",397),
    ("PL","MANCHESTER CITY",65),("PL","NOTTINGHAM FOREST",351),("PL","MANCHESTER UNITED",66),
    ("BL1","BAYERN MUNICH",5),("BL1","WERDER BREMEN",12),
    ("BL1","BAYER LEVERKUSEN",3),("BL1","TSG HOFFENHEIM",2),
    ("BL1","FC ST. PAULI",20),("BL1","MAINZ 05",15),("BL1","FC KOLN",1),
    ("FL1","LENS",546),("FL1","LYON",523),("FL1","AS MONACO",548),
    ("FL1","METZ",545),("FL1","ANGERS",532),("FL1","PARIS SAINT GERMAIN",524),
    ("SA","COMO",7397),
    ("PD","BARCELONA",81),("PD","CELTA VIGO",558),("PD","SEVILLA",559),
)
GROUPS = {"BL1":{16:{16,28,44}}, "FL1":{1045:{1045,529}},
          "SA":{98:{98,487}}, "PD":{86:{86,78,90,92},81:{81,80}}}
PACKET_NAME = "NEXT-35-repair-plan-audit-01"

def utc_now():
    return datetime.now(timezone.utc).isoformat()

def write_new(repo, packet, leaf, body):
    repo,packet = Path(repo).resolve(),Path(packet).resolve()
    if not packet.is_relative_to(repo/"tmp") or packet.name != PACKET_NAME:
        raise ValueError("Only the NEXT-35 packet under tmp is writable")
    leaf=Path(leaf)
    if leaf.is_absolute() or ".." in leaf.parts:
        raise ValueError("Output path traversal")
    destination=(packet/leaf).resolve()
    if not destination.is_relative_to(packet) or destination==packet:
        raise ValueError("Output escapes packet")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Existing output must remain unchanged")
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(body if isinstance(body,bytes) else body.encode("utf-8"))
    return identity(destination)

def emit(repo,packet,name,obj):
    raw=json.dumps(obj,ensure_ascii=False,separators=(",",":")).encode("utf-8")
    if len(raw)>25000:
        name += ".gz"
        raw=gzip.compress(raw,mtime=0)
    return write_new(repo,packet,name,raw)

def csv_text(rows,fields):
    out=io.StringIO(newline="")
    writer=csv.DictWriter(out,fieldnames=fields,extrasaction="raise")
    writer.writeheader();writer.writerows(rows)
    return out.getvalue()

def validate_pairs(pairs):
    pairs=tuple(tuple(p) for p in pairs)
    if len(pairs)!=len(set(pairs)) or any(p not in CANDIDATES for p in pairs):
        raise ValueError("Extra or duplicate candidate pair refused")
    names=defaultdict(set);ids=defaultdict(set)
    for league,name,tid in pairs:
        names[(league,normalize_name(name))].add(tid);ids[(league,tid)].add(name)
    if any(len(v)!=1 for v in names.values()) or any(len(v)!=1 for v in ids.values()):
        raise ValueError("Candidate mapping collision")
    return pairs

def teams_from(rows):
    teams=defaultdict(set)
    for r in rows:
        for side in ("home","away"):
            m=r[side+"_meta"]
            teams[r[side+"_id"]].add(tuple(m.get(k) for k in ("name","shortName","tla")))
    return teams

def prepare_aliases(league, canonical, provider, pairs=CANDIDATES):
    pairs=validate_pairs(pairs)
    exact,_=catalog(provider);teams=teams_from(provider)
    aliases={k:set(v) for k,v in exact.items()};issues=[];sheet=[]
    days=[r.day for r in canonical];lo,hi=min(days,default=None),max(days,default=None)
    span=[r for r in provider if r["date"] and lo and lo<=r["date"]<=hi]
    expected=Counter(t for r in span for t in (r["home_id"],r["away_id"]))
    canonical_counts=Counter(t for r in canonical for t in (r.home,r.away))
    for l,name,tid in pairs:
        if l!=league:continue
        variants=teams.get(tid,set())
        if len(variants)!=1:
            issues.append({"type":"CANDIDATE_ID_MISSING_OR_VARIANT","league":l,"name":name,"id":tid})
        values=next(iter(variants)) if len(variants)==1 else (None,None,None)
        norm=normalize_name(name);other=set(exact.get(norm,set()))-{tid}
        if other:issues.append({"type":"CANDIDATE_MATCHES_OTHER_ID","league":l,"name":name,"id":tid,"other_ids":sorted(other)})
        aliases.setdefault(norm,set()).add(tid)
        sheet.append({"league":l,"canonical_name":name,"candidate_id":tid,
            **dict(zip(("provider_name","provider_shortName","provider_tla"),values)),
            **{key:bool(value and normalize_name(value)==norm) for key,value in zip(("exact_name","exact_shortName","exact_tla"),values)},
            "canonical_span_sides":canonical_counts[name],"provider_span_sides":expected[tid],
            "status":"UNREVIEWED","reviewer":"","decision":""})
    for norm,values in aliases.items():
        if len(values)>1:issues.append({"type":"NAME_MULTIPLE_IDS","league":league,"normalized":norm,"ids":sorted(values)})
    return aliases,sheet,issues,span

def candidate_ids(name,aliases,groups):
    ids=set(aliases.get(normalize_name(name),set()))
    expanded=set()
    for tid in ids:expanded.update(groups.get(tid,{tid}))
    return ids,expanded

def row_plan(row,provider,aliases,groups,file_pin):
    original={s:set(aliases.get(normalize_name(getattr(row,s)),set())) for s in ("home","away")}
    affected=any(ids & set(groups) for ids in original.values())
    home=candidate_ids(row.home,aliases,groups)[1];away=candidate_ids(row.away,aliases,groups)[1]
    eligible=[r for r in provider if r["home_id"] in home and r["away_id"] in away]
    exact=[r for r in eligible if r["date"]==row.day]
    adjacent=[r for r in eligible if r["date"] and abs((date.fromisoformat(r["date"])-date.fromisoformat(row.day)).days)==1]
    # An ambiguous name mapping is itself unresolved, even if a fixture happens
    # to narrow its candidate set. Never silently adopt one of those name IDs.
    name_ambiguous=any(len(ids)>1 for ids in original.values())
    status="AMBIGUOUS" if name_ambiguous else "UNIQUE" if len(exact)==1 else "AMBIGUOUS" if len(exact)>1 else "UNMATCHED"
    merged_sides=sum(bool(ids & set(groups)) for ids in original.values())
    method="GROUP_RESTRICTED" if merged_sides==2 else "EXACT_OTHER_SIDE_ANCHOR" if affected else "EXACT_UNAFFECTED_CHECK"
    return {"source_sha256":file_pin["sha256"],"line":row.line,"date":row.day,
        "original_home":row.home,"original_away":row.away,"affected":affected,"method":method,"status":status,
        "proposed_home_id":exact[0]["home_id"] if status=="UNIQUE" else None,
        "proposed_away_id":exact[0]["away_id"] if status=="UNIQUE" else None,
        "provider_match_id":exact[0]["id"] if status=="UNIQUE" else None,
        "provider_status":exact[0]["status"] if status=="UNIQUE" else None,
        "candidate_fixture_ids":[r["id"] for r in exact],
        "adjacent_fixture_ids":[r["id"] for r in adjacent],
        "near_midnight_candidates":sum(bool(r["near_midnight"]) for r in exact+adjacent),
        "candidate_authority":"UNREVIEWED","name_mapping_ambiguous":name_ambiguous}

def simulation(league,canonical,provider,aliases,file_pin):
    groups=GROUPS.get(league,{})
    plans=[row_plan(r,provider,aliases,groups,file_pin) for r in canonical]
    days=[r.day for r in canonical];lo,hi=min(days,default=None),max(days,default=None)
    span=[r for r in provider if r["date"] and lo and lo<=r["date"]<=hi]
    expected=Counter(t for r in span for t in (r["home_id"],r["away_id"]))
    before=Counter()
    for row in canonical:
        for name in (row.home,row.away):
            ids=aliases.get(normalize_name(name),set())
            if len(ids)==1:before[next(iter(ids))]+=1
    he=[]
    for label,expanded in groups.items():
        absorbed=expanded-{label};surplus=before[label]-expected[label];absent=sum(expected[t] for t in absorbed)
        he.append({"label_id":label,"absorbed_ids":sorted(absorbed),"canonical_label_sides":before[label],
            "provider_label_span_sides":expected[label],"surplus":surplus,"absorbed_span_sides":absent,
            "status":"PASS" if surplus==absent else "FAIL"})
    unique=[r for r in plans if r["status"]=="UNIQUE"]
    unresolved=[r for r in plans if r["status"]!="UNIQUE"]
    after=Counter(t for r in unique for t in (r["proposed_home_id"],r["proposed_away_id"]))
    counts=[{"id":t,"provider_span_sides":expected[t],"uniquely_simulated_sides":after[t],"delta":after[t]-expected[t]}
            for t in sorted(set(expected)|set(after))]
    membership={(r["date"],r["home_id"],r["away_id"]) for r in provider}
    exceptions=[r for r in unique if (r["date"],r["proposed_home_id"],r["proposed_away_id"]) not in membership]
    self_rows=[r for r in unique if r["proposed_home_id"]==r["proposed_away_id"]]
    complete=not unresolved
    return {"league":league,"canonical_span":[lo,hi],"H-E_groups":he,
        "H-E":"PASS" if all(r["status"]=="PASS" for r in he) else "FAIL",
        "affected_counts":dict(Counter(r["status"] for r in plans if r["affected"])),
        "whole_table_counts":dict(Counter(r["status"] for r in plans)),
        "provider_status_counts_resolved":dict(Counter(str(r["provider_status"]) for r in unique)),
        "resolved_self_rows":self_rows,"resolved_nonprovider_exceptions":exceptions,
        "zero_self_global":"UNKNOWN" if unresolved else "PASS" if not self_rows else "FAIL",
        "provider_membership_resolved":"PASS" if not exceptions else "FAIL",
        "per_club_global":"UNKNOWN" if unresolved else "PASS" if all(r["delta"]==0 for r in counts) else "FAIL",
        "per_club":counts,"unresolved":unresolved,"plans":plans,"complete":complete}

def pair_invariant(rows):
    teams={t for r in rows for t in (r.home,r.away)}
    counts=Counter((r.home,r.away) for r in rows)
    sides=Counter(t for r in rows for t in (r.home,r.away))
    expected={(h,a) for h in teams for a in teams if h!=a}
    actual=set(counts)
    missing=sorted(expected-actual)
    repeated=[{"home":h,"away":a,"copies":n} for (h,a),n in sorted(counts.items()) if n>1]
    selfs=[{"date":r.day,"home":r.home,"line":r.line} for r in rows if r.home==r.away]
    ok=bool(teams) and actual==expected and not repeated and not selfs and all(v==2*(len(teams)-1) for v in sides.values())
    return {"status":"PASS" if ok else "FAIL","rows":len(rows),"unique_metadata_keys":len({r.key() for r in rows}),
        "teams":len(teams),"expected_rows":len(teams)*(len(teams)-1),"missing_pairs":missing,
        "repeated_pairs":repeated,"self_rows":selfs,"sides":dict(sorted(sides.items())),
        "date_min":min((r.day for r in rows),default=None),"date_max":max((r.day for r in rows),default=None)}

def cutoff_label(day,month):
    parsed=date.fromisoformat(day)
    return parsed.year if parsed.month>=month else parsed.year-1

def checked_git(argv):
    if not argv or argv[0]!="git" or len(argv)<2 or argv[1] not in ("ls-files","status","log"):
        raise ValueError("Git command not allowlisted")
    command=argv[1];options=argv[2:argv.index("--")] if "--" in argv else argv[2:]
    if command=="status":
        if options!=["--porcelain","--untracked-files=all"]:raise ValueError("Git status options refused")
    elif command=="ls-files":
        if options:raise ValueError("Git ls-files options refused")
    else:
        for option in options:
            if option not in ("--follow","--numstat","-1") and not re.fullmatch(r"--format=[%A-Za-z0-9%x\t :_-]+",option):
                raise ValueError("Git log option refused")
        if not any(x.startswith("--format=") and "%H" in x and "%cI" in x and "%s" in x for x in options):
            raise ValueError("Git log must be metadata-only with hash/date/subject")
    return argv

def git_collect(repo,argv,runner=subprocess.run):
    checked_git(argv);start=utc_now()
    try:
        result=runner(argv,cwd=repo,capture_output=True,timeout=25)
    except subprocess.TimeoutExpired as exc:
        return {"exact_command":argv,"utc_start":start,"utc_completion":utc_now(),
            "exit_code":124,"stdout":(exc.stdout or b"").decode("utf-8",errors="replace"),
            "stderr":(exc.stderr or b"").decode("utf-8",errors="replace"),"result":"TIMEOUT_RETAINED"}
    return {"exact_command":argv,"utc_start":start,"utc_completion":utc_now(),
        "exit_code":result.returncode,"stdout":result.stdout.decode("utf-8",errors="replace"),
        "stderr":result.stderr.decode("utf-8",errors="replace")}

def timestamp_collect(path,stat_func=None):
    path=Path(path);st=stat_func(path) if stat_func else path.stat()
    return {"path":str(path.resolve()),"created_UTC_windows":datetime.fromtimestamp(st.st_ctime,timezone.utc).isoformat(),
            "modified_UTC":datetime.fromtimestamp(st.st_mtime,timezone.utc).isoformat(),
            "limitation":"Filesystem creation/copy time is not logical provenance or actor evidence"}

def run(repo,packet,output_leaf=""):
    repo=Path(repo).resolve();packet=Path(packet).resolve()
    if output_leaf and not re.fullmatch(r"[a-z0-9-]+",output_leaf):
        raise ValueError("Unsafe output leaf")
    prefix=output_leaf+"/" if output_leaf else ""
    prereg=identity(packet/"HYPOTHESES.md")
    locked=json.loads((packet/"preregistration.json").read_text(encoding="utf-8"))["hypotheses"]
    if prereg["sha256"]!=locked["sha256"]:raise ValueError("Preregistration changed")
    snapshot_root=repo/"data/raw/football_data_org/20261108T193117.443560Z"
    mid,verified=verify_snapshots(snapshot_root)  # all fifteen hashes before any body parse
    sources={};csvpins={};errors=[];snapshots={}
    for v in verified:
        raw=json.loads(Path(v["identity"]["path"]).read_bytes())
        projected=project_snapshot(raw)
        del raw
        rows,bad=flatten_snapshot(projected,v["league"])
        if projected["metadata_errors"] or bad:errors.append({"snapshot":v["identity"]["path"],"errors":projected["metadata_errors"]+bad})
        snapshots[(v["league"],v["season"])]=rows
    def csv_rows(path,source,league):
        path=Path(path)
        if path not in sources:
            rows,bad,pinned=load_metadata(path,source,league)
            sources[path]=rows;csvpins[path]=pinned;errors.extend(bad)
        return sources[path]
    primary=repo/"data/processed/matches";backup=primary/"backup_pre_audit"
    canonical={l:csv_rows(primary/f"{l}_2025.csv","canonical",l) for l in LEAGUES}
    alias_sheet=[];absent_sheet=[];alias_issues=[];simulations=[]
    for league in LEAGUES:
        aliases,sheet,bad,span=prepare_aliases(league,canonical[league],snapshots[(league,2025)])
        alias_sheet.extend(sheet);alias_issues.extend(bad)
        counts=Counter(t for r in span for t in (r["home_id"],r["away_id"]))
        teams=teams_from(snapshots[(league,2025)])
        for label,group in GROUPS.get(league,{}).items():
            for tid in sorted(group-{label}):
                absent_sheet.append({"league":league,"candidate_absorbed_id":tid,"merged_label_id":label,
                    "provider_names":sorted({v[0] for v in teams.get(tid,set()) if v[0]}),
                    "provider_span_sides":counts[tid],"status":"UNREVIEWED","reviewer":"","decision":""})
        if league!="PL":
            simulations.append(simulation(league,canonical[league],snapshots[(league,2025)],aliases,csvpins[primary/f"{league}_2025.csv"]))
    if alias_issues:errors.extend(alias_issues)
    plans=[r for s in simulations for r in s["plans"] if r["affected"]]
    fields=["source_sha256","line","date","original_home","original_away","proposed_home_id","proposed_away_id","method","status","provider_match_id","provider_status","candidate_fixture_ids","adjacent_fixture_ids","candidate_authority"]
    csvplans=[{k:json.dumps(r[k]) if isinstance(r[k],list) else r[k] for k in fields} for r in plans]
    write_new(repo,packet,prefix+"alias-review-sheet.csv",csv_text(alias_sheet,list(alias_sheet[0])))
    write_new(repo,packet,prefix+"absent-club-review-sheet.csv",csv_text([{**r,"provider_names":"; ".join(r["provider_names"])} for r in absent_sheet],list(absent_sheet[0])))
    write_new(repo,packet,prefix+"repair-plan-simulation.csv",csv_text(csvplans,fields))
    collision=[]
    pd=next(s for s in simulations if s["league"]=="PD");buckets=defaultdict(list)
    for row in pd["plans"]:buckets[(row["date"],row["original_home"],row["original_away"])].append(row)
    for key,rows in buckets.items():
        if len(rows)>1:
            assigned=[r["provider_match_id"] for r in rows if r["status"]=="UNIQUE"]
            verdict="UNKNOWN" if len(assigned)!=len(rows) else "PASS" if len(set(assigned))==len(rows) else "FAIL"
            collision.append({"date":key[0],"home":key[1],"away":key[2],"rows":rows,
                "candidate_ID_union":sorted({v for r in rows for v in r["candidate_fixture_ids"]}),"H-F":verdict})
    sa_provider=snapshots[("SA",2024)];dates=[r["date"] for r in sa_provider if r["date"]]
    season_cases=[];spill_refs=[]
    def case(name,league,rows):
        season_cases.append({"case":name,"league":league,**pair_invariant(rows)})
    for league in ("BL1","PD","PL","SA"):
        a=csv_rows(primary/f"{league}_2019.csv","canonical",league)
        b=csv_rows(primary/f"{league}_2020.csv","canonical",league)
        spill=[r for r in b if r.day<"2020-09-01"];remain=[r for r in b if r.day>="2020-09-01"]
        case("processed2019_before_spill",league,a);case("processed2019_with_2020_spill",league,a+spill)
        case("processed2020_without_spill",league,remain)
        hist=sorted(p for p in (repo/"data/historical"/league).glob("*.csv") if label_year(p)==2019)
        for path in hist:case("historical2019_20:"+path.name,league,csv_rows(path,"historical",league))
        spill_refs.extend({"case":"2019_20_overrun","league":league,"path":r.path,"line":r.line,"date":r.day,"home":r.home,"away":r.away} for r in spill)
    for league in ("FL1","PD","SA"):
        a=csv_rows(primary/f"{league}_2022.csv","canonical",league);b=csv_rows(primary/f"{league}_2023.csv","canonical",league)
        spill=[r for r in b if "2023-06-01"<=r.day<"2023-07-01"];remain=[r for r in b if not "2023-06-01"<=r.day<"2023-07-01"]
        case("processed2022_before_June2023",league,a);case("processed2022_with_June2023",league,a+spill);case("processed2023_without_June2023",league,remain)
        spill_refs.extend({"case":"June2023","league":league,"path":r.path,"line":r.line,"date":r.day,"home":r.home,"away":r.away} for r in spill)
    sa24=csv_rows(primary/"SA_2024.csv","canonical","SA")
    singleton=[r for r in sa24 if r.day=="2024-06-02" and r.home=="ATALANTA" and r.away=="FIORENTINA"]
    sa23=csv_rows(primary/"SA_2023.csv","canonical","SA")
    # Earlier named June2023 spills belong to the separately tested preceding
    # campaign. Remove that known diagnostic bucket before testing June2024.
    case("SA2023_with_June2024_singleton","SA",[r for r in sa23 if not "2023-06-01"<=r.day<"2023-07-01"]+singleton)
    case("SA2024_without_June2024_singleton","SA",[r for r in sa24 if r not in singleton])
    spill_refs.extend({"case":"SA2024_singleton","league":"SA","path":r.path,"line":r.line,"date":r.day,"home":r.home,"away":r.away} for r in singleton)
    hh=[];backup2025=[];custody_targets=[]
    for league in LEAGUES:
        a=csv_rows(primary/f"{league}_2024.csv","canonical",league);q=backup/f"{league}_2024.csv"
        if q.exists():
            b=csv_rows(q,"canonical",league);cb=double_check(b);ca=double_check(a)
            ok=all(k==1 for k in cb["multiplicity_histogram"])
            hh.append({"league":league,"backup":cb,"primary":ca,"H-H":"PASS" if ok and ca["doubled_key_count"]>0 else "FAIL"})
        else:hh.append({"league":league,"H-H":"UNKNOWN","reason":"BACKUP_MISSING"})
        for q in [primary/f"{league}_2024.csv",backup/f"{league}_2024.csv"]:
            if q.exists():custody_targets.append(q)
        for q in sorted((repo/"data/historical"/league).glob("*.csv")):
            if label_year(q)==2024:custody_targets.append(q)
        q=backup/f"{league}_2025.csv"
        if q.exists():
            rr=csv_rows(q,"canonical",league)
            backup2025.append({"league":league,"path":str(q),"H-B_exact_metrics":merge_diagnostic(rr,snapshots[(league,2025)])})
        else:backup2025.append({"league":league,"reason":"BACKUP_MISSING","status":"UNKNOWN"})
    # Parse the literal cutoff; never import the naming module.
    naming=repo/"src/utils/naming.py";cutoff=re.search(r"^EUROPEAN_SEASON_CUTOFF_MONTH\s*=\s*(\d+)",naming.read_text(encoding="utf-8"),re.M)
    if not cutoff:raise ValueError("Missing literal cutoff")
    month=int(cutoff.group(1));cutoff_checks=[];unlabelled=[]
    for q in sorted(primary.glob("*.csv")):
        league=q.stem.split("_")[0]
        if league not in LEAGUES:continue
        year=label_year(q)
        if year is None:unlabelled.append(str(q));continue
        rows=csv_rows(q,"canonical",league)
        for r in rows:
            if not f"{year}-07-01"<=r.day<f"{year+1}-07-01":
                predicted=cutoff_label(r.day,month)
                cutoff_checks.append({"path":str(q),"source_sha256":csvpins[q]["sha256"],"line":r.line,"date":r.day,
                    "home":r.home,"away":r.away,"filename_year":year,"cutoff_year":predicted,"matches":predicted==year})
    custody=[];git_receipts=[]
    for path in sorted(set(custody_targets)):
        rel=path.relative_to(repo).as_posix()
        tracked=git_collect(repo,["git","ls-files","--",rel]);history=git_collect(repo,["git","log","--follow","--format=COMMIT%x09%H%x09%cI%x09%s","--numstat","--",rel])
        git_receipts.extend([tracked,history])
        for receipt in (tracked,history):
            emit(repo,packet,prefix+"git-actions/G"+str(len(list((packet/output_leaf/"git-actions").glob("*.json")))+1).zfill(3)+".json",receipt)
            if receipt["exit_code"]:
                errors.append({"type":"GIT_METADATA_FAILURE","command":receipt["exact_command"],"exit_code":receipt["exit_code"]})
        custody.append({"identity":identity(path),"timestamps":timestamp_collect(path),
            "tracked":bool(tracked["stdout"].strip()),"history_available":bool(history["stdout"].strip()),
            "git_receipt_indices":[len(git_receipts)-2,len(git_receipts)-1]})
    results={"manifest":mid,"snapshot_identities":verified,"alias_pairs":len(alias_sheet),"absent_IDs":len(absent_sheet),
        "alias_issues":alias_issues,"H-E":{s["league"]:s["H-E"] for s in simulations},
        "H-F":"UNKNOWN" if any(x["H-F"]=="UNKNOWN" for x in collision) else "FAIL" if any(x["H-F"]=="FAIL" for x in collision) else "PASS" if collision else "UNKNOWN",
        "H-G":"FAIL" if any(x["status"]=="FAIL" for x in season_cases if "before" not in x["case"]) else "PASS",
        "H-H":"FAIL" if any(x["H-H"]=="FAIL" for x in hh) else "UNKNOWN" if any(x["H-H"]=="UNKNOWN" for x in hh) else "PASS",
        "H-I":"FAIL" if any(not x["matches"] for x in cutoff_checks) else "PASS" if cutoff_checks else "UNKNOWN",
        "metadata_errors":errors,"scores":"NOT_ACCESSED_RETURNED_LOGGED; json.loads materialization accepted",
        "adoption":"NONE","network_requests":0}
    outputs={"simulation.json":simulations,"collision.json":collision,
        "SA2024-singleton-span.json":{"singleton_rows":len(singleton),"date":"2024-06-02","provider2024_span":[min(dates),max(dates)],"inside":min(dates)<="2024-06-02"<=max(dates)},
        "season-cases.json":season_cases,"spill-row-references.json":spill_refs,"custody2024.json":custody,
        "custody-git-actions.json":git_receipts,"H-H-2024.json":hh,"backup2025-metrics.json":backup2025,
        "cutoff-checks.json":{"source":identity(naming),"literal_month":month,"unlabelled_files":unlabelled,"checks":cutoff_checks},
        "input-identities.json":list(csvpins.values()),"summary.json":results}
    for name,obj in outputs.items():emit(repo,packet,prefix+"results/"+name,obj)
    return results

