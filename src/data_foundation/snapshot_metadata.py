"""NEXT-34 metadata only. JSON scores may be parsed but are never accessed."""
from collections import Counter,defaultdict
from datetime import datetime,date,timezone,timedelta
from pathlib import Path
import csv
import gzip
import hashlib
import io
import json
import os
import re
from .audit import load_metadata,season_label
from .identity import normalize_name

LEAGUES=("PL","BL1","FL1","SA","PD")
class IntegrityError(Exception):
    def __init__(self,details):
        self.details=details
        super().__init__("Snapshot integrity failure")
class ScoreAccessError(Exception):
    pass

def get_meta(obj,key,default=None):
    if "score" in key.casefold():
        raise ScoreAccessError("Score field access forbidden")
    return obj.get(key,default)

def project_team(obj):
    if not isinstance(obj,dict):return {"id":None,"name":None,"shortName":None,"tla":None}
    return {k:get_meta(obj,k) for k in ("id","name","shortName","tla")}

def project_snapshot(payload):
    """Do not iterate source match keys/items or copy an unprojected match."""
    if not isinstance(payload,dict):
        return {"resultSet":{},"matches":[],"metadata_errors":["ROOT_NOT_OBJECT"]}
    rs=get_meta(payload,"resultSet")
    result={"resultSet":{k:get_meta(rs,k) for k in ("count","first","last")} if isinstance(rs,dict) else {},
            "matches":[],"metadata_errors":[]}
    matches=get_meta(payload,"matches")
    if matches is None:
        result["metadata_errors"].append("MATCHES_ABSENT")
        return result
    if not isinstance(matches,list):
        result["metadata_errors"].append("MATCHES_NOT_LIST")
        return result
    if not matches:result["metadata_errors"].append("MATCHES_EMPTY")
    for index,match in enumerate(matches):
        if not isinstance(match,dict):
            result["metadata_errors"].append({"type":"MATCH_NOT_OBJECT","index":index});continue
        season=get_meta(match,"season")
        # Only needed season metadata. Winner/other fields deliberately unused.
        record={k:get_meta(match,k) for k in ("id","utcDate","status","matchday")}
        record["season"]={k:get_meta(season,k) for k in ("id","startDate","endDate","currentMatchday")} if isinstance(season,dict) else {}
        record["homeTeam"]=project_team(get_meta(match,"homeTeam"))
        record["awayTeam"]=project_team(get_meta(match,"awayTeam"))
        result["matches"].append(record)
    return result

def identity(path):
    path=Path(path)
    with path.open("rb") as stream:
        a=os.fstat(stream.fileno());h=hashlib.sha256();size=0
        for block in iter(lambda:stream.read(65536),b""):h.update(block);size+=len(block)
        b=os.fstat(stream.fileno())
    if (a.st_size,a.st_mtime_ns)!=(b.st_size,b.st_mtime_ns):raise IntegrityError([{"path":str(path),"type":"CHANGED_HELD_STREAM"}])
    return {"path":str(path.resolve()),"size_bytes":size,"sha256":h.hexdigest()}

def verify_snapshots(root,expected_pairs=None):
    root=Path(root).resolve();manifest_identity=identity(root/"manifest.json")
    manifest=json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    expected_pairs=expected_pairs if expected_pairs is not None else {(l,y) for l in LEAGUES for y in (2024,2025,2026)}
    entries=[x for x in manifest["entries"] if x["http_status"]==200]
    failures=[];verified=[]
    if len(entries)!=len(expected_pairs) or {(x["league"],x["season"]) for x in entries}!=set(expected_pairs):
        failures.append({"type":"BODY_INVENTORY_MISMATCH"})
    for entry in entries:
        path=Path(entry["path"]).resolve()
        if path.parent!=root or path.name!=f"{entry['league']}_{entry['season']}.json":
            failures.append({"type":"BODY_PATH_MISMATCH","path":str(path)});continue
        try:actual=identity(path)
        except OSError as exc:
            failures.append({"type":"BODY_MISSING_OR_UNREADABLE","path":str(path),"error_class":type(exc).__name__});continue
        if any(actual[k]!=entry[k] for k in ("size_bytes","sha256")):
            failures.append({"type":"HASH_OR_SIZE_MISMATCH","expected":{k:entry[k] for k in ("size_bytes","sha256")},"actual":actual})
        verified.append({"league":entry["league"],"season":entry["season"],"identity":actual})
    if failures:raise IntegrityError(failures)
    return manifest_identity,verified

def utc_details(value):
    if not isinstance(value,str) or not value:
        return None,"MISSING_UTC_DATE"
    try:
        d=datetime.fromisoformat(value.replace("Z","+00:00"))
        if d.tzinfo is None:return None,"NAIVE_UTC_DATE"
        u=d.astimezone(timezone.utc)
        return {"date":u.date().isoformat(),"utcDate":value,
                "near_UTC_midnight_1h":u.hour in (0,23),
                "nonzero_supplied_offset":d.utcoffset().total_seconds()!=0},None
    except ValueError:return None,"INVALID_UTC_DATE"

def flatten_snapshot(projected,league):
    rows=[];errors=[]
    for index,m in enumerate(projected["matches"]):
        dt,error=utc_details(m["utcDate"])
        if error:errors.append({"index":index,"id":m["id"],"type":error})
        rows.append({"id":m["id"],"date":dt["date"] if dt else None,"utcDate":m["utcDate"],"status":m["status"],
                     "matchday":m["matchday"],"season":m["season"],"league":league,
                     "home_id":m["homeTeam"]["id"],"away_id":m["awayTeam"]["id"],
                     "home":m["homeTeam"]["name"],"away":m["awayTeam"]["name"],
                     "home_meta":m["homeTeam"],"away_meta":m["awayTeam"],
                     "near_midnight":bool(dt and dt["near_UTC_midnight_1h"])})
    return rows,errors

def valid_id(value):
    return isinstance(value,int) and not isinstance(value,bool) and value>0

def snapshot_check(projected,rows,errors):
    counts=Counter();names=defaultdict(set);inverse=defaultdict(set);ids=Counter();null_teams=[];missing_names=[];pairs=Counter()
    for index,r in enumerate(rows):
        if r["id"] is not None:ids[r["id"]]+=1
        for side in ("home","away"):
            tid=r[side+"_id"];name=r[side]
            if not valid_id(tid):null_teams.append({"index":index,"side":side,"match_id":r["id"]});continue
            counts[tid]+=1
            if isinstance(name,str):
                names[tid].add(name);inverse[name].add(tid)
            if not isinstance(name,str) or not name.strip():missing_names.append({"index":index,"side":side,"id":tid})
        if valid_id(r["home_id"]) and valid_id(r["away_id"]):pairs[(r["home_id"],r["away_id"])]+=1
    n=len(counts);expected=n*(n-1);per_club=2*(n-1)
    duplicate_ids=[{"id":key,"count":v,"utcDates":sorted({str(r["utcDate"]) for r in rows if r["id"]==key})} for key,v in ids.items() if v>1]
    self_rows=[{"id":r["id"],"date":r["date"],"team_id":r["home_id"]} for r in rows if valid_id(r["home_id"]) and r["home_id"]==r["away_id"]]
    problems={"metadata_errors":projected["metadata_errors"]+errors,"null_or_invalid_team_ids":null_teams,
              "missing_team_names":missing_names,"resultSet_count_mismatch":projected["resultSet"].get("count")!=len(rows),
              "missing_match_id_count":sum(r["id"] is None for r in rows),"duplicate_match_ids":duplicate_ids,
              "one_ID_multiple_names":[{"id":k,"names":sorted(v)} for k,v in names.items() if len(v)>1],
              "one_name_multiple_IDs":[{"name":k,"ids":sorted(v)} for k,v in inverse.items() if len(v)>1],"self_fixtures":self_rows,
              "repeated_ordered_ID_pairs":[{"home_id":k[0],"away_id":k[1],"count":v} for k,v in pairs.items() if v>1]}
    schedule_complete=(len(rows)==expected and n>0 and all(v==per_club for v in counts.values()) and not any(problems.values()))
    return {"resultSet_count":projected["resultSet"].get("count"),"projected_len_matches":len(rows),
            "resultSet_len_agree":projected["resultSet"].get("count")==len(rows),"resultSet":projected["resultSet"],
            "distinct_team_IDs":n,"expected_double_round_robin_fixtures":expected,"expected_per_club":per_club,
            "per_club":[{"id":k,"names":sorted(names[k]),"fixtures":v,"delta_vs_expected":v-per_club} for k,v in sorted(counts.items())],
            "statuses":dict(Counter(str(r["status"]) for r in rows)),"date_min":min((r["date"] for r in rows if r["date"]),default=None),
            "date_max":max((r["date"] for r in rows if r["date"]),default=None),"near_midnight":[{"id":r["id"],"utcDate":r["utcDate"]} for r in rows if r["near_midnight"]],
            "schedule_completeness":"COMPLETE_DOUBLE_ROUND_ROBIN_METADATA" if schedule_complete else "INCOMPLETE_OR_INVALID_METADATA",
            "result_completion":"NOT_INFERRED_FROM_SCHEDULE","date_changes":"UNKNOWN single snapshot; duplicate-ID date variants/statuses above only",
            **problems}

def nominal_year(day):
    return int(day[:4]) if day[5:]>="07-01" else int(day[:4])-1

def label_year(path):
    stem=Path(path).stem
    canonical=re.fullmatch(r"(?:PL|BL1|FL1|SA|PD)_(\d{4})",stem)
    if canonical:return int(canonical.group(1))
    explicit=re.search(r"(?:Season_)(\d{4})_(\d{4})",stem,re.I)
    if explicit:return int(explicit.group(1))
    compact=re.search(r"_(\d{2})(\d{2})(?:_latest)?$",stem)
    if compact and int(compact.group(2))==int(compact.group(1))+1:return 2000+int(compact.group(1))
    direct=re.search(r"_(\d{4})$",stem)
    return int(direct.group(1)) if direct else None

def label_check(path,rows):
    year=label_year(path);dates=[r.day for r in rows]
    if year is None:
        return {"path":str(path),"label_year":None,"rows":len(rows),"date_min":min(dates,default=None),"date_max":max(dates,default=None),
                "inside":None,"outside":None,"outside_nominal_windows":{},"status":"LABEL_UNPARSEABLE_OR_UPCOMING"}
    lo,hi=f"{year}-07-01",f"{year+1}-07-01"
    outside=Counter(nominal_year(d) for d in dates if not lo<=d<hi)
    return {"path":str(path),"label_year":year,"nominal_window":[lo,hi],"rows":len(rows),"date_min":min(dates,default=None),"date_max":max(dates,default=None),
            "inside":sum(lo<=d<hi for d in dates),"outside":sum(outside.values()),
            "outside_nominal_windows":{f"{y}/{str(y+1)[-2:]}":v for y,v in sorted(outside.items())},"status":"METADATA_ONLY_NO_ASSIGNMENT_RULE"}

def double_check(rows):
    counts=Counter((r.day,r.home,r.away) for r in rows)
    return {"raw_rows":len(rows),"unique_fixture_keys":len(counts),"multiplicity_histogram":dict(Counter(counts.values())),
            "full_double":bool(counts) and all(v==2 for v in counts.values()),
            "doubled_key_count":sum(v==2 for v in counts.values()),
            "residual_non_doubled":[{"date":k[0],"home":k[1],"away":k[2],"copies":v} for k,v in sorted(counts.items()) if v!=2],
            "duplicated_keys":[{"date":k[0],"home":k[1],"away":k[2],"copies":v} for k,v in sorted(counts.items()) if v>1]}

def catalog(rows):
    aliases=defaultdict(set);names=defaultdict(set)
    for r in rows:
        for side in ("home","away"):
            tid=r[side+"_id"]
            if not valid_id(tid):continue
            if isinstance(r[side],str):names[tid].add(r[side])
            for field in ("name","shortName","tla"):
                text=r[side+"_meta"].get(field)
                if isinstance(text,str) and text:aliases[normalize_name(text)].add(tid)
    return aliases,names

def resolve_row(row,snapshots,aliases,inflated_ids):
    sides={}
    for side in ("home","away"):sides[side]=set(aliases.get(normalize_name(getattr(row,side)),set()))
    target=[s for s in ("home","away") if len(sides[s])==1 and next(iter(sides[s])) in inflated_ids]
    base={"date":row.day,"home":row.home,"away":row.away,"path":row.path,"line":row.line,"exact_status":"UNMATCHED","exact_candidates":[],"plus_minus_one_day_candidates":[]}
    if len(target)!=1:
        base["reason"]="NO_SINGLE_INFLATED_SIDE_OR_NO_TRUSTED_OTHER_SIDE";return base
    changed=target[0];other="away" if changed=="home" else "home"
    if len(sides[other])!=1 or next(iter(sides[other])) in inflated_ids:
        base["reason"]="OTHER_SIDE_UNMATCHED_AMBIGUOUS_OR_INFLATED";return base
    anchor=next(iter(sides[other]))
    candidates=[]
    for r in snapshots:
        if r[other+"_id"]!=anchor or not r["date"]:continue
        distance=abs((date.fromisoformat(r["date"])-date.fromisoformat(row.day)).days)
        if distance<=1:
            candidates.append({"match_id":r["id"],"date":r["date"],"home_id":r["home_id"],"away_id":r["away_id"],"changed_side":changed,"candidate_team_id":r[changed+"_id"],"distance_days":distance})
    exact=[r for r in candidates if r["distance_days"]==0]
    base["exact_candidates"]=exact
    base["exact_status"]="UNIQUE" if len(exact)==1 else "AMBIGUOUS" if len(exact)>1 else "UNMATCHED"
    base["plus_minus_one_day_candidates"]=[r for r in candidates if r["distance_days"]==1]
    base["plus_minus_one_day_status"]="UNIQUE" if len(base["plus_minus_one_day_candidates"])==1 else "AMBIGUOUS" if len(base["plus_minus_one_day_candidates"])>1 else "UNMATCHED"
    return base

def merge_diagnostic(canonical,snapshots):
    aliases,names=catalog(snapshots);raw=Counter();unresolved=Counter();self_rows=[]
    dates=[r.day for r in canonical];lo,hi=min(dates,default=None),max(dates,default=None)
    in_span=[r for r in snapshots if r["date"] and lo<=r["date"]<=hi] if lo else []
    expected=Counter();full=Counter()
    for records,target in [(snapshots,full),(in_span,expected)]:
        for r in records:
            for s in ("home","away"):
                if valid_id(r[s+"_id"]):target[r[s+"_id"]]+=1
    mapped={}
    for r in canonical:
        for s in ("home","away"):
            text=getattr(r,s);ids=aliases.get(normalize_name(text),set())
            mapped[text]=sorted(ids)
            if len(ids)==1:raw[next(iter(ids))]+=1
            else:unresolved[text]+=1
        if r.home==r.away:self_rows.append({"date":r.day,"team":r.home,"path":r.path,"line":r.line})
    absent=[{"id":i,"names":sorted(names[i]),"full_snapshot_fixture_sides":full[i],"canonical_span_fixture_sides":expected[i]}
            for i in sorted(full) if raw[i]==0]
    inflated=[{"id":i,"names":sorted(names[i]),"canonical_fixture_sides":raw[i],"snapshot_span_sides":expected[i],"surplus":raw[i]-expected[i]}
              for i in sorted(raw) if raw[i]>expected[i]]
    suspect={x["id"] for x in absent+inflated};inflated_ids={x["id"] for x in inflated}
    resolutions=[resolve_row(r,snapshots,aliases,inflated_ids) for r in canonical if any(set(mapped.get(t,[]))&inflated_ids for t in (r.home,r.away))]
    pair_counts=Counter((r["home_id"],r["away_id"]) for r in in_span if r["home_id"] in suspect and r["away_id"] in suspect)
    return {"date_span":[lo,hi],"absence_definition":"No exact provider-name/shortName/tla evidence; not guessed club identity",
            "clubs_without_exact_canonical_evidence":absent,"inflated_clubs":inflated,
            "canonical_name_ID_candidates":[{"name":k,"ids":v} for k,v in sorted(mapped.items())],
            "unmatched_or_ambiguous_canonical_names":dict(unresolved),
            "surplus_fixture_sides":sum(x["surplus"] for x in inflated),
            "absent_club_snapshot_span_sides":sum(x["canonical_span_fixture_sides"] for x in absent),
            "self_fixture_raw_rows":self_rows,"self_fixture_unique_count":len({(r["date"],r["team"]) for r in self_rows}),
            "snapshot_ordered_pairs_among_suspected_set":[{"home_id":k[0],"away_id":k[1],"fixtures":v} for k,v in sorted(pair_counts.items())],
            "resolution_counts":dict(Counter(r["exact_status"] for r in resolutions)),
            "plus_minus_one_day_counts":dict(Counter(r.get("plus_minus_one_day_status","NO_TRUSTED_ANCHOR") for r in resolutions)),
            "resolutions":resolutions,
            "hypothesis_status":"UNKNOWN" if unresolved else "PASS" if absent and inflated and self_rows else "FAIL",
            "limit":"Metadata signature only; unresolved exact aliases/date/status changes remain unknown; no mapping applied"}

def coverage_rows(all_csv,snapshot_rows):
    buckets=defaultdict(list)
    for kind,league,path,rows in all_csv:
        source="historical" if kind=="historical" else "processed"
        for r in rows:buckets[(league,nominal_year(r.day),source)].append((r.day,r.home,r.away,None,kind))
    for (league,season),rows in snapshot_rows.items():
        for r in rows:
            if r["date"]:buckets[(league,nominal_year(r["date"]),"snapshot")].append((r["date"],r["home"],r["away"],(r["home_id"],r["away_id"],r["id"]),"snapshot"))
    matrix=[];club_counts=[]
    for league in LEAGUES:
        for year in range(2012,2026):
            for source in ("historical","processed","snapshot"):
                rows=buckets.get((league,year,source),[])
                keys=set((d,h,a) for d,h,a,ids,kind in rows);counts=Counter();kinds=Counter(kind for d,h,a,ids,kind in rows)
                for d,h,a in keys:
                    for team in set((h,a)):counts[team]+=1
                names=set(counts);expected=2*(len(names)-1) if names else None
                matrix.append({"league":league,"nominal_year":year,"source":source,"raw_rows":len(rows),"unique_date_name_fixtures":len(keys),
                               "unique_provider_match_IDs":len({ids[2] for d,h,a,ids,kind in rows if ids and ids[2] is not None}) if source=="snapshot" else None,
                               "date_min":min((x[0] for x in rows),default=None),"date_max":max((x[0] for x in rows),default=None),
                               "distinct_club_names":len(names),"reference_fixtures_per_club":expected,
                               "clubs_deviating_from_reference":sum(n!=expected for n in counts.values()),"source_location_raw_counts":dict(kinds),
                               "existence":"PRESENT" if rows else "ABSENT","authority":"NOT_CHOSEN",
                               "identity_limit":"Snapshot IDs checked in T2; CSV keys remain exact name strings; no cross-source match here"})
                for team,n in sorted(counts.items()):club_counts.append({"league":league,"nominal_year":year,"source":source,"name":team,"fixtures":n,"reference":expected})
    maxima=[]
    for league in LEAGUES:
        for source in ("historical","processed","snapshot"):
            dates=[d for (l,y,s),rr in buckets.items() if l==league and s==source for d,h,a,ids,kind in rr]
            maxima.append({"league":league,"source":source,"min_date":min(dates,default=None),"max_date":max(dates,default=None)})
    gaps=[{"league":l,"nominal_year":y} for l in LEAGUES for y in range(2012,2026)
          if all(not buckets.get((l,y,s)) for s in ("historical","processed","snapshot"))]
    return matrix,club_counts,maxima,gaps

def csv_league(path,source):
    if source=="historical":return path.parent.name if path.parent.name in LEAGUES else None
    first=path.stem.split("_")[0]
    if first in LEAGUES:return first
    backfill=re.fullmatch(r"BACKFILL_(PL|BL1|FL1|SA|PD)_\d{4}",path.stem)
    return backfill.group(1) if backfill else None

def run(root,packet,result_leaf="results"):
    root=Path(root).resolve();packet=Path(packet).resolve()
    if not (packet/"HYPOTHESES.md").is_file():raise ValueError("Hypothesis preregistration required")
    if not result_leaf or Path(result_leaf).name!=result_leaf:raise ValueError("Unsafe result leaf")
    resultdir=packet/result_leaf
    if resultdir.exists():raise FileExistsError("Never overwrite audit")
    snapshot_root=root/"data/raw/football_data_org/20261108T193117.443560Z"
    mid,verified=verify_snapshots(snapshot_root)
    before=json.loads((packet/"snapshot-integrity-before.json").read_text(encoding="utf-8"))
    if mid["sha256"]!=before["manifest"]["sha256"]:raise IntegrityError([{"type":"MANIFEST_CHANGED"}])
    snapshots={};t1=[];t2=[];parse_errors=[]
    for v in verified:
        raw=json.loads(Path(v["identity"]["path"]).read_bytes())
        projected=project_snapshot(raw)
        del raw  # Score-containing parsed object is discarded immediately.
        rows,errors=flatten_snapshot(projected,v["league"])
        snapshots[(v["league"],v["season"])]=rows
        checks=snapshot_check(projected,rows,errors)
        t2.append({"league":v["league"],"file_season":v["season"],**checks})
        t1.append({"league":v["league"],"file_season":v["season"],"identity":v["identity"],
                   "resultSet_count":checks["resultSet_count"],"len_matches":checks["projected_len_matches"],"counts_agree":checks["resultSet_len_agree"]})
        parse_errors += [{"league":v["league"],"file_season":v["season"],"error":x} for x in checks["metadata_errors"]]
    all_csv=[];pins=[];csv_errors=[];labels=[]
    for source,base in [("historical",root/"data/historical"),("processed",root/"data/processed/matches")]:
        for path in sorted(base.rglob("*.csv")):
            league=csv_league(path,source)
            if league is None:
                csv_errors.append({"path":str(path),"type":"LEAGUE_NOT_IDENTIFIED"});league="UNKNOWN"
            kind=source if source=="historical" else "processed_primary" if path.parent==base else "processed_nested_backup_location"
            try:rows,errors,pin=load_metadata(path,"historical" if source=="historical" else "canonical",league)
            except OSError as exc:
                csv_errors.append({"path":str(path),"type":"UNREADABLE","error_class":type(exc).__name__});continue
            all_csv.append((kind,league,path,rows));pins.append(pin);csv_errors+=errors
            labels.append({"source_location":kind,"league":league,**label_check(path,rows)})
    ha=[]
    for league in LEAGUES:
        candidates=[(p,rr) for kind,l,p,rr in all_csv if kind=="historical" and l==league and label_year(p)==2024]
        primary=next(((p,rr) for kind,l,p,rr in all_csv if kind=="processed_primary" and l==league and p.name==f"{league}_2024.csv"),None)
        historical=[identity(p) for p,rr in candidates]
        check=double_check(primary[1]) if primary else None
        pair_identical=len(historical)==2 and historical[0]["sha256"]==historical[1]["sha256"]
        controlled=(len(historical)==1 and check and not check["doubled_key_count"]) if league=="FL1" else pair_identical and check and check["doubled_key_count"]>0
        ha.append({"league":league,"historical_files":historical,"pair_byte_identical":pair_identical,
                   "processed_2024":check,"hypothesis_status":"PASS" if controlled else "UNKNOWN" if not primary else "FAIL"})
    residual_pd=next((double_check(rr) for kind,l,p,rr in all_csv if kind=="processed_primary" and p.name=="PD_2025.csv"),None)
    hb=[]
    for league in LEAGUES:
        primary=next((rr for kind,l,p,rr in all_csv if kind=="processed_primary" and p.name==f"{league}_2025.csv"),[])
        hb.append({"league":league,**merge_diagnostic(primary,snapshots[(league,2025)])})
    special_hist=next((rr for kind,l,p,rr in all_csv if kind=="historical" and p.name=="Bundesliga_Season_2018_2019.csv"),[])
    special_proc=next((rr for kind,l,p,rr in all_csv if kind=="processed_primary" and p.name=="BL1_2017.csv"),[])
    a={(x.day,x.home,x.away) for x in special_hist};b={(x.day,x.home,x.away) for x in special_proc}
    overlap={"historical_rows":len(special_hist),"processed_rows":len(special_proc),"exact_raw_name_date_overlap":len(a&b),
             "exact_normalized_date_name_overlap":len({(r.day,normalize_name(r.home),normalize_name(r.away)) for r in special_hist}&{(r.day,normalize_name(r.home),normalize_name(r.away)) for r in special_proc}),
             "date_multiset_equal":Counter(r.day for r in special_hist)==Counter(r.day for r in special_proc),
             "historical_unique":len(a),"processed_unique":len(b),"historical_only":len(a-b),"processed_only":len(b-a),
             "historical_date_min":min((x.day for x in special_hist),default=None),"historical_date_max":max((x.day for x in special_hist),default=None),
             "limit":"Raw exact names only; no guessed aliases"}
    matrix,clubs,maxima,gaps=coverage_rows(all_csv,snapshots)
    outputs={"T1-integrity-counts.json":{"manifest":mid,"files":t1},
             "T2-snapshot-self-validation.json":t2,"H-A-doubling.json":{"leagues":ha,"PD_2025_residuals":residual_pd},
             "H-B-club-merge.json":hb,"H-C-file-windows.json":{"files":labels,"BL1_overlap":overlap},
             "T4-coverage.json":matrix,"T4-club-counts.json":clubs,"T4-source-maxima.json":maxima,
             "H-D-gaps.json":{"empty_all_source_cells":gaps,"BL1_2018_19":[x for x in matrix if x["league"]=="BL1" and x["nominal_year"]==2018]},
             "csv-input-identities.json":pins,"metadata-errors.json":{"snapshot":parse_errors,"CSV":csv_errors}}
    summary={"integrity":"PASS","bodies":len(verified),"snapshot_match_counts":[{"league":x["league"],"season":x["file_season"],"n":x["len_matches"],"bytes":x["identity"]["size_bytes"]} for x in t1],
             "CSV_files":len(pins),"primary_processed_files":sum(kind=="processed_primary" for kind,l,p,rr in all_csv),
             "nested_processed_backup_files":sum(kind=="processed_nested_backup_location" for kind,l,p,rr in all_csv),
             "H-A":[{"league":x["league"],"status":x["hypothesis_status"]} for x in ha],
             "H-B":[{"league":x["league"],"status":x["hypothesis_status"],"unresolved_names":len(x["unmatched_or_ambiguous_canonical_names"]),"resolution":x["resolution_counts"]} for x in hb],
             "H-C":"PASS" if any(x["outside"] for x in labels if x["outside"] is not None) else "UNKNOWN",
             "H-D":"UNKNOWN" if any(x["type"]=="LEAGUE_NOT_IDENTIFIED" for x in csv_errors) else "PASS" if gaps else "FAIL","empty_cells":len(gaps),"BL1_2018_19_any_source":any(x["unique_date_name_fixtures"] for x in matrix if x["league"]=="BL1" and x["nominal_year"]==2018),
             "snapshot_metadata_errors":len(parse_errors),"CSV_metadata_errors":len(csv_errors),"BL1_overlap":overlap,
             "score_access":"NONE; json.loads memory materialization accepted debt; immediate whitelist projection"}
    outputs["summary.json"]=summary
    csvout=io.StringIO(newline="");writer=csv.DictWriter(csvout,fieldnames=["league","nominal_year","source","raw_rows","unique_date_name_fixtures","date_min","date_max","distinct_club_names","reference_fixtures_per_club","clubs_deviating_from_reference","existence"])
    writer.writeheader()
    for row in matrix:writer.writerow({k:row[k] for k in writer.fieldnames})
    encoded={name:(json.dumps(value,ensure_ascii=False,separators=(",",":"))+"\n").encode() for name,value in outputs.items()}
    encoded["T4-coverage.csv"]=csvout.getvalue().encode()
    # Preserve all earlier outputs; lossless selected tables reduce evidence duplication.
    encoded={name+".gz" if name.endswith(".json") and len(body)>20000 else name:
             gzip.compress(body,mtime=0) if name.endswith(".json") and len(body)>20000 else body for name,body in encoded.items()}
    current=sum(p.stat().st_size for p in packet.rglob("*") if p.is_file())
    if current+sum(map(len,encoded.values()))>4_200_000:raise ValueError("Packet cap guard before result writes")
    resultdir.mkdir()
    for name,body in encoded.items():
        with (resultdir/name).open("xb") as out:out.write(body)
    return summary
