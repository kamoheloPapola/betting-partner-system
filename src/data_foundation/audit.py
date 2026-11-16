"""Read-only metadata audit: only names/dates decoded, never any score cell."""
from dataclasses import dataclass
from pathlib import Path
from collections import Counter,defaultdict
from datetime import date,datetime
import csv
import hashlib
import io
import json
import os
import re
from .identity import normalize_name

LEAGUES=("PL","BL1","FL1","SA","PD")
class MetadataError(Exception):
    pass
class SensitiveMetadata(Exception):
    pass
class PacketLimit(Exception):
    pass

@dataclass(frozen=True)
class MetaFixture:
    league:str
    source:str
    season_label:str
    path:str
    line:int
    day:str
    home:str
    away:str
    def key(self):
        return self.day,self.home,self.away
    def location(self):
        return {"path":self.path,"line":self.line}

def held_pin(path):
    with Path(path).open("rb") as stream:
        before=os.fstat(stream.fileno());h=hashlib.sha256();size=0
        for chunk in iter(lambda:stream.read(65536),b""):h.update(chunk);size+=len(chunk)
        after=os.fstat(stream.fileno())
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
        raise MetadataError("Input changed while held")
    return {"path":str(Path(path).resolve()),"size_bytes":size,"sha256":h.hexdigest()}

def project_metadata(raw,indices):
    # Byte-column projection follows the existing isolated project() pattern;
    # only selected name/date cells are ever UTF-8 decoded.
    out={};buf=bytearray();column=0;quoted=False;i=0
    def save():
        if column in indices:
            out[indices[column]]=buf.decode("utf-8-sig")
    while i<len(raw):
        value=raw[i]
        if value==34:
            if quoted and i+1<len(raw) and raw[i+1]==34:
                if column in indices:buf.append(34)
                i+=2;continue
            quoted=not quoted
        elif value==44 and not quoted:
            save();buf=bytearray();column+=1
        elif value in (10,13) and not quoted:
            break
        elif column in indices:
            buf.append(value)
        i+=1
    if quoted:raise MetadataError("Quoted multiline or malformed metadata record")
    save()
    return out

def parse_day(text):
    text=text.strip()
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError:
        pass
    for fmt in ("%d/%m/%Y","%d/%m/%y"):
        try:return datetime.strptime(text,fmt).date().isoformat()
        except ValueError:pass
    raise MetadataError("Date not recognised; no timestamp/season invented")

def season_label(path,source,league):
    name=Path(path).stem
    if source=="canonical":
        return name[len(league)+1:] if name.startswith(league+"_") else name
    found=re.search(r"Season_(\d{4}_\d{4})",name)
    return found.group(1) if found else name

def load_metadata(path,source,league):
    path=Path(path);rows=[];errors=[]
    before=held_pin(path)
    try:
        with path.open("rb") as stream:
            header=next(csv.reader([stream.readline(65536).decode("utf-8-sig")]))
            required={"date":"day","home_team":"home","away_team":"away"} if source=="canonical" else {"Date":"day","HomeTeam":"home","AwayTeam":"away"}
            if len(header)!=len(set(header)) or not set(required).issubset(header):
                raise MetadataError("Required name/date headers missing or duplicated")
            indices={i:required[n] for i,n in enumerate(header) if n in required}
            for line,raw in enumerate(stream,2):
                try:
                    projected=project_metadata(raw,indices)
                    if set(projected)!={"day","home","away"}:raise MetadataError("Missing metadata cells")
                    for value in [projected["home"],projected["away"]]:
                        if re.search(r"(?:token|secret|password|api[_ -]?key)\s*[:=]|[a-fA-F0-9]{32,}",value):
                            raise SensitiveMetadata()
                    if not projected["home"] or not projected["away"]:raise MetadataError("Empty club name")
                    rows.append(MetaFixture(league,source,season_label(path,source,league),str(path.resolve()),line,
                                parse_day(projected["day"]),projected["home"],projected["away"]))
                except SensitiveMetadata:
                    raise
                except (MetadataError,UnicodeError,ValueError):
                    errors.append({"path":str(path.resolve()),"line":line,"type":"INVALID_METADATA_ROW","raw_values":"NOT_LOGGED"})
    except SensitiveMetadata:
        raise
    except (OSError,MetadataError,UnicodeError,StopIteration,csv.Error) as exc:
        errors.append({"path":str(path.resolve()),"type":"UNREADABLE_OR_INVALID_METADATA_FILE","error_class":type(exc).__name__})
    after=held_pin(path)
    if any(before[k]!=after[k] for k in ("size_bytes","sha256")):
        raise MetadataError("Input changed during audit")
    return rows,errors,before

def inventory(rows):
    raw=Counter();unique=defaultdict(set);where=defaultdict(set)
    for f in rows:
        for name in set((f.home,f.away)):
            key=(f.league,f.source,f.season_label,name)
            raw[key]+=1;unique[key].add(f.key());where[(f.league,name)].add(f.source)
    return [{"league":l,"source":s,"season_label":season,"source_name":name,"raw_match_rows":n,
             "metadata_unique_fixture_count":len(unique[(l,s,season,name)]),"name_appears_in_sources":sorted(where[(l,name)])}
            for (l,s,season,name),n in sorted(raw.items())]

def audit_group(league,season,rows):
    teams=sorted({x for f in rows for x in (f.home,f.away)})
    expected=2*(len(teams)-1)
    raw_count=Counter();unique_count=Counter();duplicates=defaultdict(list);days=defaultdict(list);pairs=defaultdict(list)
    for f in rows:
        for team in set((f.home,f.away)):
            raw_count[team]+=1;days[(f.day,team)].append(f)
        duplicates[f.key()].append(f);pairs[(f.home,f.away)].append(f)
    for day,home,away in duplicates:
        for team in set((home,away)):unique_count[team]+=1
    self_rows=[{"date":f.day,"team":f.home,**f.location()} for f in rows if f.home==f.away]
    dup=[{"date":k[0],"home":k[1],"away":k[2],"count":len(v),"locations":[f.location() for f in v]}
         for k,v in sorted(duplicates.items()) if len(v)>1]
    collisions=[{"date":day,"team":team,"raw_rows":len(v),"distinct_fixtures":len({f.key() for f in v}),
                 "fixtures":[{"home":h,"away":a} for d,h,a in sorted({f.key() for f in v})]}
                for (day,team),v in sorted(days.items()) if len({f.key() for f in v})>1]
    repeated=[{"home":h,"away":a,"raw_rows":len(v),"distinct_fixture_dates":len({f.day for f in v}),"dates":sorted({f.day for f in v})}
              for (h,a),v in sorted(pairs.items()) if len(v)>1]
    counts=[{"team":t,"raw_match_rows":raw_count[t],"metadata_unique_fixtures":unique_count[t],
             "double_round_robin_reference":expected,"delta_unique_vs_reference":unique_count[t]-expected} for t in teams]
    return {"league":league,"season_label":season,"season_basis":"FILENAME_LABEL_NOT_VERIFIED_PROVIDER_SEASON",
       "raw_rows":len(rows),"metadata_unique_fixtures":len(duplicates),"distinct_teams":len(teams),
       "reference_matches_per_team":expected,"reference_total_fixtures":len(teams)*(len(teams)-1),
       "reference_warning":"Observed-team-count double round robin only; partial/mixed-season files can differ without an identity defect",
       "team_counts":counts,"self_fixtures":self_rows,"duplicate_metadata_fixtures":dup,
       "multiple_distinct_fixtures_per_team_date":collisions,
       "team_date_groups_with_multiple_raw_rows":sum(len(v)>1 for v in days.values()),
       "repeated_ordered_pairs_raw":repeated,
       "repeated_ordered_pairs_after_metadata_dedupe":[x for x in repeated if x["distinct_fixture_dates"]>1]}

def discover(root):
    root=Path(root);files=[]
    for path in sorted((root/"data/processed/matches").glob("*.csv")):
        league=path.stem.split("_")[0]
        if league in LEAGUES:files.append((path,"canonical",league))
    for league in LEAGUES:
        for path in sorted((root/"data/historical"/league).glob("*.csv")):files.append((path,"historical",league))
    return files

def run_audit(root,output_directory,packet_root):
    """Names/dates only, no matching or validator application to real fixtures."""
    rows=[];errors=[];pins=[]
    for path,source,league in discover(root):
        try:
            meta,bad,identity=load_metadata(path,source,league)
        except (OSError,MetadataError) as exc:
            errors.append({"path":str(path.resolve()),"type":"INPUT_UNREADABLE_OR_CHANGED","error_class":type(exc).__name__})
            continue
        rows+=meta;errors+=bad;pins.append(identity)
    groups=defaultdict(list)
    for f in rows:
        if f.source=="canonical":groups[(f.league,f.season_label)].append(f)
    audits=[audit_group(l,s,rr) for (l,s),rr in sorted(groups.items())]
    names=inventory(rows)
    known={"confirmed_metadata_findings":[],"reference_deviations":[],"errors":errors}
    for group in audits:
        common={"league":group["league"],"season_label":group["season_label"]}
        for kind,key in [("SELF_FIXTURE","self_fixtures"),("DUPLICATE_METADATA_FIXTURE","duplicate_metadata_fixtures"),
                         ("TEAM_MULTIPLE_DISTINCT_FIXTURES_DATE","multiple_distinct_fixtures_per_team_date"),
                         ("REPEATED_ORDERED_PAIR_DISTINCT_DATES","repeated_ordered_pairs_after_metadata_dedupe")]:
            for item in group[key]:known["confirmed_metadata_findings"].append({**common,"type":kind,**item})
        for item in group["team_counts"]:
            if item["delta_unique_vs_reference"]!=0:
                known["reference_deviations"].append({**common,"type":"MATCH_COUNT_REFERENCE_DEVIATION",**item})
    # Consolidated exact names: comparison/table of raw observed variants, no matcher applied.
    consolidated=defaultdict(lambda:defaultdict(list))
    for row in names:consolidated[(row["league"],row["source_name"])][row["source"]].append(row)
    text=io.StringIO(newline="");writer=csv.writer(text)
    writer.writerow(["league","exact_source_name","canonical_seasons_and_raw_counts","historical_seasons_and_raw_counts"])
    for (league,name),sources in sorted(consolidated.items()):
        writer.writerow([league,name,
           "; ".join(f"{x['season_label']}:{x['raw_match_rows']}" for x in sources.get("canonical",[])),
           "; ".join(f"{x['season_label']}:{x['raw_match_rows']}" for x in sources.get("historical",[]))])
    result={"scope":"Metadata only, no real matcher/validator/rebuild applied, no scores decoded",
            "input_files":len(pins),"metadata_rows":len(rows),"name_rows":len(names),"exact_names":len(consolidated),
            "canonical_groups":len(audits),"confirmed_finding_counts":dict(Counter(x["type"] for x in known["confirmed_metadata_findings"])),
            "reference_deviations":len(known["reference_deviations"]),"metadata_errors":len(errors),"sealed_scores_decoded":0}
    outputs={"name-inventory.json":names,"canonical-integrity.json":audits,"known-bad-metadata.json":known,
             "input-identities.json":pins,"audit-summary.json":result}
    encoded={name:(json.dumps(value,ensure_ascii=False,separators=(",",":"))+"\n").encode("utf-8") for name,value in outputs.items()}
    encoded["name-variants.csv"]=text.getvalue().encode("utf-8")
    packet_root=Path(packet_root).resolve();output_directory=Path(output_directory).resolve()
    if not output_directory.is_relative_to(packet_root):
        raise MetadataError("Audit outputs must stay inside packet")
    current=sum(p.stat().st_size for p in packet_root.rglob("*") if p.is_file())
    if current+sum(len(x) for x in encoded.values())>4_000_000:
        raise PacketLimit("Audit would leave insufficient room under 5 MB packet cap")
    output_directory.mkdir(parents=True,exist_ok=False)
    for name,value in encoded.items():
        with (output_directory/name).open("xb") as stream:stream.write(value)
    return result
