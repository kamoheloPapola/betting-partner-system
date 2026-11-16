"""Strict same-league exact matcher and unapplied rebuilt-table validators."""
from collections import defaultdict,Counter
from datetime import date
import csv
import math
import re
import unicodedata

LEAGUES=("PL","BL1","FL1","SA","PD")
ALIAS_COLUMNS=("league","source_name","provider_team_id","note")

def normalize_name(value):
    if not isinstance(value,str):
        raise ValueError("Name must be text")
    text=unicodedata.normalize("NFKD",value.casefold())
    text="".join(c for c in text if not unicodedata.combining(c) and not unicodedata.category(c).startswith("P"))
    return " ".join(text.split())

def provider_id(value):
    if isinstance(value,bool):
        raise ValueError("Boolean is not a provider ID")
    number=int(value)
    if number<=0 or str(number)!=str(value).strip():
        raise ValueError("Provider ID must be a positive integer")
    return number

def read_aliases(path):
    with open(path,encoding="utf-8-sig",newline="") as stream:
        reader=csv.DictReader(stream)
        if tuple(reader.fieldnames or ())!=ALIAS_COLUMNS:
            raise ValueError("Alias CSV columns/order must be league,source_name,provider_team_id,note")
        rows=[]
        for row in reader:
            if row["league"] not in LEAGUES or not row["source_name"].strip() or not row["note"].strip():
                raise ValueError("Alias league/name/note missing or invalid")
            row["provider_team_id"]=provider_id(row["provider_team_id"])
            rows.append(row)
    return rows

def match_name(league,source_name,teams_by_league,aliases=()):
    normalized=normalize_name(source_name)
    result={"league":league,"source_name":source_name,"normalized":normalized,"status":"UNMATCHED","provider_team_ids":[],"evidence":[]}
    if league not in LEAGUES or not normalized:
        return result
    teams=teams_by_league.get(league,())
    ids=set();valid_ids=set()
    for team in teams:
        tid=provider_id(team["id"]);valid_ids.add(tid)
        for field in ("name","shortName","tla"):
            value=team.get(field)
            if value and normalize_name(value)==normalized:
                ids.add(tid);result["evidence"].append({"provider_team_id":tid,"field":field})
    for alias in aliases:
        if alias["league"]!=league or normalize_name(alias["source_name"])!=normalized:
            continue
        tid=provider_id(alias["provider_team_id"])
        if tid not in valid_ids or not str(alias.get("note","")).strip():
            result["status"]="INVALID_ALIAS"
            return result
        ids.add(tid);result["evidence"].append({"provider_team_id":tid,"field":"explicit_alias","note":alias["note"]})
    result["provider_team_ids"]=sorted(ids)
    result["status"]="MATCHED" if len(ids)==1 else "AMBIGUOUS" if ids else "UNMATCHED"
    return result

def validate_mapping(rows):
    names=defaultdict(set);ids=defaultdict(set);normalized=defaultdict(set);findings=[]
    for row in rows:
        league=row["league"];name=row["source_name"];tid=provider_id(row["provider_team_id"])
        names[(league,name)].add(tid);ids[(league,tid)].add(name)
        normalized[(league,normalize_name(name))].add(name)
    for (league,name),values in names.items():
        if len(values)>1:
            findings.append({"type":"NAME_TO_MULTIPLE_IDS","league":league,"source_name":name,"ids":sorted(values)})
    for (league,tid),values in ids.items():
        if len(values)>1:
            findings.append({"type":"ID_TO_MULTIPLE_SOURCE_NAMES","league":league,"provider_team_id":tid,"names":sorted(values)})
    for (league,norm),values in normalized.items():
        if len(values)>1:
            findings.append({"type":"NORMALIZATION_COLLISION_REVIEW","league":league,"normalized":norm,"names":sorted(values)})
    return findings

def sealed_day(day):
    return "2024-07-01"<=day<"2026-05-01"

def final_score(value):
    if isinstance(value,bool):
        raise ValueError("Boolean score")
    number=float(value)
    if not math.isfinite(number) or number<0 or number!=int(number):
        raise ValueError("Invalid final score")
    return int(number)

def metadata(row):
    league=row["league"];season=str(row["season"]);day=row["date"]
    if league not in LEAGUES or date.fromisoformat(day).isoformat()!=day:
        raise ValueError("Invalid league or ISO date")
    return league,season,day,provider_id(row["home_team_id"]),provider_id(row["away_team_id"])

def validate_rebuilt(rows,*,expected_team_counts,provider_rows=None,mapping_rows=None):
    """Synthetic tests only in NEXT-33. Never inspect any sealed score."""
    findings=[];groups=defaultdict(list);team_days=defaultdict(list);pairs=defaultdict(list)
    fixture_keys=defaultdict(list);mapping=[];provider=defaultdict(list);sealed_skips=0;score_checks=0
    for index,row in enumerate(provider_rows or ()):
        try:
            key=metadata(row)
            provider[key].append(row)  # Metadata index only; no score access.
        except (ValueError,KeyError,TypeError):
            findings.append({"type":"INVALID_PROVIDER_METADATA","row":index})
    for index,row in enumerate(rows):
        try:
            league,season,day,home,away=metadata(row)
        except (ValueError,KeyError,TypeError):
            findings.append({"type":"INVALID_METADATA","row":index});continue
        groups[(league,season)].append((index,home,away))
        fixture_keys[(league,season,day,home,away)].append(index)
        pairs[(league,season,home,away)].append(index)
        for tid in set((home,away)):
            team_days[(league,day,tid)].append(index)
        if home==away:
            findings.append({"type":"SELF_FIXTURE","row":index,"league":league,"date":day,"provider_team_id":home})
        for side,tid in [("home",home),("away",away)]:
            name=row.get(side+"_source_name")
            if name:
                mapping.append({"league":league,"source_name":name,"provider_team_id":tid})
        if sealed_day(day):
            sealed_skips+=1
            continue  # Guard MUST precede all canonical AND provider score access.
        candidates=provider.get((league,season,day,home,away),[])
        if provider_rows is None:
            findings.append({"type":"SCORE_COMPARISON_UNVERIFIED","row":index});continue
        if len(candidates)!=1:
            findings.append({"type":"PROVIDER_FIXTURE_MISSING_OR_AMBIGUOUS","row":index,"count":len(candidates)});continue
        try:
            target=candidates[0]
            scores=(final_score(row["home_score"]),final_score(row["away_score"]))
            expected=(final_score(target["home_score"]),final_score(target["away_score"]))
            if min(scores+expected)<0:
                raise ValueError()
            score_checks+=1
            if scores!=expected:
                findings.append({"type":"SCORE_MISMATCH_UNSEALED","row":index,"league":league,"date":day})
        except (ValueError,KeyError,TypeError):
            findings.append({"type":"INVALID_OR_MISSING_UNSEALED_SCORE","row":index})
    for (league,season),matches in groups.items():
        teams={t for _,h,a in matches for t in (h,a)}
        expected=expected_team_counts.get((league,season))
        if expected is None:
            findings.append({"type":"TEAM_COUNT_UNVERIFIED","league":league,"season":season,"observed":len(teams)})
            continue
        if len(teams)!=expected:
            findings.append({"type":"TEAM_COUNT_MISMATCH","league":league,"season":season,"observed":len(teams),"expected":expected})
        counts=Counter(t for _,h,a in matches for t in set((h,a)))
        for tid,count in counts.items():
            if count!=2*(expected-1):
                findings.append({"type":"MATCH_COUNT_MISMATCH","league":league,"season":season,"provider_team_id":tid,"observed":count,"expected":2*(expected-1)})
    for (league,day,tid),indices in team_days.items():
        if len(indices)>1:
            findings.append({"type":"TEAM_MULTIPLE_MATCHES_DATE","league":league,"date":day,"provider_team_id":tid,"rows":indices})
    for (league,season,home,away),indices in pairs.items():
        if len(indices)>1:
            findings.append({"type":"REPEATED_ORDERED_PAIR","league":league,"season":season,"home_team_id":home,"away_team_id":away,"rows":indices})
    for key,indices in fixture_keys.items():
        if len(indices)>1:
            findings.append({"type":"DUPLICATE_FIXTURE","rows":indices})
    mappings=mapping if mapping_rows is None else list(mapping_rows)
    findings+=validate_mapping(mappings)
    covered={(r["league"],provider_id(r["provider_team_id"])) for r in mappings}
    for (league,season),matches in groups.items():
        for tid in {t for _,h,a in matches for t in (h,a)}:
            if (league,tid) not in covered:
                findings.append({"type":"NAME_ID_MAPPING_UNVERIFIED","league":league,"provider_team_id":tid})
    return {"validation_passed":not findings,"findings":findings,"unsealed_score_comparisons":score_checks,
            "sealed_score_comparisons_skipped":sealed_skips,"sealed_scores_accessed":False,
            "scope":"One-to-one mapping enforced literally; aliases do not silently waive many-to-one or normalization-collision findings"}
