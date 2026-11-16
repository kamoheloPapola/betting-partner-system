"""Synthetic only; existing autouse fixture blocks every socket connection."""
import json
from pathlib import Path
import pytest
from src.data_foundation.audit import MetaFixture
from src.data_foundation.identity import validate_rebuilt
from src.data_foundation.snapshot_metadata import (
    identity,verify_snapshots,IntegrityError,ScoreAccessError,get_meta,project_snapshot,
    flatten_snapshot,snapshot_check,double_check,label_check,merge_diagnostic,resolve_row,utc_details)

def fixture(day,home="A",away="B",line=2):
    return MetaFixture("PL","canonical","2025","synthetic",line,day,home,away)

def snap(mid,day,home_id,away_id,names=None):
    names=names or {1:"A",2:"B",3:"C",4:"D"}
    h={"id":home_id,"name":names.get(home_id),"shortName":names.get(home_id),"tla":names.get(home_id)}
    a={"id":away_id,"name":names.get(away_id),"shortName":names.get(away_id),"tla":names.get(away_id)}
    return {"id":mid,"league":"PL","date":day,"utcDate":day+"T15:00:00Z","status":"FINISHED","matchday":1,"season":{},
            "home_id":home_id,"away_id":away_id,"home":h["name"],"away":a["name"],"home_meta":h,"away_meta":a,"near_midnight":False}

@pytest.mark.parametrize("field",["size_bytes","sha256"])
def test_hash_size_mismatch_aborts_before_parse(tmp_path,field):
    body=tmp_path/"PL_2024.json";body.write_bytes(b'{"matches":[]}')
    pin=identity(body)
    entry={"league":"PL","season":2024,"http_status":200,**pin}
    entry[field]=999 if field=="size_bytes" else "0"*64
    (tmp_path/"manifest.json").write_text(json.dumps({"entries":[entry]}))
    with pytest.raises(IntegrityError):verify_snapshots(tmp_path,{("PL",2024)})

def test_missing_body_aborts(tmp_path):
    (tmp_path/"manifest.json").write_text(json.dumps({"entries":[{"league":"PL","season":2024,"http_status":200,"path":str(tmp_path/"PL_2024.json"),"size_bytes":1,"sha256":"0"*64}]}))
    with pytest.raises(IntegrityError):verify_snapshots(tmp_path,{("PL",2024)})

def test_projection_never_accesses_score_keys():
    class Poison(dict):
        def get(self,key,*args):
            if "score" in key.lower():raise AssertionError("SCORE_READ")
            return super().get(key,*args)
        def __getitem__(self,key):
            if "score" in key.lower():raise AssertionError("SCORE_READ")
            return super().__getitem__(key)
        def items(self):raise AssertionError("BULK_SOURCE_COPY_FORBIDDEN")
    team={"id":1,"name":"A","shortName":"A","tla":"A"}
    source=Poison(matches=[Poison(id=1,utcDate="2024-07-01T15:00:00Z",status="FINISHED",matchday=1,season=Poison(id=1,startDate="2024-07-01",endDate="2025-06-30",score=Poison()),
                                 homeTeam=team,awayTeam={**team,"id":2,"name":"B"},score=Poison(fullTime={"home":99,"away":88}))],
                  resultSet={"count":1,"first":"2024-07-01","last":"2024-07-01"},score=Poison())
    projected=project_snapshot(source)
    assert len(projected["matches"])==1
    assert "score" not in json.dumps(projected).lower()
    with pytest.raises(ScoreAccessError):get_meta(source,"score")

def test_existing_sealed_score_guard_still_used_on_synthetics():
    class Poison(dict):
        def __getitem__(self,key):
            if "score" in key:raise AssertionError("SEALED_SCORE_READ")
            return super().__getitem__(key)
    a={"league":"PL","season":"2024","date":"2024-07-01","home_team_id":1,"away_team_id":2,"home_source_name":"A","away_source_name":"B"}
    b={**a,"date":"2024-07-02","home_team_id":2,"away_team_id":1,"home_source_name":"B","away_source_name":"A"}
    x=validate_rebuilt([Poison(a),Poison(b)],expected_team_counts={("PL","2024"):2},provider_rows=[Poison(a),Poison(b)])
    assert x["sealed_score_comparisons_skipped"]==2 and x["unsealed_score_comparisons"]==0

@pytest.mark.parametrize("payload,error",[({}, "MATCHES_ABSENT"),({"matches":[]},"MATCHES_EMPTY"),({"matches":{}},"MATCHES_NOT_LIST")])
def test_empty_absent_matches_explicit(payload,error):
    assert error in project_snapshot(payload)["metadata_errors"]

def test_ID_name_uniqueness_null_ID_and_duplicate_match_ID():
    records=[snap(1,"2025-01-01",1,2),snap(1,"2025-01-02",1,3),snap(3,"2025-01-03",None,2)]
    records[1]["home"]="Renamed A";records[1]["away"]="B"
    projected={"resultSet":{"count":3},"matches":[],"metadata_errors":[]}
    check=snapshot_check(projected,records,[])
    assert check["duplicate_match_ids"] and check["null_or_invalid_team_ids"]
    assert check["one_ID_multiple_names"] and check["one_name_multiple_IDs"]
    assert check["schedule_completeness"]=="INCOMPLETE_OR_INVALID_METADATA"

def test_merge_two_clubs_into_third_signature():
    snapshots=[];canonical=[];mid=0
    for h,a in [(1,2),(2,1),(1,3),(3,1),(2,3),(3,2)]:
        mid+=1;day=f"2025-08-{mid:02d}"
        snapshots.append(snap(mid,day,h,a))
        canonical.append(fixture(day,"A","A",mid+1))
    result=merge_diagnostic(canonical,snapshots)
    assert {x["id"] for x in result["clubs_without_exact_canonical_evidence"]}=={2,3}
    assert [x["id"] for x in result["inflated_clubs"]]==[1]
    assert result["surplus_fixture_sides"]==8==result["absent_club_snapshot_span_sides"]
    assert result["self_fixture_unique_count"]==6 and result["hypothesis_status"]=="PASS"

def test_unresolved_names_make_merge_hypothesis_unknown():
    result=merge_diagnostic([fixture("2025-08-01","Unreviewed Alias","A")],[snap(1,"2025-08-01",1,2)])
    assert result["hypothesis_status"]=="UNKNOWN"

@pytest.mark.parametrize("kind,expected,adjacent",[("unique","UNIQUE",0),("ambiguous","AMBIGUOUS",0),("adjacent","UNMATCHED",1)])
def test_exact_resolver_separate_adjacent_bucket(kind,expected,adjacent):
    row=fixture("2025-08-01","A","D")
    records=[snap(1,"2025-08-02" if kind=="adjacent" else "2025-08-01",2,4)]
    if kind=="ambiguous":records.append(snap(2,"2025-08-01",3,4))
    result=resolve_row(row,records,{"a":{1},"d":{4}},{1})
    assert result["exact_status"]==expected
    assert len(result["plus_minus_one_day_candidates"])==adjacent

def test_self_fixture_not_guessed_from_unreliable_anchor():
    result=resolve_row(fixture("2025-08-01","A","A"),[snap(1,"2025-08-01",2,3)],{"a":{1}},{1})
    assert result["exact_status"]=="UNMATCHED" and not result["exact_candidates"]

def test_duplicate_halves_full_and_partial():
    a=fixture("2024-08-01");b=fixture("2024-08-02","B","A")
    assert double_check([a,b,a,b])["full_double"]
    partial=double_check([a,b,a,b,fixture("2024-08-03","C","D")])
    assert not partial["full_double"] and len(partial["residual_non_doubled"])==1

def test_nominal_label_COVID_overrun_is_reported_not_reassigned():
    result=label_check("PL_2019.csv",[fixture("2019-08-01"),fixture("2020-08-01")])
    assert result["inside"]==1 and result["outside"]==1
    assert result["outside_nominal_windows"]=={"2020/21":1}

def test_blank_date_and_midnight_are_explicit():
    assert utc_details("")[1]=="MISSING_UTC_DATE"
    details,error=utc_details("2025-08-01T00:15:00Z")
    assert error is None and details["near_UTC_midnight_1h"]

@pytest.mark.parametrize("filename,year",[("PL_2021.csv",2021),("F1_2021.csv",2020),("E0_2425_latest.csv",2024)])
def test_canonical_year_is_not_a_compact_season_code(filename,year):
    from src.data_foundation.snapshot_metadata import label_year
    assert label_year(filename)==year

@pytest.mark.parametrize("filename,league",[("BACKFILL_BL1_1213.csv","BL1"),("unassigned.csv",None)])
def test_explicit_backfill_league_tag_or_unknown(filename,league):
    from src.data_foundation.snapshot_metadata import csv_league
    assert csv_league(Path(filename),"processed")==league
