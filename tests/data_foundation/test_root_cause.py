"""NEXT-36 synthetic fixtures only; existing autouse fixture denies sockets."""
from datetime import datetime,timedelta,timezone
from pathlib import Path
import json
import pytest
from src.data_foundation import root_cause as rc

def packet(repo):
    return repo/"tmp"/rc.PACKET_NAME

def make_controls(before=32,after=32,offset_before=0,offset_after=None):
    if offset_after is None:offset_after=offset_before
    rows=[]
    for side,n,day,offset in [("before",before,"2026-01-04",offset_before),("after",after,"2026-04-04",offset_after)]:
        for i in range(n):
            provider=datetime.fromisoformat(day+"T12:00:00+00:00")+timedelta(minutes=i)
            local=provider+timedelta(minutes=offset)
            rows.append({"line":i+2 if side=="before" else i+60,"date":day,
                "csv_time":local.strftime("%H:%M"),"provider_utc":provider.isoformat()})
    return rows

def four_rows():
    return [{"line":179,"date":"2026-01-04","csv_time":"15:00","matched_fixture_ids":[544385]},
        {"line":182,"date":"2026-01-04","csv_time":"17:00","matched_fixture_ids":[544387]},
        {"line":295,"date":"2026-04-04","csv_time":"15:00","matched_fixture_ids":[544502]},
        {"line":296,"date":"2026-04-04","csv_time":"17:00","matched_fixture_ids":[544503]}]

def test_whole_hash_groups_identical_and_separates_CRLF_BOM(tmp_path):
    a=tmp_path/"a.csv";b=tmp_path/"b.csv";c=tmp_path/"c.csv";d=tmp_path/"d.csv"
    a.write_bytes(b"Date\n01/06/2024\n");b.write_bytes(a.read_bytes())
    c.write_bytes(b"Date\r\n01/06/2024\r\n");d.write_bytes(b"\xef\xbb\xbf"+a.read_bytes())
    groups=rc.group_hashes([a,b,c,d])
    assert sorted(len(g["files"]) for g in groups)==[1,1,2]

@pytest.mark.parametrize("copies",[1,2])
def test_date_prediction_counts_every_identical_file(tmp_path,copies):
    files=[]
    for n in range(copies):
        q=tmp_path/f"source{n}.csv";q.write_text("Date\n01/08/2024\n02/08/2024\n",encoding="utf-8")
        files.append(rc.count_dates(tmp_path,q,"Date"))
    assert rc.predict_counts(files,2024)==copies*2

def test_date_only_month_boundary_invalid_and_poison_cells(tmp_path):
    q=tmp_path/"x.csv";q.write_bytes(b"Date,home_score,odds\n31/05/2024,\xff,\xfe\n01/06/2024,\xff,\xfe\nbad,\xff,\xfe\n,\xff,\xfe\n")
    result=rc.count_dates(tmp_path,q,"Date")
    assert result["counts"]=={2023:1,2024:1}
    assert result["raw_rows"]==4 and len(result["errors"])==2

@pytest.mark.parametrize("body,reason",[(b"","EMPTY_FILE"),(b"Date\n",None)])
def test_empty_header_only_are_explicit(tmp_path,body,reason):
    q=tmp_path/"x.csv";q.write_bytes(body)
    result=rc.count_dates(tmp_path,q,"Date")
    assert result["raw_rows"]==0 and result["counts"]=={}
    assert result["errors"]==([reason] if reason else [])

def test_token_overlap_diagnostic_and_disjoint():
    assert rc.shared_tokens("AC Milan","AC Pisa")==["AC"]
    assert rc.shared_tokens("Bayern München","MUNchen")==["MUNCHEN"]
    assert rc.shared_tokens("Ajax","Liverpool")==[]

def test_import_time_file_loader_gate_prevents_behavior():
    source="def _load_season_maps():\n    return open('data.json').read()\nLEAGUE_TEAMS = _load_season_maps()\n"
    result=rc.naming_import_gate(source)
    assert result["status"]=="NOT_RUN" and result["blocking_calls"][0]["line"]==3

@pytest.mark.parametrize("root",["data","src","scripts","tests"])
def test_writer_production_root_refused(tmp_path,root):
    with pytest.raises(ValueError):rc.write_new(tmp_path,tmp_path/root/rc.PACKET_NAME,"x","forbidden")

def test_writer_exclusive_traversal(tmp_path):
    rc.write_new(tmp_path,packet(tmp_path),"x","first")
    with pytest.raises(FileExistsError):rc.write_new(tmp_path,packet(tmp_path),"x","second")
    with pytest.raises(ValueError):rc.write_new(tmp_path,packet(tmp_path),"../x","escape")
    with pytest.raises(ValueError):rc.write_new(tmp_path,packet(tmp_path),tmp_path/"outside","escape")

@pytest.mark.parametrize("field",["home_score","odds","result","corners"])
def test_cell_guard_refuses_nonwhitelist(tmp_path,field):
    with pytest.raises(ValueError):rc.selected_cells(b"anything",{0:field},repo=tmp_path,path=tmp_path/"x.csv")

def test_only_extra_kickoff_cell_is_PD2025(tmp_path):
    approved=tmp_path/"data/processed/matches/PD_2025.csv"
    assert rc.selected_cells(b"15:00,\xff",{0:"kickoff"},repo=tmp_path,path=approved)=={"kickoff":"15:00"}
    for wrong in ["PL_2025.csv","PD_2024.csv"]:
        with pytest.raises(ValueError):
            rc.selected_cells(b"15:00",{0:"kickoff"},repo=tmp_path,path=approved.with_name(wrong))

def test_HM_missing_schema_no_other_column_search(tmp_path):
    q=tmp_path/"data/processed/matches/PD_2025.csv";q.parent.mkdir(parents=True)
    q.write_text("date,home_team,away_team\n2026-01-04,A,B\n",encoding="utf-8")
    assert rc.kickoff_schema(tmp_path,q)["status"]=="NOT_RUN"

@pytest.mark.parametrize("offset",[0,60])
def test_calibration_constant_offset(offset):
    rule=rc.calibrate_times(make_controls(offset_before=offset))
    assert rule["status"]=="PASS" and rule["offset_minutes"]=={"before":offset,"after":offset}

def test_calibration_DST_sides_and_minimum():
    rule=rc.calibrate_times(make_controls(offset_before=60,offset_after=120))
    assert rule["status"]=="PASS" and rule["mode"]=="DST_OFFSET"
    assert rc.calibrate_times(make_controls(before=29,offset_before=60,offset_after=120))["status"]=="UNKNOWN"

@pytest.mark.parametrize("value",["00:00","",None])
def test_control_placeholder_refused(value):
    controls=make_controls();controls[0]["csv_time"]=value
    result=rc.calibrate_times(controls)
    assert result["status"]=="UNKNOWN" and result["placeholders_or_exceptions"]==1

def test_mixed_offset_unknown_not_tuned():
    controls=make_controls()
    controls[0]["csv_time"]="13:00"
    assert rc.calibrate_times(controls)["status"]=="UNKNOWN"

def test_repeated_default_is_not_assumed_valid():
    controls=make_controls()
    for r in controls:r["csv_time"]="12:00";r["provider_utc"]=r["date"]+"T12:00:00Z"
    assert rc.calibrate_times(controls)["status"]=="UNKNOWN"

def synthetic_time_file(repo):
    q=repo/"data/processed/matches/PD_2025.csv";q.parent.mkdir(parents=True)
    q.write_text("date,Time,home_score,odds\n"+"".join(f"2026-01-04,15:{i%60:02d},SEALED,SEALED\n" for i in range(2,297)),encoding="utf-8")
    return q

def test_freeze_written_before_targets_and_changed_rule_refused(tmp_path):
    q=synthetic_time_file(tmp_path)
    with pytest.raises(ValueError):rc.read_kickoff_cells(tmp_path,q,"Time",rc.TARGET_LINES,"targets")
    with pytest.raises(ValueError):rc.read_kickoff_cells(tmp_path,q,"Time",{179},"controls")
    receipt=rc.freeze_rule(tmp_path,packet(tmp_path),rc.calibrate_times(make_controls()))
    assert Path(receipt["identity"]["path"]).exists() and not receipt["target_read_started"]
    result=rc.read_kickoff_cells(tmp_path,q,"Time",rc.TARGET_LINES,"targets",receipt)
    assert set(result)==rc.TARGET_LINES and receipt["target_read_started"]
    with pytest.raises(ValueError):rc.read_kickoff_cells(tmp_path,q,"Time",rc.TARGET_LINES,"targets",receipt)
    Path(receipt["identity"]["path"]).write_text("{}",encoding="utf-8")
    with pytest.raises(ValueError,match="changed"):rc.verify_frozen(receipt)

@pytest.mark.parametrize("kind,expected",[
    ("distinct","PASS"),("equal_clock","UNKNOWN"),("same_fixture","FAIL"),
    ("neither_candidate","UNKNOWN"),("outside_candidate","FAIL"),("missing","UNKNOWN")])
def test_HM_verdicts_no_row_order_tie_break(kind,expected):
    rows=four_rows()
    if kind=="equal_clock":rows[1]["csv_time"]=rows[0]["csv_time"]
    if kind=="same_fixture":rows[1]["matched_fixture_ids"]=rows[0]["matched_fixture_ids"]
    if kind=="neither_candidate":rows[0]["matched_fixture_ids"]=[]
    if kind=="outside_candidate":rows[0]["outside_candidate"]=True
    if kind=="missing":rows[0]["csv_time"]=""
    assert rc.hm_verdict(rows)==expected
    assert rc.hm_verdict(list(reversed(rows)))==expected

def test_apply_frozen_rule_distinct_exact_times(tmp_path):
    receipt=rc.freeze_rule(tmp_path,packet(tmp_path),rc.calibrate_times(make_controls()))
    receipt["target_read_started"]=True
    rows=four_rows()
    candidates={}
    for r in rows:
        candidates[r["line"]]=[{"id":i,"utcDate":r["date"]+"T"+clock+":00Z","status":"FINISHED"}
            for i,clock in ((544385,"15:00"),(544387,"17:00"))] if r["line"]<200 else [
            {"id":i,"utcDate":r["date"]+"T"+clock+":00Z","status":"FINISHED"}
            for i,clock in ((544502,"15:00"),(544503,"17:00"))]
    assert rc.apply_targets(rows,candidates,receipt)["status"]=="PASS"

