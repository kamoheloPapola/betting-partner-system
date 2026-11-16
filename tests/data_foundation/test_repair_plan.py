"""Synthetic-only NEXT-35 candidate simulation and hard-boundary checks."""
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import subprocess
import pytest
from src.data_foundation import repair_plan as rp
from src.data_foundation.audit import MetaFixture
from src.data_foundation.snapshot_metadata import project_snapshot,get_meta,ScoreAccessError

FP={"sha256":"a"*64}

def row(day,home,away,line=2):
    return MetaFixture("BL1","canonical","2025","SYNTHETIC",line,day,home,away)

def fixture(mid,day,h,a,names=None):
    names=names or {h:str(h),a:str(a)}
    return {"id":mid,"date":day,"home_id":h,"away_id":a,
        "home":names[h],"away":names[a],"home_meta":{"id":h,"name":names[h],"shortName":names[h],"tla":names[h]},
        "away_meta":{"id":a,"name":names[a],"shortName":names[a],"tla":names[a]},
        "status":"FINISHED","near_midnight":False}

@pytest.mark.parametrize("extra",[
    ("PL","UNLISTED",99),("BL1","BAYERN MUNICH",99),("PD","BARCELONA",80)])
def test_alias_pairs_refuse_any_extra(extra):
    with pytest.raises(ValueError):
        rp.validate_pairs(rp.CANDIDATES+(extra,))

def test_alias_duplicate_pair_refused():
    with pytest.raises(ValueError):
        rp.validate_pairs(rp.CANDIDATES+(rp.CANDIDATES[0],))

def test_fixed_alias_sheet_unreviewed_with_blank_decisions():
    provider=[fixture(1,"2025-08-01",67,65,{67:"Newcastle United FC",65:"Manchester City FC"})]
    aliases,sheet,issues,span=rp.prepare_aliases("PL",[row("2025-08-01","NEWCASTLE UNITED","MANCHESTER CITY")],provider,
        pairs=(("PL","NEWCASTLE UNITED",67),("PL","MANCHESTER CITY",65)))
    assert len(sheet)==2 and not issues
    assert all(x["status"]=="UNREVIEWED" and x["reviewer"]==x["decision"]=="" for x in sheet)
    assert aliases[rp.normalize_name("NEWCASTLE UNITED")]=={67}

def test_candidate_provider_name_collision_is_reported():
    provider=[fixture(1,"2025-08-01",67,65,{67:"Elsewhere",65:"NEWCASTLE UNITED"})]
    _,_,issues,_=rp.prepare_aliases("PL",[],provider,pairs=(("PL","NEWCASTLE UNITED",67),))
    assert {r["type"] for r in issues}>={"CANDIDATE_MATCHES_OTHER_ID","NAME_MULTIPLE_IDS"}

def test_three_absorbed_plus_label_closes_synthetic_only(monkeypatch):
    ids=[16,28,44,99];provider=[];canonical=[]
    for index,(home,away) in enumerate((h,a) for h in ids for a in ids if h!=a):
        day=f"2025-08-{index+1:02d}"
        provider.append(fixture(index,day,home,away))
        canonical.append(row(day,"X","X",index+2))
    monkeypatch.setitem(rp.GROUPS,"BL1",{16:set(ids)})
    sim=rp.simulation("BL1",canonical,provider,{rp.normalize_name("X"):{16}},FP)
    assert sim["H-E"]=="PASS"
    assert sim["affected_counts"]=={"UNIQUE":12}
    assert sim["zero_self_global"]==sim["per_club_global"]==sim["provider_membership_resolved"]=="PASS"
    assert all(r["delta"]==0 for r in sim["per_club"])

def test_group_restricted_same_date_ambiguity_not_broken():
    provider=[fixture(1,"2025-08-01",16,28),fixture(2,"2025-08-01",44,16)]
    plan=rp.row_plan(row("2025-08-01","X","X"),provider,{"x":{16}},{16:{16,28,44}},FP)
    assert plan["status"]=="AMBIGUOUS" and plan["candidate_fixture_ids"]==[1,2]
    assert plan["proposed_home_id"] is None and plan["method"]=="GROUP_RESTRICTED"

def test_collision_is_two_candidate_fixtures_not_silent_duplicate():
    provider=[fixture(1,"2025-08-01",16,28),fixture(2,"2025-08-01",44,16)]
    plans=[rp.row_plan(row("2025-08-01","X","X",line),provider,{"x":{16}},{16:{16,28,44}},FP) for line in [2,3]]
    assert plans[0]["source_sha256"]==plans[1]["source_sha256"]
    assert plans[0]["line"]!=plans[1]["line"]
    assert all(p["status"]=="AMBIGUOUS" for p in plans)
    assert {i for p in plans for i in p["candidate_fixture_ids"]}=={1,2}

@pytest.mark.parametrize("delta,status",[(0,"UNIQUE"),(1,"UNMATCHED")])
def test_exact_anchor_adjacent_never_applied(delta,status):
    provider=[fixture(5,f"2025-08-{1+delta:02d}",28,7)]
    p=rp.row_plan(row("2025-08-01","X","Y"),provider,{"x":{16},"y":{7}},{16:{16,28}},FP)
    assert p["status"]==status
    assert p["adjacent_fixture_ids"]==([5] if delta else [])
    assert p["proposed_home_id"]==(28 if not delta else None)

def test_unmatched_other_side_remains_unmatched():
    p=rp.row_plan(row("2025-08-01","X","UNKNOWN"),[fixture(1,"2025-08-01",28,7)],{"x":{16}},{16:{16,28}},FP)
    assert p["status"]=="UNMATCHED"

def test_ambiguous_name_is_not_adopted_from_single_fixture():
    p=rp.row_plan(row("2025-08-01","X","Y"),[fixture(1,"2025-08-01",28,7)],
        {"x":{16},"y":{7,8}},{16:{16,28}},FP)
    assert p["name_mapping_ambiguous"] and p["status"]=="AMBIGUOUS"
    assert p["proposed_home_id"] is None and p["proposed_away_id"] is None

def test_pair_invariant_COVID_style_spill_closes():
    full=[row("2020-06-01","A","B"),row("2020-07-01","B","A",3)]
    assert rp.pair_invariant(full[:1])["status"]=="FAIL"
    assert rp.pair_invariant(full)["status"]=="PASS"
    assert rp.pair_invariant(full+full)["status"]=="FAIL"

@pytest.mark.parametrize("day,expected",[
    ("2020-05-31",2019),("2020-06-01",2020),("2020-07-01",2020),("2024-06-02",2024)])
def test_calendar_cutoff_checker(day,expected):
    assert rp.cutoff_label(day,6)==expected

def test_git_collector_injected_runner_and_timing():
    calls=[]
    def fake(argv,**kwargs):
        calls.append((argv,kwargs))
        return SimpleNamespace(returncode=0,stdout=b"COMMIT\tabc\t2025-01-01T00:00:00Z\tcreated\n",stderr=b"")
    r=rp.git_collect(Path("."),["git","log","--follow","--format=%H%x09%cI%x09%s","--numstat","--","data/example.csv"],fake)
    assert r["exit_code"]==0 and r["utc_start"]<=r["utc_completion"]
    assert calls[0][1]["timeout"]==25

def test_git_timeout_retained():
    def fake(*args,**kwargs):
        raise subprocess.TimeoutExpired(args[0],25,output=b"partial",stderr=b"timeout")
    r=rp.git_collect(Path("."),["git","ls-files","--","data/a.csv"],fake)
    assert r["exit_code"]==124 and r["stdout"]=="partial"

@pytest.mark.parametrize("argv",[
    ["git","show","HEAD:data/a.csv"],["git","diff","--","data/a.csv"],
    ["git","blame","data/a.csv"],["git","log","--patch","--format=%H%cI%s"],
    ["git","log","--output=data/a.csv","--format=%H%cI%s"]])
def test_git_allowlist_refuses(argv):
    with pytest.raises(ValueError):rp.checked_git(argv)

def test_timestamp_collector_injected():
    r=rp.timestamp_collect("fake",lambda _:SimpleNamespace(st_ctime=0,st_mtime=60))
    assert r["created_UTC_windows"].startswith("1970-01-01T00:00:00")
    assert r["modified_UTC"].startswith("1970-01-01T00:01:00")

@pytest.mark.parametrize("directory",["data","src","scripts","tests"])
def test_writer_refuses_production_roots(tmp_path,directory):
    with pytest.raises(ValueError):
        rp.write_new(tmp_path,tmp_path/directory/rp.PACKET_NAME,"output.csv","no")

def test_writer_exclusive_path_traversal_and_comma_encoding(tmp_path):
    packet=tmp_path/"tmp"/rp.PACKET_NAME
    text=rp.csv_text([{"name":"Café, Club"}],["name"])
    pin=rp.write_new(tmp_path,packet,"example.csv",text)
    assert pin["size_bytes"] and '"Café, Club"' in (packet/"example.csv").read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):rp.write_new(tmp_path,packet,"example.csv","different")
    with pytest.raises(ValueError):rp.write_new(tmp_path,packet,"../data/a.csv","no")
    with pytest.raises(ValueError):rp.write_new(tmp_path,packet,tmp_path/"outside.csv","no")

def test_score_guard_and_projection():
    class Guard(dict):
        def get(self,key,*args):
            if "score" in key.lower():raise AssertionError("SCORE_TOUCHED")
            return super().get(key,*args)
    payload=Guard(matches=[Guard(id=1,utcDate="2025-08-01T12:00:00Z",score=object(),homeTeam={},awayTeam={})])
    projected=project_snapshot(payload)
    assert "score" not in projected["matches"][0]
    with pytest.raises(ScoreAccessError):get_meta({},"score")

def test_partial_closure_remains_unknown(monkeypatch):
    provider=[fixture(1,"2025-08-01",16,28),fixture(2,"2025-08-01",28,16)]
    monkeypatch.setitem(rp.GROUPS,"BL1",{16:{16,28}})
    s=rp.simulation("BL1",[row("2025-08-01","X","X")],provider,{"x":{16}},FP)
    assert s["zero_self_global"]==s["per_club_global"]=="UNKNOWN"

@pytest.mark.parametrize("leaf",["../data","/absolute"])
def test_checkpoint_leaf_refuses_escape_before_read(tmp_path,leaf):
    with pytest.raises(ValueError,match="Unsafe output leaf"):
        rp.run(tmp_path,tmp_path/"tmp"/rp.PACKET_NAME,leaf)

