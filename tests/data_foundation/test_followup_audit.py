"""Synthetic-only NEXT-37 contracts; no production normalizer or ML imports."""
from pathlib import Path
from types import SimpleNamespace
import json
import os
import subprocess
import sys
import pytest
from src.data_foundation import followup_audit as a

_RECEIPT_DIR = None

@pytest.fixture(autouse=True)
def receipt_scope(tmp_path_factory, monkeypatch):
    base = tmp_path_factory.getbasetemp().resolve()
    destination = Path(os.environ.get("BPS_TEST_EVIDENCE_DIR", str(base.parent/(base.name+"-receipts")))).resolve()
    if destination.is_relative_to(base) or "NEXT-37b-normaliser-probe-02" in destination.parts:
        raise RuntimeError("Test receipts must not use fixture scratch or an accepted packet")
    destination.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(sys.modules[__name__], "_RECEIPT_DIR", destination)

def retain_receipt(label,value):
    import uuid
    if _RECEIPT_DIR is None:
        raise RuntimeError("Receipt scope is not initialized")
    destination = _RECEIPT_DIR/(label+"-"+uuid.uuid4().hex+".json")
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(value, stream)
    return destination

def plan(line,ids,home=None,away=None):
    return {"source_sha256":"a"*64,"line":line,"date":"2025-08-01","original_home":"X","original_away":"Y",
        "candidate_fixture_ids":ids,"proposed_home_id":home,"proposed_away_id":away}

def test_quarantine_two_unique_none_and_row_order():
    rows=[plan(2,[1,2]),plan(3,[3],1,2),plan(4,[])]
    selected=a.quarantine_rows(rows)
    assert [r["line"] for r in selected]==[2]
    assert a.quarantine_rows(list(reversed(rows)))==selected

def test_quarantine_ignores_prohibited_fields():
    class Guard(dict):
        def __getitem__(self,key):
            if key in {"score","odds","time","status","provider_status"}:raise AssertionError("FORBIDDEN")
            return super().__getitem__(key)
    assert len(a.quarantine_rows([Guard(plan(2,[1,2]))]))==1

def test_filename_both_conventions_unrelated_ignored():
    text="a='E0_2425.csv'\nb='PL_season_2024_2025.csv'\nPHASE0_PATH='unrelated'\nx='Q4_report.csv'\n"
    hits=a.filename_hits(text)
    assert len(hits)==2 and hits[0]["short_matches"] and hits[1]["long_matches"]

@pytest.mark.parametrize("source",[
    "import socket\n", "import requests\n", "import os\nx=os.getenv('FAKE')\n",
    "import os\nx=os.environ['FAKE']\n", "open('map.json','w').write('x')\n",
    "SEASON_MAPS_PATH.write_text('x')\n", "SEASON_MAPS_PATH.open('w')\n"])
def test_sandbox_static_refuses_network_env_map_write(source):
    assert a.static_sandbox_gate(source,"same","same")["status"]=="NOT_RUN"

def test_sandbox_hash_and_missing_dependency_refused():
    assert a.static_sandbox_gate("x=1","expected","wrong")["status"]=="NOT_RUN"
    assert a.static_sandbox_gate("x=1","same","same",False)["status"]=="NOT_RUN"

def test_copy_hash_refusal(tmp_path):
    q=tmp_path/"original.py";q.write_text("x=1",encoding="utf-8")
    with pytest.raises(ValueError):
        a.checked_copy(tmp_path,tmp_path/"tmp"/a.PACKET_NAME,q,"sandbox/x.py",{"size_bytes":3,"sha256":"wrong"})

def test_static_string_replace_is_not_filesystem_mutation():
    assert a.static_sandbox_gate("def f(clean):\n    return clean.replace('&','AND')\n","same","same")["status"]=="PASS"

def test_call_recorder_exact_environment_and_output(tmp_path):
    def fake(argv,**kwargs):
        assert kwargs["env"]["PYTHONHASHSEED"]=="7" and kwargs["timeout"]==35
        return SimpleNamespace(returncode=0,stdout=b'{"signature":"f(name, league=PL)","log_text":"VERBATIM\\n"}',stderr=b"raw stderr")
    rec=a.run_child(["python","synthetic"],tmp_path,7,runner=fake)
    assert rec["exact_command"]==["python","synthetic"]
    assert json.loads(rec["stdout"])["log_text"]=="VERBATIM\n"
    assert rec["stderr"]=="raw stderr" and rec["utc_start"]<=rec["utc_completion"]

def test_child_timeout_retains_context(tmp_path):
    def fake(*args,**kwargs):raise subprocess.TimeoutExpired(args[0],35,output=b"partial",stderr=b"timeout")
    r=a.run_child(["python","synthetic"],tmp_path,1,runner=fake)
    assert r["exit_code"]==124 and r["stdout"]=="partial" and "retained" in r["context"]

def test_seed_sweep_fresh_process_real_synthetic_tie(tmp_path):
    records=[];outputs=[]
    for seed in a.SEEDS:
        work=tmp_path/str(seed);work.mkdir()
        r=a.run_child([sys.executable,"-B","-c","import json;print(json.dumps(next(iter({'FC A','FC B','FC C'}))))"],work,seed,timeout=10)
        assert r["exit_code"]==0
        value=json.loads(r["stdout"]);outputs.append(value)
        records.append(r)
    retain_receipt("tie-processes",records)
    assert len(records)==20 and len(set(outputs))>1
    dist=a.distributions([{"variant":"SYNTHETIC","name":"tie","output":v} for v in outputs])
    assert sum(dist[0]["outputs"].values())==20

def test_token_collision_and_dead_alias_flags():
    rows=a.token_collision_pairs(["AC PISA","LIVERPOOL"],["AC MILAN","ARSENAL"],lambda x,y:85.5 if "AC " in x and "AC " in y else 20)
    assert rows==[{"provider":"AC PISA","canonical":"AC MILAN","shared_tokens":["AC"],"similarity":85.5}]
    aliases={"L":{2025:{"CAFÉ":"X","KÃ¶LN":"Y","STRAßE":"Z"}}}
    flags=a.alias_key_flags(aliases)
    assert len(flags)==3 and flags[1]["mojibake_marker"]
    assert flags[2]["upper_still_non_ascii"] is False

@pytest.mark.parametrize("field",["home_score","away_score","odds","time","status","corners"])
def test_whitelist_refuses_every_extra_cell(field):
    with pytest.raises(ValueError):a.name_date_cells(b"ignored",{0:field})

def test_date_names_only_skip_poison_cells():
    assert a.name_date_cells(b'2025-08-01,A,B,\xff,\xfe',{0:"date",1:"home",2:"away"})=={"date":"2025-08-01","home":"A","away":"B"}

@pytest.mark.parametrize("root",["data","src","scripts","tests"])
def test_writer_refuses_production_roots(tmp_path,root):
    with pytest.raises(ValueError):a.write_new(tmp_path,tmp_path/root/a.PACKET_NAME,"out","no")

def test_exclusive_writer_and_traversal(tmp_path):
    packet=tmp_path/"tmp"/a.PACKET_NAME
    a.write_new(tmp_path,packet,"file.txt","first")
    with pytest.raises(FileExistsError):a.write_new(tmp_path,packet,"file.txt","second")
    with pytest.raises(ValueError):a.write_new(tmp_path,packet,"../escape","no")
    with pytest.raises(ValueError):a.write_new(tmp_path,packet,tmp_path/"absolute","no")

def synthetic_driver_setup(tmp_path,source,provider_name="AC PISA 1909"):
    repo=Path(__file__).resolve().parents[2]
    actual_packet=repo/"tmp/BL1-T3-offline-dependency-manifest-binding-20261027-01/NEXT-37b-normaliser-probe-02"
    sandbox=tmp_path/"sandbox";module=sandbox/"src/utils/naming.py";module.parent.mkdir(parents=True)
    module.write_text(source,encoding="utf-8")
    maps=sandbox/"data/processed/season_maps.json";maps.parent.mkdir(parents=True);maps.write_text("{}",encoding="utf-8")
    inputs={"probes":[{"kind":"provider_surface","league":"PL","name":provider_name}],
        "providers":[{"league":"PL","id":1,"name":provider_name}],"reference_ids":[],"absorbed":[]}
    inp=sandbox/"inputs.json";inp.write_text(json.dumps(inputs),encoding="utf-8")
    work=sandbox/"work";work.mkdir()
    argv=[sys.executable,"-B",str(actual_packet/"sandbox/probe_driver.py"),
        "--module",str(module),"--sandbox",str(sandbox),"--runtime",str(repo/"venv"),
        "--inputs",str(inp),"--variant","SYNTHETIC","--seed","0"]
    return argv,work,maps

def test_real_harness_synthetic_module_logs_and_surface(tmp_path):
    source="""import logging
from pathlib import Path
SEASON_MAPS_PATH=Path(__file__).resolve().parents[2]/'data/processed/season_maps.json'
logger=logging.getLogger(__name__)
_normalize_warned=set()
LEAGUE_TEAMS={'PL':{2025:{'AC MILAN'}},'PD':{2025:{'ESPANOL','ESPANYOL'}}}
LEAGUE_ALIASES={'PL':{2025:{'CAFÉ':'MISSING','KÃ¶LN':'OTHER'}},'PD':{}}
def normalize_team_name(name,league=None,season=None):
    clean=name.upper()
    candidates=[clean]
    canonical_set=LEAGUE_TEAMS[league][2025]
    logger.warning('VERBATIM '+name)
    return next(iter(canonical_set))
"""
    argv,work,_=synthetic_driver_setup(tmp_path,source)
    r=a.run_child(argv,work,0,timeout=20)
    retain_receipt("synthetic-driver",r)
    assert r["exit_code"]==0,r["stderr"]
    result=json.loads(r["stdout"])
    assert result["calls"][0]["signature"]=="normalize_team_name('AC PISA 1909', league='PL', season=2025)"
    assert result["calls"][0]["log_text"]=="VERBATIM AC PISA 1909\n"
    assert any(x["shared_tokens"]==["AC"] for x in result["mechanisms"]["M2_clearing_pairs"])
    assert len(result["mechanisms"]["M5_non_ascii_aliases"])==2
    assert result["write_attempts"]==result["network_attempts"]==[]

def test_real_harness_dynamic_map_write_refused(tmp_path):
    source="""from pathlib import Path
p=Path(__file__).resolve().parents[2]/'data/processed/season_maps.json'
getattr(p,'write'+'_'+'text')('MUTATE')
"""
    argv,work,maps=synthetic_driver_setup(tmp_path,source)
    r=a.run_child(argv,work,0,timeout=20)
    retain_receipt("dynamic-write",r)
    assert r["exit_code"]==2 and "SANDBOX_WRITE_DENIED" in r["stderr"]
    assert maps.read_text(encoding="utf-8")=="{}"

def test_original_short_pair_is_negative_regression_not_trace_loss(tmp_path):
    source="""import logging
from pathlib import Path
SEASON_MAPS_PATH=Path(__file__).resolve().parents[2]/'data/processed/season_maps.json'
logger=logging.getLogger(__name__)
_normalize_warned=set()
LEAGUE_TEAMS={'PL':{2025:{'AC MILAN'}},'PD':{2025:{'ESPANOL','ESPANYOL'}}}
LEAGUE_ALIASES={}
def normalize_team_name(name,league=None,season=None):
    clean=name.upper()
    candidates=[clean]
    logger.warning('VERBATIM '+name)
    return next(iter(LEAGUE_TEAMS[league][2025]))
"""
    argv,work,_=synthetic_driver_setup(tmp_path,source,provider_name="AC PISA")
    r=a.run_child(argv,work,0,timeout=20);retain_receipt("short-pair-regression",r)
    assert r["exit_code"]==0
    result=json.loads(r["stdout"])
    assert result["calls"][0]["trace"]["candidates"]==["AC PISA"]
    assert result["calls"][0]["log_text"]=="VERBATIM AC PISA\n"
    assert result["mechanisms"]["M2_clearing_pairs"]==[]
    assert result["mechanisms"]["M2_shared_token_below85"][0]["score"]==pytest.approx(66.66666666666667)

def test_refusal_stack_metadata_never_needs_source_lookup(tmp_path):
    source="""from pathlib import Path
p=Path(__file__).resolve().parents[2]/'data/processed/season_maps.json'
getattr(p,'write'+'_'+'text')('MUTATE')
"""
    argv,work,maps=synthetic_driver_setup(tmp_path,source)
    r=a.run_child(argv,work,0,timeout=20);retain_receipt("stack-metadata-regression",r)
    assert r["exit_code"]==2
    error=json.loads(r["stderr"])
    assert error["error"]=="SANDBOX_WRITE_DENIED"
    assert error["traceback_frames_no_source_reads"]
    assert maps.read_text(encoding="utf-8")=="{}"

def test_fixture_scope_is_builtin_and_packet_local(tmp_path,request,tmp_path_factory):
    repo=Path(__file__).resolve().parents[2]
    assert tmp_path.resolve().is_relative_to(tmp_path_factory.getbasetemp().resolve())
    assert not tmp_path.resolve().is_relative_to(repo/".tmp")
    assert not _RECEIPT_DIR.is_relative_to(tmp_path_factory.getbasetemp().resolve())
    assert request._get_active_fixturedef("tmp_path").func.__module__=="_pytest.tmpdir"

def test_inherited_catalog_identity_is_required_not_rebased(tmp_path):
    path=str((tmp_path/"data/processed/season_maps.json").resolve())
    identity={"path":path,"size_bytes":2,"sha256":"pinned"}
    assert a.catalog_pin_from_gate(tmp_path,{"checks":[{"observed":identity}]})==identity
    with pytest.raises(ValueError):a.catalog_pin_from_gate(tmp_path,{"checks":[]})

def test_checkpoint_leaf_refuses_escape(tmp_path):
    with pytest.raises(ValueError,match="Unsafe checkpoint"):
        a.prepare(tmp_path,tmp_path/"tmp"/a.PACKET_NAME,"../outside")

