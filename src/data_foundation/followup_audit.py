"""NEXT-37 diagnostic-only helpers. No production matching logic is mirrored."""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import ast
import csv
import gzip
import io
import json
import os
import re
import subprocess
import sys
from .audit import load_metadata, project_metadata

PACKET_NAME="NEXT-37b-normaliser-probe-02"
LEAGUES=("PL","BL1","FL1","SA","PD")
SEEDS=tuple(range(20))

def utc_now():return datetime.now(timezone.utc).isoformat()

def pin(path):
    import hashlib
    path=Path(path)
    with path.open("rb") as stream:
        before=os.fstat(stream.fileno());digest=hashlib.sha256();size=0
        for chunk in iter(lambda:stream.read(65536),b""):digest.update(chunk);size+=len(chunk)
        after=os.fstat(stream.fileno())
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns) or size!=before.st_size:
        raise ValueError(f"Held input changed file={path}")
    return {"path":str(path.resolve()),"size_bytes":size,"sha256":digest.hexdigest()}

def write_new(repo,packet,leaf,body):
    repo,packet=Path(repo).resolve(),Path(packet).resolve()
    if packet.name!=PACKET_NAME or not packet.is_relative_to(repo/"tmp"):
        raise ValueError("Production-root/packet boundary refusal")
    leaf=Path(leaf)
    if leaf.is_absolute() or ".." in leaf.parts:raise ValueError("Traversal refused")
    path=(packet/leaf).resolve()
    if path==packet or not path.is_relative_to(packet):raise ValueError("Packet escape")
    if path.exists() or path.is_symlink():raise FileExistsError(f"Immutable output exists: {path}")
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("xb") as stream:stream.write(body if isinstance(body,bytes) else body.encode("utf-8"))
    return pin(path)

def emit(repo,packet,name,value):
    data=json.dumps(value,ensure_ascii=False,separators=(",",":")).encode("utf-8")
    if len(data)>20000:name+=".gz";data=gzip.compress(data,mtime=0)
    return write_new(repo,packet,name,data)

def load_ref(path):
    path=Path(path)
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix==".gz" else path.read_bytes())

def csv_text(rows,fields):
    stream=io.StringIO(newline="");writer=csv.DictWriter(stream,fieldnames=fields)
    writer.writeheader();writer.writerows(rows);return stream.getvalue()

def name_date_cells(raw,indices):
    if not set(indices.values()).issubset({"date","home","away"}):
        raise ValueError("Only date/name cells; score/odds/time/status forbidden")
    return project_metadata(raw,indices)

def quarantine_rows(plans):
    selected=[]
    for row in plans:
        ids=sorted(set(row["candidate_fixture_ids"]))
        if len(ids)<2 or row["proposed_home_id"] is not None or row["proposed_away_id"] is not None:continue
        selected.append({key:row[key] for key in ("source_sha256","line","date","original_home","original_away")})
        selected[-1]["candidate_fixture_ids"]=ids
    return selected

def filename_hits(text):
    short=r"(?<![A-Za-z0-9])(?:E0|D1|I1|SP1|F1)_(?:[A-Za-z0-9{])|[\"'](?:E0|D1|I1|SP1|F1)[\"']"
    long=r"PL_season_|Bundesliga_Season_|La_Liga_Season_|Serie_A_Season_|Ligue(?:_?1)?_Season_"
    rows=[]
    for n,line in enumerate(text.splitlines(),1):
        s=re.findall(short,line);l=re.findall(long,line,re.I)
        if s or l:rows.append({"line":n,"short_matches":s,"long_matches":l})
    return rows

def static_sandbox_gate(source,expected_hash,actual_hash,dependency_available=True):
    if expected_hash!=actual_hash:return {"status":"NOT_RUN","reason":"Source hash mismatch"}
    if not dependency_available:return {"status":"NOT_RUN","reason":"RapidFuzz unavailable; no installation"}
    tree=ast.parse(source);imports=[];src_imports=[];forbidden=[];opens=[]
    for node in ast.walk(tree):
        if isinstance(node,(ast.Import,ast.ImportFrom)):
            modules=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module or ""]
            imports.extend(modules);src_imports.extend(m for m in modules if m=="src" or m.startswith("src."))
            if any(m.split(".")[0] in {"socket","requests","urllib","http","dotenv"} for m in modules):
                forbidden.append({"line":node.lineno,"reason":"Network/environment import"})
        if isinstance(node,ast.Attribute) and node.attr in {"environ","getenv","putenv"}:
            forbidden.append({"line":node.lineno,"reason":"Environment access"})
        if isinstance(node,ast.Call):
            name=node.func.attr if isinstance(node.func,ast.Attribute) else node.func.id if isinstance(node.func,ast.Name) else ""
            if name in {"write","write_text","write_bytes","rename","unlink","remove","mkdir","makedirs","__import__","eval","exec"} or (name=="replace" and isinstance(node.func,ast.Attribute) and ast.unparse(node.func.value)=="os"):
                forbidden.append({"line":node.lineno,"reason":"Filesystem write/mutation call"})
            if name=="open":
                position=0 if isinstance(node.func,ast.Attribute) else 1
                mode=next((x.value for x in node.keywords if x.arg=="mode"),node.args[position] if len(node.args)>position else ast.Constant("r"))
                if not isinstance(mode,ast.Constant) or not isinstance(mode.value,str) or any(c in mode.value for c in "wax+"):
                    forbidden.append({"line":node.lineno,"reason":"Write/dynamic open mode"})
                opens.append({"line":node.lineno,"expression":ast.unparse(node)})
    if src_imports:return {"status":"NOT_RUN","reason":"Additional src closure requires independent pin/review","src_imports":src_imports}
    if forbidden:return {"status":"NOT_RUN","reason":"Unsafe source access/mutation","findings":forbidden}
    # Real source is additionally pinned and reviewed against the single explicit map read.
    return {"status":"PASS","reason":"Pinned static source has no additional src/network/env/write path","imports":imports,"opens":opens}

def checked_copy(repo,packet,original,leaf,expected):
    observed=pin(original)
    if any(observed[k]!=expected[k] for k in ("size_bytes","sha256")):raise ValueError(f"Copy source mismatch file={original}")
    result=write_new(repo,packet,leaf,Path(original).read_bytes())
    if result["sha256"]!=expected["sha256"]:raise ValueError(f"Copied bytes mismatch file={result['path']}")
    return result

def child_environment(work,seed=None):
    env={k:v for k,v in os.environ.items() if k.upper() in {"SYSTEMROOT","WINDIR","COMSPEC","PATH","PATHEXT"}}
    env.update(PYTHONDONTWRITEBYTECODE="1",PYTHONIOENCODING="utf-8",TMP=str(work),TEMP=str(work))
    if seed is not None:env["PYTHONHASHSEED"]=str(seed)
    return env

def run_child(argv,work,seed,runner=subprocess.run,timeout=35):
    start=utc_now();env=child_environment(work,seed)
    try:
        result=runner(argv,cwd=work,env=env,capture_output=True,timeout=timeout)
        return {"exact_command":argv,"cwd":str(work),"child_environment":env,
            "utc_start":start,"utc_completion":utc_now(),"exit_code":result.returncode,
            "stdout":result.stdout.decode("utf-8",errors="replace"),"stderr":result.stderr.decode("utf-8",errors="replace")}
    except subprocess.TimeoutExpired as exc:
        return {"exact_command":argv,"cwd":str(work),"child_environment":env,
            "utc_start":start,"utc_completion":utc_now(),"exit_code":124,
            "stdout":(exc.stdout or b"").decode("utf-8",errors="replace"),
            "stderr":(exc.stderr or b"").decode("utf-8",errors="replace"),"context":"Sandbox child timeout; retained, do not retry automatically"}

def distributions(records,key_fields=("variant","kind","league","name","style","season")):
    grouped=defaultdict(Counter)
    for row in records:
        key=tuple(row.get(k) for k in key_fields);grouped[key][row.get("output")]+=1
    return [{**dict(zip(key_fields,k)),"outputs":dict(v)} for k,v in grouped.items()]

def alias_key_flags(aliases):
    rows=[]
    for league,seasons in aliases.items():
        for season,entries in seasons.items():
            for key,target in entries.items():
                if any(ord(c)>127 for c in key):
                    rows.append({"league":league,"season":season,"key":key,"target":target,
                        "non_ascii":True,"mojibake_marker":any(x in key for x in ("Ã","Â","�")),
                        "upper_still_non_ascii":any(ord(c)>127 for c in key.upper())})
    return rows

def catalog_pin_from_gate(repo,before):
    expected=str((Path(repo)/"data/processed/season_maps.json").resolve())
    found=[x["observed"] for x in before["checks"] if x["observed"]["path"]==expected]
    if not found or len({(x["size_bytes"],x["sha256"]) for x in found})!=1:
        raise ValueError("Required catalog identity missing/conflicting in current gate")
    return found[0]

def token_collision_pairs(provider_names,canonical_names,scorer):
    """Diagnostic surface only; never chooses a canonical identity."""
    rows=[]
    for provider in provider_names:
        for canonical in canonical_names:
            shared=sorted(set(re.findall(r"[A-Z0-9]+",provider.upper())) & set(re.findall(r"[A-Z0-9]+",canonical.upper())))
            value=scorer(provider,canonical)
            if shared and value>=85:rows.append({"provider":provider,"canonical":canonical,"shared_tokens":shared,"similarity":value})
    return rows

def prepare(repo,packet,checkpoint_leaf=""):
    repo,packet=Path(repo).resolve(),Path(packet).resolve()
    if checkpoint_leaf and not re.fullmatch(r"[a-z0-9-]+",checkpoint_leaf):raise ValueError("Unsafe checkpoint leaf")
    prefix=checkpoint_leaf+"/" if checkpoint_leaf else ""
    reg=load_ref(packet/"preregistration.json")
    if pin(packet/"HYPOTHESES.md")["sha256"]!=reg["hypotheses"]["sha256"]:raise ValueError("Preregistration changed")
    before=load_ref(packet/"preservation-before.json");catalog_pin=catalog_pin_from_gate(repo,before)
    source=repo/"src/utils/naming.py";source_pin=pin(source)
    gate=static_sandbox_gate(source.read_text(encoding="utf-8"),"7785b0aeeac10a91567464d502f6714424f1e39b24f17bf38a0ca859f32fafe9",source_pin["sha256"])
    emit(repo,packet,prefix+"sandbox-static-gate.json",gate)
    if gate["status"]!="PASS":raise ValueError("Sandbox static gate failed; do not run")
    catalog=load_ref(catalog_pin["path"]);p3=[];csvpins=[]
    for league in LEAGUES:
        rows,errors,pinned=load_metadata(repo/"data/processed/matches"/f"{league}_2025.csv","canonical",league)
        if errors:raise ValueError(f"Catalog comparison metadata error league={league} file={pinned['path']}")
        csvpins.append(pinned);actual={n for row in rows for n in (row.home,row.away)};members=set(catalog[league].get("2025",[]))
        p3.append({"league":league,"catalog_count":len(members),"processed_names_count":len(actual),
            "catalog_only":sorted(members-actual),"processed_only":sorted(actual-members),"status":"PASS" if members==actual else "FAIL"})
    emit(repo,packet,prefix+"P3-exact-membership.json",p3)
    emit(repo,packet,prefix+"P3-owner-disposition.json",{"utc":utc_now(),"original_prediction":"WITHDRAWN_BY_NEXT37b","comparison_role":"Descriptive membership differences for static M8; no primary P3 verdict","replacement":"Owner M8 static criterion investigation"})
    diagnostics=[]
    for x in p3:
        rows,_,_=load_metadata(repo/"data/processed/matches"/f"{x['league']}_2025.csv","canonical",x["league"])
        actual={n.upper() for row in rows for n in (row.home,row.away)};members={n.upper() for n in catalog[x["league"]].get("2025",[])}
        diagnostics.append({"league":x["league"],"casefold_catalog_only":sorted(members-actual),"casefold_processed_only":sorted(actual-members),"equal":members==actual})
    emit(repo,packet,prefix+"P3-casefold-secondary.json",diagnostics)
    prior35=packet.parent/"NEXT-35-repair-plan-audit-01";prior36=packet.parent/"NEXT-36-root-cause-audit-01"
    sims=load_ref(prior35/"selected-final/results/simulation.json.gz");pl=load_ref(prior36/"results/H-L.json.gz")
    all_sims={s["league"]:s for s in sims};all_sims["PL"]=pl
    quarantine=[];counts=[]
    for league in LEAGUES:
        plans=all_sims[league]["plans"];picked=quarantine_rows(plans)
        for row in picked:row["league"]=league
        quarantine.extend(picked);counts.append({"league":league,"selected":len(picked),"total":len(plans)})
    emit(repo,packet,prefix+"quarantine-dry-run.json",{"rule":"QUARANTINE_RULE_V1","counts":counts,"rows":quarantine,
        "status":"PASS" if {(r["league"],r["line"]) for r in quarantine}=={("PD",i) for i in (179,182,295,296)} else "FAIL","applied":False})
    write_new(repo,packet,prefix+"quarantine-dry-run.csv",csv_text([{**r,"candidate_fixture_ids":json.dumps(r["candidate_fixture_ids"])} for r in quarantine],["source_sha256","line","date","original_home","original_away","candidate_fixture_ids","league"]))
    with (prior36/"alias-review-sheet-v2.csv").open(encoding="utf-8",newline="") as stream:
        reader=csv.DictReader(stream);fields=reader.fieldnames;sheet=list(reader)
    for row in sheet:
        row["unresolved_rows"]=sum(r["league"]==row["league"] and row["canonical_name"] in (r["original_home"],r["original_away"]) for r in quarantine)
    write_new(repo,packet,prefix+"alias-review-sheet-v3.csv",csv_text(sheet,fields+["unresolved_rows"]))
    # Provider names are reused from already pinned NEXT-34 metadata, no new body/status reads.
    p34=packet.parent/"NEXT-34-snapshot-metadata-audit-01"
    metadata=load_ref(p34/"results-final/T2-snapshot-self-validation.json.gz")
    providers=[{"league":s["league"],"id":c["id"],"name":n} for s in metadata if s["file_season"]==2025 for c in s["per_club"] for n in c["names"]]
    k=load_ref(prior36/"results/H-K.json")
    absorbed=[{"kind":"absorbed","league":x["league"],"name":x["absorbed_provider_name"],"provider_id":x["provider_id"],"merged_label":x["evidence_label_string"],"merged_id":x["merged_label_id"]} for x in k["pairs"]]
    controls=[{"kind":"alias_control","league":r["league"],"name":r["provider_name"],"provider_id":int(r["candidate_id"])} for r in sheet]
    unknown=[{"kind":"synthetic_unknown","league":l,"name":n} for l,n in [("PD","Zzyzx Real FC"),("SA","Qwerty AC Town"),("PD","Nowhere Madrid")]]
    canonical=[{"kind":"canonical_control","league":l,"name":n} for l in LEAGUES for n in sorted(catalog[l].get("2025",[]))]
    full=[{"kind":"provider_surface",**p} for p in providers]
    edges=[{"kind":"edge","league":l,"name":n} for l,n in [("PD",""),("PD","   "),("PD","   "),("PD","  real madrid  "),("BL1","Bayern München"),("BL1","Bayern MÃ¼nchen"),("PD","Club AtlÃ©tico de Madrid")]]
    references=[{"league":r["league"],"canonical_name":r["canonical_name"],"id":int(r["candidate_id"]),"basis":"CANDIDATE_NOT_ADOPTED"} for r in sheet]
    references += [{"league":x["league"],"canonical_name":x["evidence_label_string"],"id":x["merged_label_id"],"basis":"OWNER_SUPPLIED_MERGED_LABEL_REFERENCE"} for x in k["pairs"]]
    inputs={"probes":absorbed+controls+unknown+canonical+full+edges,"providers":providers,"absorbed":absorbed,"reference_ids":references,
        "odds_utils":"NOT_RUN: optional wrapper is not in naming.py's src import closure; copying/importing it would extend the strict sandbox closure",
        "seed_plan":list(SEEDS)}
    write_new(repo,packet,"sandbox/probe-inputs.json",json.dumps(inputs,ensure_ascii=False))
    a_source=checked_copy(repo,packet,source,"sandbox/src/utils/naming.py",source_pin)
    a_map=checked_copy(repo,packet,Path(catalog_pin["path"]),"sandbox/data/processed/season_maps.json",catalog_pin)
    b_source=checked_copy(repo,packet,source,"sandbox/variant-b/src/utils/naming.py",source_pin)
    variant={l:{s:names for s,names in seasons.items() if s!="2025"} for l,seasons in catalog.items()}
    b_map=write_new(repo,packet,"sandbox/variant-b/data/processed/season_maps.json",json.dumps(variant,ensure_ascii=False,sort_keys=True,separators=(",",":")))
    setup={"source_original":source_pin,"map_original":catalog_pin,"variants":{"A":{"source":a_source,"map":a_map,"classification":"BYTE_IDENTICAL_ORIGINAL_MAP"},"B":{"source":b_source,"map":b_map,"classification":"SYNTHETIC_2025_REMOVED_ONLY"}},
        "inputs":pin(packet/"sandbox/probe-inputs.json"),"driver":pin(packet/"sandbox/probe_driver.py"),"gate":gate,"csv_pins":csvpins,"P3":"WITHDRAWN_DESCRIPTIVE_ONLY"}
    emit(repo,packet,"sandbox-setup.json",setup)
    return setup

def preserve_after_child(repo,packet,setup):
    before=load_ref(Path(packet)/"preservation-before.json");unique={}
    for x in before["checks"]:unique[x["observed"]["path"]]=x["observed"]
    stopped=load_ref(before["stopped_packet_index"]["path"])
    for x in stopped["files"]:unique[x["path"]]=x
    unique[setup["map_original"]["path"]]=setup["map_original"]
    for key in ("inputs","driver"):unique[setup[key]["path"]]=setup[key]
    for v in setup["variants"].values():
        for k in ("source","map"):unique[v[k]["path"]]=v[k]
    bad=[];observations=[]
    for expected in unique.values():
        actual=pin(expected["path"]);observations.append((actual["path"],actual["size_bytes"],actual["sha256"]))
        if any(actual[k]!=expected[k] for k in ("size_bytes","sha256")):bad.append({"expected":expected,"observed":actual})
    import hashlib
    return {"utc":utc_now(),"checked_unique_files":len(unique),"reference":"preservation-before.json + sandbox-setup.json",
        "observations_digest":hashlib.sha256(json.dumps(sorted(observations),separators=(",",":")).encode()).hexdigest(),
        "mismatches":bad,"status":"FAIL" if bad else "PASS","method":"Actual held-stream observations; digest is not attestation"}

def run_seed(repo,packet,variant,seed):
    repo,packet=Path(repo).resolve(),Path(packet).resolve()
    if variant not in ("A","B") or seed not in SEEDS:raise ValueError("Unregistered variant/seed")
    setup=load_ref(packet/"sandbox-setup.json");v=setup["variants"][variant]
    for key in ("inputs","driver"):
        if pin(setup[key]["path"])["sha256"]!=setup[key]["sha256"]:raise ValueError("Sandbox harness/input changed")
    for key in ("source","map"):
        actual=pin(v[key]["path"])
        if actual["sha256"]!=v[key]["sha256"]:raise ValueError(f"Sandbox pre-run hash mismatch variant={variant} key={key}")
    work=packet/"sandbox"/"work"/variant/f"seed-{seed:02d}"
    work.mkdir(parents=True,exist_ok=False)
    argv=[str(repo/"venv/Scripts/python.exe"),"-B",str(packet/"sandbox/probe_driver.py"),
        "--module",v["source"]["path"],"--sandbox",str(packet/"sandbox"),"--runtime",str(repo/"venv"),
        "--inputs",setup["inputs"]["path"],"--variant",variant,"--seed",str(seed)]
    receipt=run_child(argv,work,seed,timeout=40)
    stdout=receipt.pop("stdout");stderr=receipt.pop("stderr")
    stem=f"receipts/{variant}-{seed:02d}"
    receipt["stdout_reference"]=write_new(repo,packet,stem+".stdout.json.gz",gzip.compress(stdout.encode("utf-8"),mtime=0))
    receipt["stderr_reference"]=write_new(repo,packet,stem+".stderr.txt",stderr)
    post=preserve_after_child(repo,packet,setup);receipt["preservation_after"]=post
    emit(repo,packet,stem+".receipt.json",receipt)
    if receipt["exit_code"] or post["status"]!="PASS":
        raise ValueError(f"Sandbox child/gate failed variant={variant} seed={seed}; retained evidence; stop")
    return json.loads(stdout)

