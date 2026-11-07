"""Generate the offline FFPS Phase 2 count-market evidence bundle.

The runner loads local manifest artifacts directly and writes only beneath the
selected Phase 2 baseline directory. It never calls providers or the registry.
"""

from __future__ import annotations
import argparse
import json, math, logging, hashlib, inspect
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run-id",default="phase2-count-markets-79110f9a-20260811-v1")
parser.add_argument("--phase1-run-id",default="phase1-pl-1x2-79110f9a-20240801-v1")
args=parser.parse_args()
OUT=ROOT/"baselines"/"phase2"/args.run_id
OUT.mkdir(parents=True,exist_ok=True)
logging.disable(logging.WARNING)
import src.cli.commands.prediction as pred
from src.ml.count_domains import sanitize_corner_count_mean,sanitize_card_count_mean
from src.ml.distributions import NegativeBinomialEngine,ZeroInflatedEngine
from src.simulation.match_simulator import clamp_lambda

def clean(v):
    if isinstance(v,dict): return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)): return [clean(x) for x in v]
    if isinstance(v,np.integer): return int(v)
    if isinstance(v,(np.floating,float)): return float(v) if np.isfinite(v) else None
    return v
def dump(name,payload):
    path=OUT/name
    text=json.dumps(clean(payload),indent=2,sort_keys=True)+"\n"
    path.write_text(text,encoding="utf-8")
    return {"path":path.relative_to(ROOT).as_posix(),"sha256":hashlib.sha256(text.encode()).hexdigest()}

mpath=ROOT/"src/ml/models/manifest.json"
manifest=json.loads(mpath.read_text(encoding="utf-8"))
roles={"mh_corn":"mh_corn_PL","ma_corn":"ma_corn_PL","m_corners_lgbm":"m_corners_lgbm_PL","m_corners_xgb":"m_corners_xgb_PL","m_cards_lgbm":"cards_PL","m_cards_xgb":"cards_xgb_PL"}
suite={"m_cards":None}
artifacts={}
for role,slot in roles.items():
    key=manifest["active_models"][slot]
    meta=dict(manifest[key])
    path=ROOT/"src/ml/models"/meta["filename"]
    model=joblib.load(path)
    try: model.meta=meta
    except Exception: pass
    suite[role]=model
    artifacts[role]={"slot":slot,"manifest_key":key,"path":path.relative_to(ROOT).as_posix(),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"target":meta.get("target"),"feature_count":len(meta.get("features",[])),"feature_order_sha256":hashlib.sha256(json.dumps(meta.get("features",[]),separators=(",",":")).encode()).hexdigest()}

fpath=ROOT/"data/features/feature_matrix.csv"
df=pd.read_csv(fpath,low_memory=False)
pl=df[df["league"].astype(str).eq("PL")].copy()
fxpath=ROOT/"baselines"/"phase1"/args.phase1_run_id/"fixture_predictions.jsonl"
fixture_ids=[]
with fxpath.open(encoding="utf-8") as h:
    for line in h:
        if line.strip(): fixture_ids.append(str(json.loads(line)["fixture"]["match_hash"]))
idset=set(fixture_ids)
sample=pl[pl["match_hash"].astype(str).isin(idset)].drop_duplicates("match_hash").sort_values(["date","match_hash"])
if len(sample)!=len(idset): raise RuntimeError(f"join mismatch fixtures={len(idset)} rows={len(sample)}")

class W:
    def add(self,*args,**kwargs): pass
class CNB(NegativeBinomialEngine):
    calls=[]
    def calculate_probabilities(self,mh,vh,ma,va):
        type(self).calls.append((float(mh),float(vh),float(ma),float(va)))
        return super().calculate_probabilities(mh,vh,ma,va)
class OldZIP(ZeroInflatedEngine):
    def calculate_probabilities(self,mu_total,pi_zero=.03,var_mult=1.0):
        x=np.arange(self.max_count+1); probs=self.pmf(x,mu_total*var_mult,pi_zero)
        u45=self._apply_policy(np.sum(probs[:5]),"CARDS_U45")
        o45=1-u45
        o25=self._apply_policy(np.sum(probs[3:]),"CARDS_O25")
        u55=self._apply_policy(np.sum(probs[:6]),"CARDS_U55")
        return {"cards_under_4_5":self._clamp(u45),"cards_over_4_5":self._clamp(o45),"cards_over_2_5":self._clamp(o25),"cards_under_5_5":self._clamp(u55,max_p=self.CLAMP_CEILING_U55)}
class COldZIP(OldZIP):
    calls=[]
    def calculate_probabilities(self,mu_total,pi_zero=.03,var_mult=1.0):
        type(self).calls.append((float(mu_total),float(pi_zero),float(var_mult)))
        return super().calculate_probabilities(mu_total,pi_zero,var_mult)
class CNewZIP(ZeroInflatedEngine):
    calls=[]
    def calculate_probabilities(self,mu_total,pi_zero=.03,var_mult=1.0):
        type(self).calls.append((float(mu_total),float(pi_zero),float(var_mult)))
        return super().calculate_probabilities(mu_total,pi_zero,var_mult)
def old_corner(v,*,context=""):
    x=float(v); return float(clamp_lambda(x)) if "CornersTotal" in context else x
def old_card(v,*,context=""):
    x=float(v); return float(clamp_lambda(x)) if context.endswith(":Cards:Ensemble") else x
projection_block='''        # H2H, calibration and soft-cap layers operate independently. Project
        # the exposed nested events back onto a coherent probability contract.
        if card_u55 is not None:
            card_o25 = max(card_o25, 1.0 - ZeroInflatedEngine.CLAMP_CEILING_U55)
            card_u55 = max(card_u55, 1.0 - card_o25)
            card_u45 = min(max(card_u45, 1.0 - card_o25), card_u55)

'''
old_cards_source=inspect.getsource(pred._calc_cards)
if old_cards_source.count(projection_block)!=1: raise RuntimeError('old card wrapper projection marker mismatch')
old_cards_globals=dict(pred.__dict__)
exec(old_cards_source.replace(projection_block,''),old_cards_globals)
old_calc_cards=old_cards_globals['_calc_cards']

def calc(row,old):
    pred.NegativeBinomialEngine=CNB; CNB.calls.clear()
    if old:
        pred.sanitize_corner_count_mean=old_corner; pred.sanitize_card_count_mean=old_card
        pred.ZeroInflatedEngine=COldZIP; COldZIP.calls.clear()
    else:
        pred.sanitize_corner_count_mean=sanitize_corner_count_mean; pred.sanitize_card_count_mean=sanitize_card_count_mean
        pred.ZeroInflatedEngine=CNewZIP; CNewZIP.calls.clear()
    ctx=f"Phase2:{row.get('match_hash')}"
    corn,_=pred._calc_corners(row,suite,ctx,"PL",False,W())
    if old:
        old_cards_globals["sanitize_card_count_mean"]=old_card
        old_cards_globals["ZeroInflatedEngine"]=COldZIP
        card,_=old_calc_cards(row,suite,ctx,"PL",False,W())
    else:
        card,_=pred._calc_cards(row,suite,ctx,"PL",False,W())
    nb=CNB.calls[-1]; z=(COldZIP.calls if old else CNewZIP.calls)[-1]
    return {"corner_mean":nb[0]+nb[2],"corner_home_mean":nb[0],"corner_away_mean":nb[2],"card_mean":z[0],**corn,**card}

rows=[]
for _,row in sample.iterrows():
    rows.append({"match_hash":str(row["match_hash"]),"date":str(row.get("date")),"home_team":str(row.get("home_team")),"away_team":str(row.get("away_team")),"old":calc(row,True),"new":calc(row,False)})
markets=["corn_u11","corn_o75","corn_1x2_h","corn_1x2_d","corn_1x2_a","card_u45","card_o25","card_u55"]
means=["corner_mean","corner_home_mean","corner_away_mean","card_mean"]
thresholds={"corn_u11":[.85],"corn_o75":[.70,.85],"card_o25":[.60,.70],"card_u55":[.70,.75]}
def stats(vals):
    a=np.asarray(vals,dtype=float)
    return {"min":float(a.min()),"median":float(np.median(a)),"p95":float(np.quantile(a,.95)),"max":float(a.max()),"mean":float(a.mean())}
summary={"schema_version":1,"purpose":"Controlled offline before/after of Phase 2 count-domain repair; not performance validation.","population":{"source":fxpath.relative_to(ROOT).as_posix(),"feature_source":"data/features/feature_matrix.csv","league":"PL","fixture_count":len(rows),"phase1_fixture_count":len(idset),"join_complete":len(rows)==len(idset)},"models":artifacts,"old_behavior_emulation":{"corner":"Goal clamp [0.05,4.0] on both specialist members and ensemble; team members and postprocessing unbounded.","card":"Goal clamp [0.05,4.0] on specialist ensemble; legacy ZIP and wrapper returned independently damped/capped market probabilities without final coherence projection."},"means":{},"probability_impacts":{},"threshold_crossings":{}}
for k in means:
    a=[r["old"][k] for r in rows]; b=[r["new"][k] for r in rows]
    summary["means"][k]={"old":stats(a),"new":stats(b),"changed_count":sum(abs(x-y)>1e-12 for x,y in zip(a,b)),"old_at_or_below_4_new_above_4":sum(x<=4+1e-12 and y>4+1e-12 for x,y in zip(a,b))}
for k in markets:
    pairs=[(float(r["old"][k]),float(r["new"][k])) for r in rows]
    ds=np.array([b-a for a,b in pairs])
    summary["probability_impacts"][k]={"changed_count":int(np.sum(np.abs(ds)>1e-12)),"mean_delta":float(ds.mean()),"mean_abs_delta":float(np.abs(ds).mean()),"p95_abs_delta":float(np.quantile(np.abs(ds),.95)),"max_abs_delta":float(np.abs(ds).max()),"min_old":min(a for a,b in pairs),"max_old":max(a for a,b in pairs),"min_new":min(b for a,b in pairs),"max_new":max(b for a,b in pairs)}
    for t in thresholds.get(k,[]):
        summary["threshold_crossings"][f"{k}@{t}"]={"old_below_new_at_or_above":sum(a<t<=b for a,b in pairs),"old_at_or_above_new_below":sum(a>=t>b for a,b in pairs),"note":"Probability-floor crossing only; not an end-to-end selection claim."}
summary["changed_fixture_count_any_market"]=sum(any(abs(float(r["old"][k])-float(r["new"][k]))>1e-12 for k in markets) for r in rows)
ranked=[]
for r in rows:
    ranked.append({"max_abs_probability_delta":max(abs(float(r["new"][k])-float(r["old"][k])) for k in markets),**r})
summary["largest_changes"]=sorted(ranked,key=lambda x:x["max_abs_probability_delta"],reverse=True)[:20]
summary["all_probability_outputs_finite_bounded"]=all(math.isfinite(float(r[s][k])) and 0<=float(r[s][k])<=1 for r in rows for s in ("old","new") for k in markets)
summary["new_nested_card_contract_holds"]=all(r["new"]["card_o25"]+1e-12>=1-r["new"]["card_u45"] and r["new"]["card_u45"]<=r["new"]["card_u55"]+1e-12 for r in rows)
impact=dump("impact_analysis.json",summary)

target={}
for name,col in {"home_goals":"home_score","away_goals":"away_score","home_corners":"home_corners","away_corners":"away_corners","home_cards":"home_cards","away_cards":"away_cards"}.items():
    vals=pd.to_numeric(df[col],errors="coerce").dropna(); target[name]={"count":len(vals),**stats(vals)}
for name,a,b in (("total_corners","home_corners","away_corners"),("total_cards","home_cards","away_cards")):
    vals=(pd.to_numeric(df[a],errors="coerce")+pd.to_numeric(df[b],errors="coerce")).dropna(); target[name]={"count":len(vals),**stats(vals)}
targets=dump("historical_target_ranges.json",{"source":"data/features/feature_matrix.csv","source_sha256":hashlib.sha256(fpath.read_bytes()).hexdigest(),"row_count":len(df),"ranges":target,"conclusion":"Observed corner/card targets materially exceed 4; the goal-domain upper bound is not a defensible count-domain contract."})

def controlled_cases():
    old_corner_total=4.0
    new_corner_total=.6*10.0+.4*8.0
    old_h,old_a=6.0*(old_corner_total/10.0),4.0*(old_corner_total/10.0)
    new_h,new_a=6.0*(new_corner_total/10.0),4.0*(new_corner_total/10.0)
    nb=NegativeBinomialEngine()
    old_corner_probs=nb.calculate_probabilities(old_h,old_h*1.3,old_a,old_a*1.3)
    new_corner_probs=nb.calculate_probabilities(new_h,new_h*1.3,new_a,new_a*1.3)
    old_card_total=4.0
    new_card_total=.6*6.0+.4*5.0
    old_card_probs=OldZIP().calculate_probabilities(old_card_total,pi_zero=.03)
    new_card_probs=ZeroInflatedEngine().calculate_probabilities(new_card_total,pi_zero=.03)
    return [{"case":"corner_specialist_count_domain","inputs":{"team_home_mean":6.0,"team_away_mean":4.0,"specialist_lgbm_mean":10.0,"specialist_xgb_mean":8.0,"weights":{"lgbm":.6,"xgb":.4}},"old_specialist_total":old_corner_total,"new_specialist_total":new_corner_total,"old_distribution_inputs":{"home_mean":old_h,"away_mean":old_a},"new_distribution_inputs":{"home_mean":new_h,"away_mean":new_a},"old_probabilities":old_corner_probs,"new_probabilities":new_corner_probs,"calculation":"0.6*10 + 0.4*8; old goal clamp versus finite-positive corner sanitizer.","deterministic":True},{"case":"card_specialist_count_domain","inputs":{"specialist_lgbm_mean":6.0,"specialist_xgb_mean":5.0,"weights":{"lgbm":.6,"xgb":.4},"pi_zero":.03,"intensity_multiplier":1.0},"old_specialist_total":old_card_total,"new_specialist_total":new_card_total,"old_probabilities":old_card_probs,"new_probabilities":new_card_probs,"calculation":"0.6*6 + 0.4*5; old goal clamp/independent market postprocessing versus finite-positive card sanitizer/coherence projection.","deterministic":True}]

cases_1=controlled_cases()
cases_2=controlled_cases()
goldens=dump("controlled_goldens.json",{"purpose":"Deterministic engine-path regression cases; behavior evidence, not a performance endorsement.","models":artifacts,"random_seed":None,"simulation_count":None,"cases":cases_1,"repetition":{"run_1":cases_1,"run_2":cases_2,"exact_match":cases_1==cases_2}})
print(json.dumps({"impact":impact,"targets":targets,"goldens":goldens,"fixtures":len(rows),"changed":summary["changed_fixture_count_any_market"]},indent=2))
