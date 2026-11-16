"""E17 features shared by training and serving; goals/results/dates and stable IDs only."""
from collections import defaultdict
from copy import deepcopy
from datetime import date
import math
from .clock import CanonicalError, utc, digest

BASE_NAMES = tuple(f"{side}_rolling_{metric}_{window}" for side in ("home","away")
                   for window in (3,5,10) for metric in ("goals_scored","goals_conceded"))
# Freeze-C order has form after each side's six rolling statistics.
FEATURE_NAMES = tuple(name for side in ("home","away")
    for name in [*(f"{side}_rolling_{metric}_{window}" for window in (3,5,10)
                  for metric in ("goals_scored","goals_conceded")), f"{side}_form_rating"]) + (
    "home_elo","away_elo","elo_difference")
FEATURE_VERSION = "E17-goals-date-prefix-v1"
RULES = {"initial":1500.0,"home_advantage":75.0,"scale":400.0,"K":20.0}
ALPHA = 1-0.9


def margin_multiplier(margin):
    return 1.0 if margin<=1 else 1.5 if margin==2 else 1.75 if margin==3 else 1.75+(margin-3)/8


def _fresh():
    return {"elo":1500.0,"for":[],"against":[],"form":None,"matches":0}


def _key(league,team):
    return str(league),str(team)


def _vector(states,league,home,away):
    h=states.get(_key(league,home),_fresh());a=states.get(_key(league,away),_fresh())
    values=[]
    for state in (h,a):
        for window in (3,5,10):
            for metric in ("for","against"):
                prior=state[metric][-window:]
                values.append(math.fsum(prior)/len(prior) if prior else 1.35)
        values.append(state["form"] if state["form"] is not None else 1.5)
    values.extend([h["elo"],a["elo"],h["elo"]-a["elo"]])
    if len(values)!=17 or not all(math.isfinite(x) for x in values):
        raise CanonicalError("INVALID_E17_VECTOR")
    return values


def _advance_day(states,rows):
    occupied=set();changes=defaultdict(float)
    for row in sorted(rows,key=lambda x:(x["league"],str(x["id"]))):
        hk,ak=_key(row["league"],row["home_id"]),_key(row["league"],row["away_id"])
        if hk==ak or hk in occupied or ak in occupied:
            raise CanonicalError("DUPLICATE_TEAM_DATE")
        occupied.update((hk,ak))
        h,a=row["home_goals"],row["away_goals"]
        if any(isinstance(v,bool) or not isinstance(v,int) or v<0 for v in (h,a)):
            raise CanonicalError("INVALID_FEATURE_RESULT")
        hs=states.setdefault(hk,_fresh());aws=states.setdefault(ak,_fresh())
        expected=1/(1+10**(-(hs["elo"]+75-aws["elo"])/400))
        actual=1.0 if h>a else .5 if h==a else 0.0
        delta=20*margin_multiplier(abs(h-a))*(actual-expected)
        changes[hk]+=delta;changes[ak]-=delta
        for state,scored,conceded in ((hs,h,a),(aws,a,h)):
            points=3.0 if scored>conceded else 1.0 if scored==conceded else 0.0
            state["for"]=(state["for"]+[float(scored)])[-10:]
            state["against"]=(state["against"]+[float(conceded)])[-10:]
            state["form"]=points if state["form"] is None else .9*state["form"]+ALPHA*points
            state["matches"]+=1
    for key,delta in changes.items():states[key]["elo"]+=delta


def training_features(rows):
    """Every day's vectors precede that day's result updates; no fitted transformation here."""
    groups=defaultdict(list)
    for row in rows:
        day=date.fromisoformat(row["day"]).isoformat()
        groups[day].append(row)
    states={};vectors={}
    for day,records in sorted(groups.items()):
        for row in records:
            if row["id"] in vectors:raise CanonicalError("DUPLICATE_TRAINING_FIXTURE")
            vectors[row["id"]]=_vector(states,row["league"],row["home_id"],row["away_id"])
        _advance_day(states,records)
    seed={"feature_version":FEATURE_VERSION,"last_day":max(groups) if groups else None,
          "states":[{"league":league,"team_id":team,"state":state} for (league,team),state in sorted(states.items())]}
    return vectors,seed


class FeatureEngine:
    def __init__(self,seed,history,*,asof):
        if seed.get("feature_version")!=FEATURE_VERSION:
            raise CanonicalError("FEATURE_SEED_VERSION_MISMATCH")
        self.seed=deepcopy(seed);self.asof=utc(asof);self.groups=defaultdict(list);self.known=[]
        seen=set()
        for row in history:
            observed=utc(row["observed_utc"])
            if observed>self.asof:
                raise CanonicalError("FUTURE_RESULT_IN_HISTORY")
            day=utc(row["kickoff_utc"])[:10]
            if seed.get("last_day") and day<=seed["last_day"]:
                raise CanonicalError("SEED_HISTORY_OVERLAP")
            if row["fixture_id"] in seen:raise CanonicalError("DUPLICATE_HISTORY_FIXTURE")
            seen.add(row["fixture_id"])
            self.groups[day].append({"id":row["fixture_id"],"league":row["league"],
              "home_id":row["home_id"],"away_id":row["away_id"],"home_goals":row["home_goals"],"away_goals":row["away_goals"]})
            self.known.append(observed)
        self.cache={}

    def vector(self,fixture):
        day=utc(fixture["kickoff_utc"])[:10]
        if self.seed.get("last_day") and day<=self.seed["last_day"]:
            raise CanonicalError("FORECAST_NOT_AFTER_MODEL_TRAINING")
        if day not in self.cache:
            states={_key(row["league"],row["team_id"]):deepcopy(row["state"]) for row in self.seed["states"]}
            for before,records in sorted(self.groups.items()):
                if before<day:_advance_day(states,records)
            self.cache[day]=states
        values=_vector(self.cache[day],fixture["league"],fixture["home_id"],fixture["away_id"])
        return values,{"feature_hash":digest([FEATURE_VERSION,values]),"history_known_utc":max(self.known) if self.known else None}

