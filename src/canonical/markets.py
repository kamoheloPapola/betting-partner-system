"""Versioned, deterministic goal markets. No calibration, dampening or market caps."""
import math
from .clock import CanonicalError

MARKET_VERSION = "poisson-dc-raw-v1"
RHO = -0.13
TAIL_TOLERANCE = 1e-12


def _rate(value):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
        raise CanonicalError("INVALID_EXPECTED_GOALS")
    return float(value)


def _pmf(rate):
    if rate==0:
        return [1.0]
    count=max(12,int(rate+12*math.sqrt(rate+1)+20))
    if count>2000:
        raise CanonicalError("MARKET_NUMERICAL_BUDGET")
    result=[math.exp(-rate+k*math.log(rate)-math.lgamma(k+1)) for k in range(count+1)]
    if max(0.0,1-math.fsum(result))>TAIL_TOLERANCE:
        raise CanonicalError("MARKET_TAIL_BOUND_EXCEEDED")
    return result


def _tau(h,a,home,away):
    if h==a==0:return 1-home*away*RHO
    if h==1 and a==0:return 1+away*RHO
    if h==0 and a==1:return 1+home*RHO
    if h==a==1:return 1-RHO
    return 1.0


def probabilities(home_xg,away_xg,*,version=MARKET_VERSION):
    if version!=MARKET_VERSION:
        raise CanonicalError("UNKNOWN_MARKET_VERSION")
    home,away=_rate(home_xg),_rate(away_xg)
    if min(_tau(h,a,home,away) for h in (0,1) for a in (0,1))<0:
        # Reject an invalid Dixon-Coles domain; never silently clamp a rate/change models.
        raise CanonicalError("DIXON_COLES_DOMAIN_INVALID")
    hp,ap=_pmf(home),_pmf(away)
    if len(hp)*len(ap)>4_000_000:
        raise CanonicalError("MARKET_NUMERICAL_BUDGET")
    totals={"home_win":0.0,"draw":0.0,"away_win":0.0,"btts_yes":0.0}
    for threshold in range(6):
        totals[f"under_{threshold}_5"]=0.0
    for threshold in (0,1,2):
        totals[f"home_under_{threshold}_5"]=0.0
        totals[f"away_under_{threshold}_5"]=0.0
    mass=0.0
    for h,p in enumerate(hp):
        for a,q in enumerate(ap):
            v=p*q*_tau(h,a,home,away)
            mass+=v
            totals["home_win" if h>a else "away_win" if a>h else "draw"]+=v
            if h and a:totals["btts_yes"]+=v
            for threshold in range(6):
                if h+a<=threshold:totals[f"under_{threshold}_5"]+=v
            for threshold in (0,1,2):
                if h<=threshold:totals[f"home_under_{threshold}_5"]+=v
                if a<=threshold:totals[f"away_under_{threshold}_5"]+=v
    if not math.isfinite(mass) or mass<=0 or abs(1-mass)>2*TAIL_TOLERANCE:
        raise CanonicalError("INVALID_SCORE_DISTRIBUTION")
    # Only normalize the explicitly bounded numerical truncation, never confidence.
    totals={k:v/mass for k,v in totals.items()}
    for threshold in range(6):
        totals[f"over_{threshold}_5"]=1-totals[f"under_{threshold}_5"]
    totals["btts_no"]=1-totals["btts_yes"]
    totals["double_chance_1x"]=totals["home_win"]+totals["draw"]
    totals["double_chance_x2"]=totals["away_win"]+totals["draw"]
    totals["double_chance_12"]=totals["home_win"]+totals["away_win"]
    return {"version":MARKET_VERSION,"numerical_tail_bound":2*TAIL_TOLERANCE,
            "expected_home_goals":home,"expected_away_goals":away,**totals}


def lead_outcome(home_goals,away_goals,*,side,lead):
    """What final scores logically prove about ever leading; unknown is not false."""
    if side not in ("home","away") or lead not in (1,2):
        raise CanonicalError("UNSUPPORTED_LEAD_MARKET")
    if any(isinstance(x,bool) or not isinstance(x,int) or x<0 for x in (home_goals,away_goals)):
        raise CanonicalError("INVALID_FINAL_SCORE")
    own,other=(home_goals,away_goals) if side=="home" else (away_goals,home_goals)
    if own-other>=lead:return "TRUE"
    if own<lead:return "FALSE"
    return "UNDETERMINED"

