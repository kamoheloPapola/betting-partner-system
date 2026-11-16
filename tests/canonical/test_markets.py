import math
import pytest
from src.canonical.markets import probabilities,lead_outcome,MARKET_VERSION
from src.canonical.clock import CanonicalError


@pytest.mark.parametrize("h,a",[(0,0),(1.4,1.1),(5,.2),(.2,5),(7.6,7.6)])
def test_markets_are_one_coherent_distribution(h,a):
    p=probabilities(h,a)
    assert p["home_win"]+p["draw"]+p["away_win"]==pytest.approx(1,abs=1e-12)
    assert p["btts_yes"]+p["btts_no"]==pytest.approx(1,abs=1e-12)
    for n in range(6):assert p[f"under_{n}_5"]+p[f"over_{n}_5"]==pytest.approx(1,abs=1e-12)


def test_no_probability_ceiling_or_lambda_clamp():
    p=probabilities(5,.2)
    assert p["expected_home_goals"]==5 and p["home_win"]>.92
    assert probabilities(0,0)["draw"]==1


def test_symmetric_rates_symmetric_outcomes():
    p=probabilities(1.5,1.5)
    assert p["home_win"]==pytest.approx(p["away_win"],abs=1e-14)


@pytest.mark.parametrize("args",[(float("nan"),1),(float("inf"),1),(-1,1),(True,1),(8,1)])
def test_invalid_rates_or_dc_domain_fail_not_substitute(args):
    with pytest.raises(CanonicalError):probabilities(*args)


def test_unknown_version_cannot_reinterpret_a_frozen_forecast():
    with pytest.raises(CanonicalError):probabilities(1,1,version="unknown")


@pytest.mark.parametrize("h,a,side,lead,expected",[
 (0,0,"home",1,"FALSE"),(1,0,"home",1,"TRUE"),(1,1,"home",1,"UNDETERMINED"),
 (2,1,"home",2,"UNDETERMINED"),(2,0,"home",2,"TRUE"),(1,3,"home",2,"FALSE"),
 (3,1,"away",1,"UNDETERMINED"),(0,1,"away",1,"TRUE")])
def test_three_state_lead_outcome(h,a,side,lead,expected):
    assert lead_outcome(h,a,side=side,lead=lead)==expected

