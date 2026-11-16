import copy
import pytest
from src.data_foundation.identity import normalize_name,match_name,validate_mapping,validate_rebuilt,final_score

CAT={"PL":[{"id":1,"name":"Atlético Example","shortName":"Atletico","tla":"ATM"},
           {"id":2,"name":"Other FC","shortName":"Other","tla":"OTH"}],
     "PD":[{"id":3,"name":"Other FC","shortName":"Other","tla":"OTH"}]}

@pytest.mark.parametrize("name,expected",[("  ATLÉTICO.   Example! ","atletico example"),("A-C","ac"),("AC","ac"),(" Straẞe ","strasse")])
def test_exact_normalization(name,expected):
    assert normalize_name(name)==expected

@pytest.mark.parametrize("name",["ATLÉTICO EXAMPLE","Atletico","ATM"])
def test_provider_name_short_tla(name):
    assert match_name("PL",name,CAT)["provider_team_ids"]==[1]

def test_no_fuzzy_or_cross_league_search():
    assert match_name("PL","Othr",CAT)["status"]=="UNMATCHED"
    assert match_name("BL1","Other FC",CAT)["status"]=="UNMATCHED"
    assert match_name("PD","Other FC",CAT)["provider_team_ids"]==[3]

def test_explicit_alias_and_conflicting_alias_not_override():
    alias={"league":"PL","source_name":"Legacy Club","provider_team_id":1,"note":"synthetic reviewed alias"}
    assert match_name("PL","Legacy Club",CAT,[alias])["status"]=="MATCHED"
    conflict={**alias,"source_name":"Other FC"}
    assert match_name("PL","Other FC",CAT,[conflict])["status"]=="AMBIGUOUS"
    assert match_name("PL","Legacy Club",CAT,[{**alias,"provider_team_id":999}])["status"]=="INVALID_ALIAS"

def test_normalization_collision_regression_two_distinct_sources():
    teams={"PL":[{"id":1,"name":"AC"},{"id":2,"name":"A-C"}]}
    assert match_name("PL","AC",teams)["status"]=="AMBIGUOUS"
    mapped=[{"league":"PL","source_name":"AC","provider_team_id":1},
            {"league":"PL","source_name":"A-C","provider_team_id":1}]
    types={x["type"] for x in validate_mapping(mapped)}
    assert "NORMALIZATION_COLLISION_REVIEW" in types and "ID_TO_MULTIPLE_SOURCE_NAMES" in types

def rows(day="2023-01-01"):
    return [{"league":"PL","season":"synthetic","date":day,"home_team_id":1,"away_team_id":2,"home_source_name":"A","away_source_name":"B","home_score":1,"away_score":0},
            {"league":"PL","season":"synthetic","date":"2023-01-02","home_team_id":2,"away_team_id":1,"home_source_name":"B","away_source_name":"A","home_score":0,"away_score":0}]

def validate(data,provider=None,expected=2):
    return validate_rebuilt(data,expected_team_counts={("PL","synthetic"):expected},provider_rows=copy.deepcopy(data) if provider is None else provider)

def test_valid_synthetic_double_round_robin():
    result=validate(rows())
    assert result["validation_passed"] and result["unsealed_score_comparisons"]==2

@pytest.mark.parametrize("mutation,kind",[
    ("self","SELF_FIXTURE"),("same_date","TEAM_MULTIPLE_MATCHES_DATE"),
    ("pair","REPEATED_ORDERED_PAIR"),("team_count","TEAM_COUNT_MISMATCH"),
    ("score","SCORE_MISMATCH_UNSEALED"),("duplicate","DUPLICATE_FIXTURE"),
    ("missing_match","MATCH_COUNT_MISMATCH"),
])
def test_validator_failure_modes(mutation,kind):
    data=rows();provider=copy.deepcopy(data);expected=2
    if mutation=="self":data[0]["away_team_id"]=1
    if mutation=="same_date":data[1]["date"]=data[0]["date"]
    if mutation=="pair":data[1].update(home_team_id=1,away_team_id=2)
    if mutation=="team_count":expected=3
    if mutation=="score":data[0]["home_score"]=9
    if mutation=="duplicate":data.append(copy.deepcopy(data[0]))
    if mutation=="missing_match":data.pop()
    result=validate(data,provider,expected)
    assert kind in {x["type"] for x in result["findings"]} and not result["validation_passed"]

def test_sealed_score_getters_never_touched():
    class Poison(dict):
        def __getitem__(self,key):
            if "score" in key:raise AssertionError("SEALED_SCORE_READ")
            return super().__getitem__(key)
    a=rows();a[0]["date"]="2024-07-01";a[1]["date"]="2024-07-02"
    data=[Poison(x) for x in a];provider=[Poison(x) for x in a]
    result=validate_rebuilt(data,expected_team_counts={("PL","synthetic"):2},provider_rows=provider)
    assert result["validation_passed"] and result["sealed_score_comparisons_skipped"]==2
    assert result["unsealed_score_comparisons"]==0

def test_missing_official_expected_team_count_is_unverified():
    result=validate_rebuilt(rows(),expected_team_counts={},provider_rows=rows())
    assert "TEAM_COUNT_UNVERIFIED" in {x["type"] for x in result["findings"]}

def test_missing_mapping_does_not_silently_pass():
    data=rows()
    for row in data:
        row.pop("home_source_name");row.pop("away_source_name")
    result=validate(data)
    assert "NAME_ID_MAPPING_UNVERIFIED" in {x["type"] for x in result["findings"]}

@pytest.mark.parametrize("value",[1.5,-1,float("nan"),True])
def test_fractional_or_invalid_scores_rejected(value):
    with pytest.raises(ValueError):final_score(value)

def test_name_to_multiple_ids_is_not_accepted():
    data=[{"league":"PL","source_name":"A","provider_team_id":1},{"league":"PL","source_name":"A","provider_team_id":2}]
    assert "NAME_TO_MULTIPLE_IDS" in {x["type"] for x in validate_mapping(data)}
