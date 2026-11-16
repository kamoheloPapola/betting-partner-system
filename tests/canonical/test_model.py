import json
import pytest
from src.canonical.model import Model,read_artifact,MODEL_SCHEMA,PARAMS,TARGET_CAP
from src.canonical.features import FEATURE_NAMES,FEATURE_VERSION,training_features
from src.canonical.markets import MARKET_VERSION
from src.canonical.clock import CanonicalError,digest


def payload():
    _,seed=training_features([])
    values={"mean":[0.]*17,"scale":[1.]*17,"coef":[0.]*17,"intercept":0.}
    return {"schema":MODEL_SCHEMA,"feature_names":list(FEATURE_NAMES),"feature_version":FEATURE_VERSION,
            "market_version":MARKET_VERSION,"parameters":dict(PARAMS),"target_cap":TARGET_CAP,
            "coefficients":{"home":dict(values),"away":dict(values)},"feature_seed":seed,
            "baseline_1x2":{"PL":[.4,.3,.3]}}


def test_portable_model_has_no_serialized_code(tmp_path):
    p=payload();m=Model(p)
    artifact=tmp_path/"model.json"
    artifact.write_text(json.dumps({"model":p,"validation":{"status":"PASS_LOCKED_REPLAY","model_id":m.model_id}}))
    loaded,_,_=read_artifact(artifact)
    assert loaded.predict([0.]*17)==(1.,1.)


def test_unvalidated_candidate_cannot_serve(tmp_path):
    artifact=tmp_path/"model.json";artifact.write_text(json.dumps({"model":payload(),"validation":{"status":"NOT_RUN"}}))
    with pytest.raises(CanonicalError,match="NOT_VALIDATED"):read_artifact(artifact)


def test_artifact_hash_before_parsing(tmp_path):
    artifact=tmp_path/"model.json";artifact.write_text("not JSON")
    with pytest.raises(CanonicalError,match="HASH_MISMATCH"):read_artifact(artifact,"0"*64)


def test_recipe_and_order_cannot_silently_change():
    p=payload();p["feature_names"]=list(reversed(p["feature_names"]))
    with pytest.raises(CanonicalError,match="RECIPE"):Model(p)


def test_nonfinite_coefficient_rejected():
    p=payload();p["coefficients"]["home"]["coef"][0]=float("inf")
    with pytest.raises((ValueError,CanonicalError)):Model(p)

