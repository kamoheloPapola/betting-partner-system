"""Portable E17 Poisson models: JSON coefficients, no pickle execution on serving."""
from importlib import metadata
import hashlib
import json
import math
import os
import warnings
from pathlib import Path

from .clock import CanonicalError, canonical, digest, finite_positive
from .features import FEATURE_NAMES, FEATURE_VERSION, training_features
from .markets import MARKET_VERSION

MODEL_SCHEMA = "canonical.E17.poisson.v1"
PARAMS = {"alpha":1.0,"fit_intercept":True,"solver":"lbfgs","max_iter":1000,"tol":1e-8}
TARGET_CAP = 6
TRAINING_END = "2024-07-01"


def read_artifact(path, expected_sha256=None, *, require_validation=True):
    path=Path(path)
    if path.name.startswith(".env"):
        raise CanonicalError("SECRET_IS_NOT_MODEL")
    with path.open("rb") as f:
        before=os.fstat(f.fileno())
        if before.st_size>8_000_000:raise CanonicalError("MODEL_ARTIFACT_TOO_LARGE")
        raw=f.read()
        after=os.fstat(f.fileno())
    sha=hashlib.sha256(raw).hexdigest()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns) or len(raw)!=before.st_size:
        raise CanonicalError("MODEL_CHANGED_DURING_READ")
    if expected_sha256 is not None and sha!=expected_sha256:
        raise CanonicalError("MODEL_ARTIFACT_HASH_MISMATCH")
    try:document=json.loads(raw)
    except (ValueError,UnicodeError) as exc:raise CanonicalError("MODEL_ARTIFACT_PARSE_FAILED") from exc
    model=Model(document["model"])
    gate=document.get("validation",{})
    if require_validation and (gate.get("status")!="PASS_LOCKED_REPLAY" or gate.get("model_id")!=model.model_id):
        raise CanonicalError("MODEL_NOT_VALIDATED")
    return model,{"path":str(path.resolve()),"size_bytes":len(raw),"sha256":sha},gate


class Model:
    def __init__(self,payload):
        self.payload=json.loads(canonical(payload))
        if (payload.get("schema")!=MODEL_SCHEMA or payload.get("feature_names")!=list(FEATURE_NAMES)
            or payload.get("feature_version")!=FEATURE_VERSION or payload.get("market_version")!=MARKET_VERSION
            or payload.get("parameters")!=PARAMS or payload.get("target_cap")!=TARGET_CAP):
            raise CanonicalError("MODEL_RECIPE_MISMATCH")
        self.model_id=digest(self.payload)
        for side in ("home","away"):
            values=payload["coefficients"][side]
            if any(len(values[k])!=17 for k in ("mean","scale","coef")):
                raise CanonicalError("MODEL_COEFFICIENT_SHAPE")
            numbers=[*values["mean"],*values["scale"],*values["coef"],values["intercept"]]
            if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in numbers) or any(x<=0 for x in values["scale"]):
                raise CanonicalError("MODEL_COEFFICIENT_INVALID")
        for league,baseline in payload["baseline_1x2"].items():
            if len(baseline)!=3 or any(not math.isfinite(v) or v<=0 for v in baseline) or abs(sum(baseline)-1)>1e-12:
                raise CanonicalError("INVALID_FROZEN_BASELINE")

    @property
    def seed(self):
        return self.payload["feature_seed"]

    def predict(self,vector):
        if len(vector)!=17 or not all(math.isfinite(v) for v in vector):
            raise CanonicalError("INVALID_E17_VECTOR")
        predictions=[]
        for side in ("home","away"):
            p=self.payload["coefficients"][side]
            linear=p["intercept"]+math.fsum(c*((x-m)/s) for x,m,s,c in zip(vector,p["mean"],p["scale"],p["coef"]))
            try:rate=math.exp(linear)
            except OverflowError as exc:raise CanonicalError("MODEL_RATE_OVERFLOW") from exc
            predictions.append(finite_positive(rate))
        return tuple(predictions)


def fit_model(rows,*,source_manifest_sha256,identity_audit):
    """Fixed E recipe only. This does not adopt or score a sealed test window."""
    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import PoissonRegressor
    from sklearn.exceptions import ConvergenceWarning
    if len(rows)<1000 or any(r["day"]>=TRAINING_END for r in rows):
        raise CanonicalError("MODEL_TRAINING_WINDOW_OR_SAMPLE_INVALID")
    vectors,seed=training_features(rows)
    x=np.asarray([vectors[r["id"]] for r in rows],dtype=float)
    coefficients={}
    fitted={}
    for side in ("home","away"):
        y=np.asarray([min(r[side+"_goals"],TARGET_CAP) for r in rows],dtype=float)
        estimator=Pipeline([("scale",StandardScaler()),("poisson",PoissonRegressor(**PARAMS))])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            estimator.fit(x,y)
        if any(issubclass(w.category,ConvergenceWarning) for w in caught):
            raise CanonicalError("MODEL_FIT_DID_NOT_CONVERGE")
        scale,poisson=estimator["scale"],estimator["poisson"]
        coefficients[side]={"mean":scale.mean_.tolist(),"scale":scale.scale_.tolist(),
                            "coef":poisson.coef_.tolist(),"intercept":float(poisson.intercept_)}
        fitted[side]=estimator
    counts={}
    for row in rows:
        values=counts.setdefault(row["league"],[1,1,1])
        values[0 if row["home_goals"]>row["away_goals"] else 2 if row["home_goals"]<row["away_goals"] else 1]+=1
    baseline={league:[v/sum(values) for v in values] for league,values in counts.items()}
    payload={"schema":MODEL_SCHEMA,"feature_names":list(FEATURE_NAMES),"feature_version":FEATURE_VERSION,
      "market_version":MARKET_VERSION,"parameters":PARAMS,"target_cap":TARGET_CAP,
      "coefficients":coefficients,"feature_seed":seed,"baseline_1x2":baseline,
      "baseline_definition":"Same-league training 1X2 frequencies with one pseudocount per class; frozen at fit",
      "training":{"n":len(rows),"minimum_date":min(r["day"] for r in rows),"maximum_date":max(r["day"] for r in rows),
                  "source_manifest_sha256":source_manifest_sha256,"identity_audit":identity_audit},
      "packages":{name:metadata.version(name) for name in ("numpy","scikit-learn")}}
    model=Model(payload)
    expected=np.column_stack([fitted[s].predict(x) for s in ("home","away")])
    actual=np.asarray([model.predict(vector) for vector in x])
    if not np.allclose(actual,expected,rtol=1e-12,atol=1e-12):
        raise CanonicalError("PORTABLE_MODEL_PARITY_FAILED")
    return model,{"portable_prediction_parity":"PASS","max_absolute_error":float(np.max(np.abs(actual-expected))),
                  "training_rows":len(rows),"model_id":model.model_id}

