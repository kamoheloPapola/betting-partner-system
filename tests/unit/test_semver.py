
import pytest
from src.ml.registry import SemVer, ModelRegistry

def test_semver_sorting():
    v1 = SemVer("1.0.0")
    v2 = SemVer("1.2.0")
    v3 = SemVer("1.10.0")
    v4 = SemVer("2.0.0")
    
    assert v1 < v2
    assert v2 < v3 # The critical check: 1.2 < 1.10
    assert v3 < v4
    
    versions = [v3, v1, v4, v2]
    versions.sort()
    
    assert versions == [v1, v2, v3, v4]

def test_increment_logic():
    # Mock Registry with fake manifest
    reg = ModelRegistry()
    reg.manifest = {
        "model_v1.0.0": {"name": "test_model", "version": "1.0.0", "league": "PL"},
        "model_v1.1.0": {"name": "test_model", "version": "1.1.0", "league": "PL"}
    }
    
    next_v = reg.get_next_version("test_model", league="PL", increment="minor")
    assert next_v == "1.2.0"
    
    next_major = reg.get_next_version("test_model", league="PL", increment="major")
    assert next_major == "2.0.0"

def test_new_model_version():
    reg = ModelRegistry()
    reg.manifest = {}
    
    next_v = reg.get_next_version("new_model", league="PD")
    assert next_v == "1.0.0"
