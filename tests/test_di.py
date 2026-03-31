
import pytest
from src.core.container import ServiceContainer
from src.ml.registry import ModelRegistry
from src.features.pipeline import FeaturePipeline

def test_container_singleton():
    """Verify that get_instance always returns the same object."""
    ServiceContainer._instance = None # Reset
    c1 = ServiceContainer.get_instance()
    c2 = ServiceContainer.get_instance()
    assert c1 is c2

def test_container_registry_lazy_load():
    """Verify registry is created only when accessed."""
    ServiceContainer._instance = None
    c = ServiceContainer.get_instance()
    assert c._registry is None
    
    reg = c.registry
    assert isinstance(reg, ModelRegistry)
    assert c._registry is not None
    
    # Verify cached
    reg2 = c.registry
    assert reg is reg2

def test_container_pipeline_lazy_load():
    """Verify pipeline is created only when accessed."""
    ServiceContainer._instance = None
    c = ServiceContainer.get_instance()
    assert c._pipeline is None
    
    pipe = c.pipeline
    assert isinstance(pipe, FeaturePipeline)
    assert c._pipeline is not None
    
    # Verify cached
    pipe2 = c.pipeline
    assert pipe is pipe2
