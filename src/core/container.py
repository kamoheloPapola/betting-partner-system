"""
Service Container (Dependency Injection).

Provides a thread-safe singleton container for shared services like
ModelRegistry and FeaturePipeline. Uses lazy initialization for performance.

Usage:
    from src.core.container import ServiceContainer
    
    container = ServiceContainer.get_instance()
    registry = container.registry
    pipeline = container.pipeline
"""
import logging
import threading
from typing import Optional

from src.ml.registry import ModelRegistry
from src.features.pipeline import FeaturePipeline

# Define public API
__all__ = ["ServiceContainer"]

logger = logging.getLogger(__name__)


class ServiceContainer:
    """
    Thread-safe Dependency Injection Container.
    
    Implements the Singleton pattern with double-checked locking to ensure
    shared services across the application. Services are lazy-loaded on first access.
    
    Attributes:
        registry: Lazy-loaded ModelRegistry instance.
        pipeline: Lazy-loaded FeaturePipeline instance.
    """
    _instance: Optional['ServiceContainer'] = None
    _lock: threading.Lock = threading.Lock()
    _initialized: bool = False
    
    def __init__(self) -> None:
        """
        Initialize the container.
        
        Note: Use get_instance() instead of direct instantiation.
        """
        if ServiceContainer._initialized and ServiceContainer._instance is not None:
            raise RuntimeError(
                "ServiceContainer is a singleton. Use ServiceContainer.get_instance() instead."
            )
        self._registry: Optional[ModelRegistry] = None
        self._pipeline: Optional[FeaturePipeline] = None
        logger.debug("ServiceContainer initialized")
    
    @classmethod
    def get_instance(cls) -> 'ServiceContainer':
        """
        Get the singleton instance of the container.
        
        Thread-safe with double-checked locking pattern.
        
        Returns:
            The singleton ServiceContainer instance.
        """
        if cls._instance is None:
            with cls._lock:
                # Double-check after acquiring lock
                if cls._instance is None:
                    cls._instance = cls()
                    cls._initialized = True
        return cls._instance
    
    @property
    def registry(self) -> ModelRegistry:
        """Lazy-loaded ModelRegistry."""
        if self._registry is None:
            logger.debug("Initializing ModelRegistry")
            self._registry = ModelRegistry()
        return self._registry
    
    @property
    def pipeline(self) -> FeaturePipeline:
        """Lazy-loaded FeaturePipeline."""
        if self._pipeline is None:
            logger.debug("Initializing FeaturePipeline")
            self._pipeline = FeaturePipeline()
        return self._pipeline

    def reset(self) -> None:
        """
        Reset all cached services.
        
        Primarily used for testing to ensure clean state between tests.
        """
        logger.debug("Resetting ServiceContainer services")
        self._registry = None
        self._pipeline = None
    
    @classmethod
    def reset_singleton(cls) -> None:
        """
        Fully reset the singleton instance.
        
        WARNING: Only use in tests. Invalidates all existing references.
        """
        with cls._lock:
            cls._instance = None
            cls._initialized = False
            logger.debug("ServiceContainer singleton reset")

