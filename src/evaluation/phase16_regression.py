"""Publish the v2 factual regression while preserving the Phase12-15 record."""
from src.evaluation import phase16_regression_facts as facts
from src.evaluation.phase12_operations import put


def publish():
    return put(facts.RUN / 'regression/BL1-565839.json', facts.reconstruct())
