import logging
import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Append src to path
sys.path.append(os.getcwd())

from src.ml.registry import ModelRegistry

class TestSmartRouting(unittest.TestCase):
    
    def setUp(self):
        # Mock Manifest
        self.mock_manifest = {
            # Global Model (Baseline)
            "nb_corners_global_v1": {
                "name": "nb_corners",
                "version": "v1",
                "league": "Global",
                "status": "productive",
                "filename": "nb_corners_global_v1.pkl",
                "registered_at": "2025-01-01T10:00:00",
                "metrics": {"calibration_score": 0.05} # 5% ECE
            },
            # PL Model (Better)
            "nb_corners_PL_v1": {
                "name": "nb_corners",
                "version": "v1",
                "league": "PL",
                "status": "productive",
                "filename": "nb_corners_PL_v1.pkl",
                "registered_at": "2025-01-02T10:00:00",
                "metrics": {"calibration_score": 0.03} # 3% ECE (Better)
            },
            # SA Model (Worse)
            "nb_corners_SA_v1": {
                "name": "nb_corners",
                "version": "v1",
                "league": "SA",
                "status": "productive",
                "filename": "nb_corners_SA_v1.pkl",
                "registered_at": "2025-01-02T10:00:00",
                "metrics": {"calibration_score": 0.08} # 8% ECE (Worse)
            },
             # BL1 Model (No Score)
            "nb_corners_BL1_v1": {
                "name": "nb_corners",
                "version": "v1",
                "league": "BL1",
                "status": "productive",
                "filename": "nb_corners_BL1_v1.pkl",
                "registered_at": "2025-01-02T10:00:00",
                "metrics": {} # Missing
            }
        }
        
    def test_routing_prefer_local(self):
        """Should pick PL because 0.03 < 0.05"""
        registry = ModelRegistry()
        registry.manifest = self.mock_manifest
        with patch("pathlib.Path.exists", return_value=True):
            meta = registry.get_production_model_for_league("PL", "nb_corners")
        self.assertIsNotNone(meta)
        self.assertEqual(meta['league'], 'PL')
        
    def test_routing_prefer_global(self):
        """Should pick Global because 0.05 < 0.08 (SA is worse)"""
        registry = ModelRegistry()
        registry.manifest = self.mock_manifest
        with patch("pathlib.Path.exists", return_value=True):
            meta = registry.get_production_model_for_league("SA", "nb_corners")
        self.assertIsNotNone(meta)
        self.assertEqual(meta['league'], 'Global')
        
    def test_fallback_missing_local(self):
        """Should pick Global if Local missing"""
        registry = ModelRegistry()
        registry.manifest = self.mock_manifest
        with patch("pathlib.Path.exists", return_value=True):
            meta = registry.get_production_model_for_league("PD", "nb_corners")
        self.assertIsNotNone(meta)
        self.assertEqual(meta['league'], 'Global')
        
    def test_missing_scores(self):
        """Should pick Global because Local has no score (999 vs 0.05)"""
        registry = ModelRegistry()
        registry.manifest = self.mock_manifest
        with patch("pathlib.Path.exists", return_value=True):
            meta = registry.get_production_model_for_league("BL1", "nb_corners")
        self.assertEqual(meta['league'], 'Global')

if __name__ == '__main__':
    logging.basicConfig(level=logging.ERROR)
    unittest.main()
