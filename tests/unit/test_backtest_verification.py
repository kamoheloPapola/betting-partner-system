
import unittest
import pandas as pd
import numpy as np
import logging
from src.ml.trainer import ModelTrainer
from src.ml.registry import ModelRegistry
from src.backtest.engine import Backtester
from src.features.pipeline import FeaturePipeline
from unittest.mock import MagicMock, patch

class TestBacktestVerification(unittest.TestCase):
    def setUp(self):
        self.registry = MagicMock(spec=ModelRegistry)
        self.trainer = ModelTrainer(self.registry)
        self.backtester = Backtester()
        # Mocking components to prevent disk I/O
        self.backtester.trainer = self.trainer

    def test_train_model_in_memory_guards(self):
        """Verify ModelTrainer.train_model_in_memory sample size guard."""
        df_small = pd.DataFrame({
            'home_score': [1] * 40,  # Only 40 matches
            'feat1': np.random.rand(40)
        })
        
        with self.assertRaises(ValueError) as cm:
            self.trainer.train_model_in_memory(df_small, 'home_score', features=['feat1'])
        self.assertIn("Insufficient training data", str(cm.exception))

        # Check with enough data
        df_large = pd.DataFrame({
            'home_score': [1] * 60,
            'feat1': np.random.rand(60)
        })
        model = self.trainer.train_model_in_memory(df_large, 'home_score', features=['feat1'])
        self.assertIsNotNone(model)

    @patch('src.features.pipeline.FeaturePipeline.load_stored_matches')
    def test_backtest_run_leak_logic(self, mock_load):
        """Smoke test for Backtester.run walk-forward logic."""
        # Create dummy raw data
        raw_rows = []
        # 3 Seasons (2021, 2022, 2023)
        for season in [2021, 2022, 2023]:
            for i in range(100):
                raw_rows.append({
                    'match_id': f"{season}_{i}",
                    'season': str(season),
                    'competition': 'Premier League',
                    'date': pd.to_datetime(f"{season}-08-01") + pd.Timedelta(days=i),
                    'home_team': f"TeamA_{i % 10}",
                    'away_team': f"TeamB_{i % 10}",
                    'home_score': np.random.randint(0, 4),
                    'away_score': np.random.randint(0, 4),
                    'home_corners': 5, 'away_corners': 5,
                    'home_total_cards': 2, 'away_total_cards': 2
                })
        
        mock_load.return_value = pd.DataFrame(raw_rows)
        
        # Run backtest: Train on 2021, 2022. Test on 2023.
        results = self.backtester.run(train_seasons=[2021, 2022], test_season=2023, league="PL")
        
        self.assertFalse(results.empty)
        self.assertTrue('actual' in results.columns)
        self.assertTrue('pred' in results.columns)
        # Check if Brier scores were collected
        self.assertTrue('brier_home' in results.columns)
        
        # Verify batching (test match count should match 2023 data)
        self.assertEqual(len(results), 100)

if __name__ == '__main__':
    unittest.main()
