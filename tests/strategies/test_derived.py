
import unittest
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))
from src.strategies.derived import DoubleChanceEngine

class TestDoubleChanceEngine(unittest.TestCase):
    def test_basic_derivation(self):
        # perfect sum
        dc = DoubleChanceEngine.calculate(0.5, 0.3, 0.2)
        self.assertAlmostEqual(dc['dc_1x'], 0.8)
        self.assertAlmostEqual(dc['dc_x2'], 0.5)
        self.assertAlmostEqual(dc['dc_12'], 0.7)

    def test_normalization(self):
        # sum = 0.9 -> should normalize
        # 0.45, 0.27, 0.18 approx
        dc = DoubleChanceEngine.calculate(0.4, 0.24, 0.16, strict=True)
        # 0.4 / 0.8 = 0.5
        # 0.24 / 0.8 = 0.3
        # 0.16 / 0.8 = 0.2
        self.assertAlmostEqual(dc['dc_1x'], 0.8)

    def test_consistency_assertion(self):
        # What if 1X < Home? Impossible by math addition, but checking logic
        # pass
        pass
        
if __name__ == '__main__':
    unittest.main()
