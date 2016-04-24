import unittest
import numpy as np
import pandas as pd
from dmc.evaluation import dmc_cost, dmc_cost_relative, precision, gini_ratio, feature_purities


class DMCCostTest(unittest.TestCase):
    def setUp(self):
        self.a = np.array([1, 2, 3, 4, 5])
        self.b = np.array([0, 2, 4, 4, 3])

    def test_dmc_cost(self):
        res = dmc_cost(self.a, self.b)
        self.assertEqual(res, 4)

    def test_dmc_relative(self):
        res = dmc_cost_relative(self.a, self.b)
        self.assertEqual(res, 0.8)

    def test_precision(self):
        res = precision(self.a, self.b)
        self.assertEqual(res, 0.4)

    def test_gini_ratio(self):
        self.assertAlmostEquals(0.8, np.round(gini_ratio(self.a), decimals=1))
        self.assertAlmostEquals(0.72, np.round(gini_ratio(self.b), decimals=2))

    def test_feature_purity(self):
        df = pd.DataFrame({
            'A': ['a', 'a', 'b', 'b', 'b'],
            'B': [0, 0, 1, 1, 2],
            'labels': [1, 0, 1, 0, 0]
        })
        purities = feature_purities(df, label_col='labels')
        self.assertEquals(0.5, purities['A']['a'])
        self.assertEquals(0.5, purities['B'][0])
