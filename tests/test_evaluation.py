import unittest
import numpy as np
import pandas as pd

from dmc.evaluation import gini_ratio, dmc_cost, dmc_cost_relative, \
    precision, features, column_purities


class DMCCostTest(unittest.TestCase):
    def setUp(self):
        self.a = pd.Series([1, 2, 3, 4, 5])
        self.b = pd.Series([0, 2, 4, 4, 3])
        self.df = pd.DataFrame({
            'A': ['a', 'a', 'b', 'b', 'b'],
            'B': [0, 0, 1, 1, 2],
            'returnQuantity': [1, 0, 5, 0, 0]
        })

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
        self.assertAlmostEqual(0.8, np.round(gini_ratio(self.a), decimals=1))
        self.assertAlmostEqual(0.72, np.round(gini_ratio(self.b), decimals=2))

    def test_features(self):
        purities = features(self.df)
        self.assertEqual(1 / 3, purities.loc[('A', 'b'), 'retProb'])
        self.assertEqual(2.5, purities.loc[('B', 1), 'avgRet'])
        self.assertEqual(2.5, purities.loc[('B', 1), 'avgRet'])
        self.assertEqual(2.89, np.round(purities.loc[('A', 'b'), 'stdRet'], 2))

    def test_column_purities(self):
        purities = column_purities(self.df)
        self.assertEqual(0.47, np.round(purities['A'], 2))
        self.assertEqual(0.4, purities['B'])
