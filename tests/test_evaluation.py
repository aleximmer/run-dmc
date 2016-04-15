import unittest
import numpy as np
from dmc.evaluation import dmc_cost, dmc_cost_relative


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
