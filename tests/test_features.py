import unittest
import pandas as pd
import numpy as np

import dmc.features as features


class FeaturingTest(unittest.TestCase):
    def setUp(self):
        self.train = pd.DataFrame({'colorCode': [0, 0, 4, 4, 4, 4, 5, 5, 6, 6],
                                   'returnQuantity': [0, 1, 0, 1, 1, 1, 1, 0, 0, 1]})
        self.test = pd.DataFrame({'colorCode': [1, 3, 4, 6]})

    def test_group_return_probability(self):
        group = pd.Series([0, 100])
        self.assertEqual(features.group_return_probability(group), 0.5)

        with self.assertRaises(ZeroDivisionError):
            group = pd.Series([])
            features.group_return_probability(group)

    def test_apply_return_probs(self):
        train, test = features.apply_return_probs(self.train, self.test,
                                                  'colorCode', 'colorReturnProb')
        # Check probability calculation
        self.assertListEqual(train['colorReturnProb'].loc[[0, 2]].tolist(), [0.5, 0.75])
        # Check unknown values in test data
        self.assertTrue(np.isnan(test['colorReturnProb'][0]))

    def test_binned_color_code_return_probability(self):
        train, test = features.binned_color_code_return_probability(self.train, self.test)
        self.assertListEqual(test['colorReturnProb'].tolist(), [0.5, 0.5, 0.75, 0.5])
        self.assertListEqual(test['binnedColorCode'].tolist(), [0, 0, 1, 2])

        train2 = pd.DataFrame({'colorCode': [0, 0, 1, 1, 2],
                               'returnQuantity': [0, 1, 0, 1, 1]})
        test2 = pd.DataFrame({'colorCode': [0, 4]})
        train, test = features.binned_color_code_return_probability(train2, test2)
        self.assertAlmostEqual(test['colorReturnProb'][1], 0.6)
        self.assertListEqual(test['binnedColorCode'].tolist(), [0, 2])
