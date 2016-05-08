import unittest
import pandas as pd
import numpy as np

import dmc.features as features
import dmc.preprocessing as preprocessing


class FeaturingTest(unittest.TestCase):
    def setUp(self):
        raw_data = pd.read_csv('tests/test_data.txt', delimiter=';')
        train_ids = ['a1000001', 'a1000002', 'a1000003']
        test_ids = ['a1000007', 'a1000008']

        self.data = preprocessing.cleanse(raw_data)
        self.train, self.test = preprocessing.split_train_test(self.data, train_ids, test_ids)

    def test_group_return_probability(self):
        group = pd.Series([0, 100])
        self.assertEqual(features.group_return_probability(group), 0.5)

        with self.assertRaises(ZeroDivisionError):
            group = pd.Series([])
            features.group_return_probability(group)

    def test_apply_return_probs(self):
        train = pd.DataFrame({'colorCode': [0, 0, 4, 4, 4, 4, 5, 5, 6, 6],
                              'returnQuantity': [0, 1, 0, 1, 1, 1, 1, 0, 0, 1]})
        test = pd.DataFrame({'colorCode': [1, 3, 4, 6]})
        train_enriched, test_enriched = features.apply_return_probs(train, test,
                                                                    'colorCode', 'colorReturnProb')
        # Check probability calculation
        self.assertListEqual(train_enriched['colorReturnProb'].loc[[0, 2]].tolist(), [0.5, 0.75])
        # Check unknown values in test data
        self.assertTrue(np.isnan(test_enriched['colorReturnProb'][0]))

    def test_binned_color_code_return_probability(self):
        train1 = pd.DataFrame({'colorCode': [0, 0, 4, 4, 4, 4, 5, 5, 6, 6],
                               'returnQuantity': [0, 1, 0, 1, 1, 1, 1, 0, 0, 1]})
        test1 = pd.DataFrame({'colorCode': [1, 3, 4, 6]})
        train1, test1 = features.binned_color_code_return_probability(train1, test1)
        self.assertListEqual(test1['colorReturnProb'].tolist(), [0.5, 0.5, 0.75, 0.5])
        self.assertListEqual(test1['binnedColorCode'].tolist(), [0, 0, 1, 2])

        train2 = pd.DataFrame({'colorCode': [0, 0, 1, 1, 2],
                               'returnQuantity': [0, 1, 0, 1, 1]})
        test2 = pd.DataFrame({'colorCode': [0, 4]})
        train1, test1 = features.binned_color_code_return_probability(train2, test2)
        self.assertAlmostEqual(test1['colorReturnProb'][1], 0.6)
        self.assertListEqual(test1['binnedColorCode'].tolist(), [0, 2])

    def test_add_dependent_features(self):
        train, test = features.add_dependent_features(self.train, self.test)
        self.assertIn('customerReturnProb', test.columns)
        self.assertIn('productGroupReturnProb', test.columns)
        self.assertIn('colorReturnProb', test.columns)
        self.assertIn('sizeReturnProb', test.columns)
        self.assertIn('customerReturnProb', train.columns)
        self.assertIn('productGroupReturnProb', train.columns)
        self.assertIn('colorReturnProb', train.columns)
        self.assertIn('sizeReturnProb', train.columns)

    def test_add_independent_features(self):
        data = features.add_independent_features(self.data)
        self.assertIn('totalOrderShare', data.columns)
