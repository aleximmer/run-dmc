import unittest
import pandas as pd
import numpy as np
from pandas.util.testing import assert_frame_equal

from dmc.preprocessing import cleanse, apply_features, clean_ids, split_train_test


class FeaturingTest(unittest.TestCase):
    def setUp(self):
        self.raw_data = pd.read_csv('tests/test_data.txt', delimiter=';')
        self.raw_data = self.raw_data.head(50)
        train_ids = ['a1000001', 'a1000002', 'a1000003']
        test_ids = ['a1000007', 'a1000008']

        clean_data = cleanse(self.raw_data)
        self.data = {'data': clean_data, 'train_ids': train_ids, 'test_ids': test_ids}

    @staticmethod
    def content_equal(a, b):
        try:
            assert_frame_equal(a.sort_index(axis=1), b.sort_index(axis=1), check_names=True)
            return True
        except (AssertionError, ValueError, TypeError):
            return False

    def test_color_return_probability(self):
        processed_data = apply_features(self.data)['train']
        actual_processed = processed_data[['colorCode', 'colorReturnProb']]
        expected_processed = pd.DataFrame({'colorCode': [1972, 3854, 2974, 1992,
                                                         1968, 1972, 1001, 3976],
                                           'colorReturnProb': [0., 0., 0., 1., 0., 0., 0., 0.]})
        self.assertTrue(self.content_equal(actual_processed, expected_processed))

    def test_product_group_return_probability(self):
        processed_data = apply_features(self.data)['train']
        actual_processed = processed_data[['productGroupReturnProb']]
        expected_processed = pd.DataFrame({
            'productGroupReturnProb': [0., 0., 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6]})
        self.assertTrue(self.content_equal(actual_processed, expected_processed))

    def test_size_return_probability(self):
        processed_data = apply_features(self.data)['train']
        actual_processed = processed_data[['sizeReturnProb']]
        expected_processed = pd.DataFrame({
            'sizeReturnProb': [0., 0., 0.5, 0.5, 0, 0, 0, 0]})
        self.assertTrue(self.content_equal(actual_processed, expected_processed))

    def test_is_german_holiday(self):
        processed_data = apply_features(self.data)['train']
        actual_processed = processed_data[['orderIsOnGermanHoliday']]
        expected_processed = pd.DataFrame({'orderIsOnGermanHoliday': [1, 0, 0, 1, 1, 1, 1, 1]})
        self.assertTrue(self.content_equal(actual_processed, expected_processed))

    def test_binned_color_return_probability(self):
        processed = apply_features(self.data)['train']
        self.assertListEqual(['[0, 1992)', '[1993, 10000)', '[1993, 10000)', '[1992, 1993)',
                              '[0, 1992)', '[0, 1992)', '[0, 1992)', '[1993, 10000)'],
                             list(processed.binnedColorCode))
