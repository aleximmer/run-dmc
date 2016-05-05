import unittest
import pandas as pd
from pandas.util.testing import assert_frame_equal

from dmc.preprocessing import cleanse, feature, add_features


class PreprocessingTest(unittest.TestCase):
    def setUp(self):
        raw_data = pd.read_csv('tests/test_data.txt', delimiter=';')
        raw_data = raw_data.head(50)
        clean_data = cleanse(raw_data)
        self.data = clean_data

    @staticmethod
    def content_equal(a, b):
        try:
            assert_frame_equal(a.sort_index(axis=1), b.sort_index(axis=1), check_names=True)
            return True
        except (AssertionError, ValueError, TypeError):
            return False

    def test_preprocess(self):
        processed_data = feature(self.data)
        self.assertIn('customerReturnProb', processed_data.columns)
        self.assertIn('totalOrderShare', processed_data.columns)
        self.assertIn('productGroupReturnProb', processed_data.columns)
        self.assertIn('colorReturnProb', processed_data.columns)
        self.assertIn('sizeReturnProb', processed_data.columns)

    def test_color_return_probability(self):
        processed_data = feature(self.data)
        actual_processed = processed_data[['colorCode', 'colorReturnProb']]
        expected_processed = pd.DataFrame({'colorCode': [1972, 3854, 2974, 1992,
                                                         1968, 1972, 1001, 3976],
                                           'colorReturnProb': [0., 0., 0., 1., 0., 0., 0., 0.]})
        self.assertTrue(self.content_equal(actual_processed, expected_processed))

    def test_product_group_return_probability(self):
        processed_data = feature(self.data)
        actual_processed = processed_data[['productGroupReturnProb']]
        expected_processed = pd.DataFrame({
            'productGroupReturnProb': [0., 0., 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6]})
        self.assertTrue(self.content_equal(actual_processed, expected_processed))

    def test_size_return_probability(self):
        processed_data = feature(self.data)
        actual_processed = processed_data[['sizeReturnProb']]
        expected_processed = pd.DataFrame({
            'sizeReturnProb': [0., 0., 0.5, 0.5, 0, 0, 0, 0]})
        self.assertTrue(self.content_equal(actual_processed, expected_processed))

    def test_is_german_holiday(self):
        processed_data = feature(self.data)
        actual_processed = processed_data[['orderIsOnGermanHoliday']]
        expected_processed = pd.DataFrame({'orderIsOnGermanHoliday': [1, 0, 0, 1, 1, 1, 1, 1]})
        self.assertTrue(self.content_equal(actual_processed, expected_processed))

    def test_binned_color_return_probability(self):
        processed = add_features(self.data)
        self.assertListEqual(['[0, 1992)', '[1993, 10000)', '[1993, 10000)', '[1992, 1993)',
                              '[0, 1992)', '[0, 1992)', '[0, 1992)', '[1993, 10000)'],
                             list(processed.binnedColorCode))
