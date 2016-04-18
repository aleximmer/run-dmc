import unittest
import pandas as pd
from pandas.util.testing import assert_frame_equal
import dmc


class PreprocessingTest(unittest.TestCase):
    def setUp(self):
        raw_data = pd.read_csv('tests/test_data.txt', delimiter=';')
        raw_data = raw_data.head(50)
        clean_data = dmc.cleansing.cleanse(raw_data)
        self.data = clean_data

    def content_equal(self, a, b):
        try:
            assert_frame_equal(
                a.sort_index(axis=1), b.sort_index(axis=1), check_names=True)
            return True
        except (AssertionError, ValueError, TypeError):
            return False

    def test_preprocess(self):
        processed_data = dmc.preprocessing.preprocess(self.data)
        self.assertIn('customerReturnProb', processed_data.columns)
        self.assertIn('totalOrderShare', processed_data.columns)
        self.assertIn('sizeReturnProb', processed_data.columns)

    def test_size_return_probability(self):
        processed_data = dmc.preprocessing.preprocess(self.data)
        actual_processed = processed_data[
            ['0sizeCode', '1sizeCode', '2sizeCode', 'sizeReturnProb']]
        expected_processed = pd.DataFrame({'0sizeCode': [0., 0., 1., 1., 0., 0., 0., 0.],
                                           '1sizeCode': [0., 0., 0., 0., 1., 1., 1., 0.],
                                           '2sizeCode': [1., 1., 0., 0., 0., 0., 0., 1.],
                                           'sizeReturnProb': [0., 0., 0.5, 0.5, 0, 0, 0, 0]})
        self.assertTrue(self.content_equal(actual_processed, expected_processed))
