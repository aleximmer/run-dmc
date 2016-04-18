import unittest
import pandas as pd

import dmc


class PreprocessingTest(unittest.TestCase):
    def setUp(self):
        raw_data = pd.read_csv('test/test_data.txt', delimiter=';')
        raw_data = raw_data.head(50)
        clean_data = dmc.cleansing.cleanse(raw_data)
        self.data = clean_data

    def test_preprocess(self):
        processed_data = dmc.preprocessing.preprocess(self.data)
        self.assertIn('customerReturnProb', processed_data.columns)
        self.assertIn('totalOrderShare', processed_data.columns)
