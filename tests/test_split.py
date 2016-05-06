import unittest
import numpy as np
import pandas as pd

from dmc.preprocessing import cleanse
from dmc.ensemble import split


class SplitTest(unittest.TestCase):
    def setUp(self):
        raw_data = pd.read_csv('tests/test_data.txt', delimiter=';')
        raw_data = raw_data.head(50)
        self.clean_data = cleanse(raw_data)
        self.train = self.clean_data[:5]
        self.test = self.clean_data[5:]

    def test_standard_split(self):
        res = split(self.train, self.test)
        for s in res:
            self.assertTrue(len(s[0].columns) <= len(self.train.columns))
            self.assertTrue(len(s[1].columns) == len(s[0].columns))
            x = len([e for e in s[2].split('-') if 'unknown' in e])
            self.assertTrue(len(s[1].columns) == len(self.train.columns) - x)

    def test_split_with_nans(self):
        data = self.clean_data.copy()
        unknown = {1001974, 1001976}
        data['returnProb'] = data['articleID'].apply(lambda l: 1 if l not in unknown else np.nan)
        train, test = data[:5], data[5:]
        res = split(train, test)
        for s in res:
            self.assertTrue(len(s[0].columns) <= len(data.columns))
            self.assertTrue(len(s[1].columns) == len(s[0].columns))
            # double because unknown col has also nan col in this test
            x = 2 * len([e for e in s[2].split('-') if 'unknown' in e])
            self.assertTrue(len(s[1].columns) == len(data.columns) - x)
