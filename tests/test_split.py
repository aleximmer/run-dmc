import unittest
import numpy as np
import pandas as pd

from dmc.preprocessing import cleanse
from dmc.ensemble import split


class SplitTest(unittest.TestCase):
    def setUp(self):
        raw_data = pd.read_csv('tests/test_data_old.txt', delimiter=';')
        raw_data = raw_data.head(50)
        self.clean_data = cleanse(raw_data)
        self.train = self.clean_data[:5]
        self.test = self.clean_data[5:]
        self.cats = ['articleID', 'customerID', 'voucherID', 'productGroup']

    def test_standard_split(self):
        res = split(self.train, self.test, self.cats)
        new_res = [(res[k]['train'], res[k]['test'], k) for k in res]
        for s in new_res:
            self.assertTrue(len(s[0].columns) <= len(self.train.columns))
            self.assertTrue(len(s[1].columns) == len(s[0].columns))
            x = s[2].count('u')
            self.assertTrue(len(s[1].columns) == len(self.train.columns) - x)

    def test_split_with_nans(self):
        data = self.clean_data.copy()
        unknown = {1001974, 1001976}
        data['returnProb'] = data['articleID'].apply(lambda l: 1 if l not in unknown else np.nan)
        train, test = data[:5], data[5:]
        res = split(train, test, self.cats)
        new_res = [(res[k]['train'], res[k]['test'], k) for k in res]
        for s in new_res:
            self.assertTrue(len(s[0].columns) <= len(data.columns))
            self.assertTrue(len(s[1].columns) == len(s[0].columns))
            # double because unknown col has also nan col in this test
            x = 2 *  s[2].count('u') # len([e for e in s[2].split('-') if 'unknown' in e])
            self.assertTrue(len(s[1].columns) == len(data.columns) - x)
