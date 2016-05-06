import unittest
import pandas as pd

from dmc.preprocessing import cleanse
from dmc.ensemble import split


class SplitTest(unittest.TestCase):
    def setUp(self):
        raw_data = pd.read_csv('tests/test_data.txt', delimiter=';')
        raw_data = raw_data.head(50)
        clean_data = cleanse(raw_data)
        self.train = clean_data[:5]
        self.test = clean_data[5:]

    def test(self):
        res = split(self.train, self.test)
        for s in res:
            self.assertTrue(len(s[0].columns) <= len(self.train.columns))
            self.assertTrue(len(s[1].columns) == len(s[0].columns))
            x = len([e for e in s[2].split('-') if 'unknown' in e])
            print(x, len(self.train.columns), len(s[1].columns))
            self.assertTrue(len(s[1].columns) == len(self.train.columns) - x)
