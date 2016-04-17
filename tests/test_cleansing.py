import unittest
import pandas as pd

import dmc


class CleansingTest(unittest.TestCase):
    def setUp(self):
        raw_data = pd.read_csv('../data/orders_train.txt', delimiter=';')
        raw_data = raw_data.head(50)
        self.data = raw_data

    def test_cleanse(self):
        dmc.cleansing.cleanse(self.data)
