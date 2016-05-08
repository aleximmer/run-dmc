import unittest
import pandas as pd

from dmc.preprocessing import cleanse


class CleansingTest(unittest.TestCase):
    def setUp(self):
        raw_data = pd.read_csv('tests/test_data_old.txt', delimiter=';')
        raw_data = raw_data.head(50)
        self.data = raw_data

    def test_cleanse(self):
        cleanse(self.data)
