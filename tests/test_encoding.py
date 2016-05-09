import unittest
import pandas as pd
import numpy as np

from dmc.preprocessing import cleanse
from dmc.transformation import encode_features


class EncodingTest(unittest.TestCase):
    def setUp(self):
        raw_data = pd.read_csv('tests/test_data_old.txt', delimiter=';')
        raw_data = raw_data.head(50)
        clean_data = cleanse(raw_data)
        self.data = clean_data

    def test_product_group_encoding(self):
        X = encode_features(self.data, 'productGroup')
        X = X.toarray()
        self.assertTrue((X.T[0] == np.array([1., 1., 0., 0., 0., 0., 0., 0.])).all())
        self.assertTrue((X.T[1] == np.array([0., 0., 1., 1., 1., 1., 1., 1.])).all())

    def test_size_code_encoding(self):
        X = encode_features(self.data, 'sizeCode')
        X = X.toarray()
        exp = np.array([
            [0., 0., 1., 1., 0., 0., 0., 0.],
            [0., 0., 0., 0., 1., 1., 1., 0.],
            [1., 1., 0., 0., 0., 0., 0., 1.]
        ])
        self.assertTrue((exp.T == X).all())
