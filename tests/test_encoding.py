import unittest
import pandas as pd
import numpy as np
import dmc


class PreprocessingTest(unittest.TestCase):
    def setUp(self):
        raw_data = pd.read_csv('tests/test_data.txt', delimiter=';')
        raw_data = raw_data.head(50)
        clean_data = dmc.cleansing.cleanse(raw_data)
        self.data = clean_data

    def test_product_group_encoding(self):
        X = dmc.encoding.encode_features(self.data, 'productGroup')
        self.assertTrue((X.T[0] == np.array([1., 1., 0., 0., 0., 0., 0., 0.])).all())
        self.assertTrue((X.T[1] == np.array([0., 0., 1., 1., 1., 1., 1., 1.])).all())

    def test_size_code_encoding(self):
        X = dmc.encoding.encode_features(self.data, 'sizeCode')
        exp = np.array([
            [0., 0., 1., 1., 0., 0., 0., 0.],
            [0., 0., 0., 0., 1., 1., 1., 0.],
            [1., 1., 0., 0., 0., 0., 0., 1.]
        ])
        self.assertTrue((exp.T == X).all())
