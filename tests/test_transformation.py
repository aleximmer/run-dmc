import unittest
import pandas as pd
import numpy as np

from dmc.preprocessing import cleanse
from dmc.features import add_independent_features
from dmc.transformation import transform


class TransformationTest(unittest.TestCase):
    def setUp(self):
        raw_data = pd.read_csv('tests/test_data_old.txt', delimiter=';')
        raw_data = raw_data.head(50)
        clean_data = cleanse(raw_data)
        featured_data = add_independent_features(clean_data)
        self.X, self.Y = transform(featured_data)
        self.X = self.X.toarray()

    def test_product_group_encoding(self):
        self.assertTrue((self.X.T[11] == np.array([1., 1., 0., 0., 0., 0., 0., 0.])).all())
        self.assertTrue((self.X.T[12] == np.array([0., 0., 1., 1., 1., 1., 1., 1.])).all())

    def test_size_code_encoding(self):
        exp = np.array([
            [0., 0., 1., 1., 0., 0., 0., 0.],
            [0., 0., 0., 0., 1., 1., 1., 0.],
            [1., 1., 0., 0., 0., 0., 0., 1.]
        ])
        self.assertTrue((exp == self.X.T[8:11]).all())

    def test_target_vector(self):
        exp = np.array([0, 0, 0, 1, 0, 0, 0, 0])
        self.assertTrue((exp == self.Y).all())
