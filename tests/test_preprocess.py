import unittest
import pandas as pd
import numpy as np
from pandas.util.testing import assert_frame_equal

from dmc.preprocessing import cleanse, apply_features, clean_ids, split_train_test


class PreprocessingTest(unittest.TestCase):
    def setUp(self):
        self.raw_data = pd.read_csv('tests/test_data.txt', delimiter=';')
        self.raw_data = self.raw_data.head(50)
        train_ids = ['a1000001', 'a1000002', 'a1000003']
        test_ids = ['a1000007', 'a1000008']

        clean_data = cleanse(self.raw_data)
        self.data = {'data': clean_data, 'train_ids': train_ids, 'test_ids': test_ids}

    @staticmethod
    def content_equal(a, b):
        try:
            assert_frame_equal(a.sort_index(axis=1), b.sort_index(axis=1), check_names=True)
            return True
        except (AssertionError, ValueError, TypeError):
            return False

    def test_cleanse(self):
        df = cleanse(self.raw_data)
        # Column values
        self.assertTrue((df.quantity != 0).all())
        self.assertTrue((df.quantity >= df.returnQuantity).all())
        # Column types
        self.assertTrue(df.orderDate.dtype == np.dtype('<M8[ns]'))
        self.assertTrue(df.orderID.dtype == np.int)
        self.assertTrue(df.articleID.dtype == np.int)
        self.assertTrue(df.customerID.dtype == np.int)
        self.assertTrue(df.voucherID.dtype == np.float)

    def test_preprocess(self):
        processed_data = apply_features(self.data)['train']
        self.assertIn('customerReturnProb', processed_data.columns)
        self.assertIn('totalOrderShare', processed_data.columns)
        self.assertIn('productGroupReturnProb', processed_data.columns)
        self.assertIn('colorReturnProb', processed_data.columns)
        self.assertIn('sizeReturnProb', processed_data.columns)

    def test_split(self):
        train, test = split_train_test(**self.data)
        train_ids = set(train.orderID.tolist())
        test_ids = set(test.orderID.tolist())
        for test_id in clean_ids(self.data['test_ids']):
            self.assertIn(test_id, test_ids)
            self.assertNotIn(test_ids, train_ids)
        for train_id in clean_ids(self.data['train_ids']):
            self.assertIn(train_id, train_ids)
            self.assertNotIn(train_id, test_ids)
