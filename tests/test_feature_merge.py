import unittest
import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal
import dmc


class FeatureMergeTest(unittest.TestCase):
    def setUp(self):
        self.df = pd.read_csv('tests/test_data.txt', delimiter=';')
        self.df = dmc.cleansing.cleanse(self.df)

    def content_equal(self, a, b):
        try:
            assert_frame_equal(
                a.sort_index(axis=1), b.sort_index(axis=1), check_names=True)
            return True
        except (AssertionError, ValueError, TypeError):
            return False

    def test_single_merge(self):
        feature_df = self.df.copy()
        feature_df['f1'] = pd.Series(np.arange(len(feature_df.index)), index=feature_df.index)
        feature_df['f2'] = pd.Series(np.arange(len(feature_df.index)), index=feature_df.index)

        merged_df = dmc.preprocessing.merge_features(self.df, [feature_df])
        self.assertEqual(len(merged_df.columns), len(feature_df.columns))
        self.assertIn('f1', merged_df)
        self.assertIn('f2', merged_df)

    def test_empty_merge(self):
        merged_df = dmc.preprocessing.merge_features(self.df, [])
        assert_frame_equal(self.df, merged_df)

    def test_multiple_merge(self):
        feature_dfs = []
        for i in range(4):
            key = 'f' + str(i)
            feature_df = self.df.copy()
            feature_df[key] = pd.Series(np.random.uniform(1, 10, size=len(feature_df.index)),
                                        index=feature_df.index)
            feature_dfs.append(feature_df)
        # Get unique flatted list of all column names
        merged_columns = [f.columns.values.tolist() for f in feature_dfs]
        merged_columns = list(set([item for sublist in merged_columns for item in sublist]))

        merged_df = dmc.preprocessing.merge_features(self.df, feature_dfs)
        self.assertEqual(len(merged_df.columns), len(merged_columns))

    def test_conflicting_keys(self):
        feature_dfs = [self.df.copy()]
        for i in range(4):
            key = 'f' + str(i)
            feature_df = feature_dfs[-1].copy()
            feature_df[key] = pd.Series(np.random.uniform(1, 10, size=len(feature_df.index)),
                                        index=feature_df.index)
            feature_dfs.append(feature_df)
        # Get unique flatted list of all column names
        merged_columns = [f.columns.values.tolist() for f in feature_dfs]
        merged_columns = list(set([item for sublist in merged_columns for item in sublist]))

        merged_df = dmc.preprocessing.merge_features(self.df, feature_dfs)
        self.assertEqual(len(merged_df.columns), len(merged_columns))
