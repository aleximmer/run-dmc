import pandas as pd
import numpy as np
import pandas as pd

from dmc.transformation import transform_preserving_headers, transform_target_vector
from dmc.classifiers import Forest, DecisionTree


def dmc_cost(predicted: np.array, ground_truth: np.array) -> int:
    """Cost function defined for the DMC"""
    diff = np.abs(predicted - ground_truth)
    return np.sum(diff)


def dmc_cost_relative(predicted: np.array, ground_truth: np.array) -> float:
    cost = dmc_cost(predicted, ground_truth)
    return cost / len(predicted)


def precision(predicted: np.array, ground_truth: np.array) -> int:
    diff = predicted - ground_truth
    return 1 - np.count_nonzero(diff) / len(predicted)


def gini_ratio(arr: pd.Series) -> float:
    """Return impurity of array
    """
    _, counts = np.unique(arr, return_counts=True)
    squared_ratio = np.vectorize(lambda count: np.square(np.divide(count, len(arr))))
    return 1.0 - np.sum(squared_ratio(counts))


def features(df: pd.DataFrame) -> pd.DataFrame:
    """Returns a MultiIndex'd (col, val) DataFrame
    with gini impurity, average return quantity, and return probability
    """
    columns = ['gini', 'avgRet', 'retProb']
    purities = pd.DataFrame(columns=['gini', 'avgRet', 'retProb'], index=pd.MultiIndex(labels=[[], []], levels=[[], []],
                                                                 names=['column', 'value']))
    feature_cols = df.drop('returnQuantity', axis=1).columns

    def column_info(labels: pd.Series) -> pd.Series:
        gini = gini_ratio(labels)
        avg = labels.mean()
        ret_prob = labels.astype(bool).sum() / len(labels)
        return gini, avg, ret_prob

    for col in feature_cols:
        value_infos = df.groupby(col)['returnQuantity'].apply(column_info)
        for i, row in value_infos.iteritems():
            purities.loc[(col, i), :] = row

    return purities


def column_purities(df: pd.DataFrame) -> pd.Series:
    feature_cols = df.drop('returnQuantity', axis=1).columns
    purities = pd.Series(None, index=feature_cols)

    def weighted_gini(group: pd.DataFrame) -> float:
        return len(group) / len(df) * gini_ratio(group['returnQuantity'])

    for col in feature_cols:
        summed_gini = df.groupby(col).apply(weighted_gini).sum()
        purities[col] = summed_gini

    return purities


def eval_features_by_ensemble(df: pd.DataFrame) -> pd.DataFrame:
    """Returns a dataframe giving each feature an importance factor"""
    X, fts = transform_preserving_headers(df)
    Y = transform_target_vector(df)
    forest = Forest(X, Y).fit()
    tree = DecisionTree(X, Y).fit()
    ft_eval = pd.DataFrame({
        'feature': fts,
        'forest': forest.clf.feature_importances_,
        'tree': tree.clf.feature_importances_,
    })
    return ft_eval.groupby('feature').sum()
