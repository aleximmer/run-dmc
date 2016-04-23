import pandas as pd
import numpy as np
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
