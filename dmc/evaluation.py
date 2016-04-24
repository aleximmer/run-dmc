import numpy as np
import pandas as pd


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


def gini_ratio(arr: list) -> float:
    """Return impurity of array
    """
    _, counts = np.unique(arr, return_counts=True)
    squared_ratio = np.vectorize(lambda count: np.square(np.divide(count, len(arr))))
    return 1.0 - np.sum(squared_ratio(counts))


def feature_purities(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Returns a dictionary of dictionaries containing the impurity
    of each column of each unique element in it
    """
    purities = {}
    feature_cols = df.drop(label_col, axis=1).columns
    for col in feature_cols:
        purities[col] = df.groupby(col)[label_col].apply(gini_ratio).to_dict()
    return purities


def column_purities(df: pd.DataFrame, label_col: str) -> pd.Series:
    feature_cols = df.drop(label_col, axis=1).columns
    purities = pd.Series(None, index=feature_cols)

    def weighted_gini(group: pd.DataFrame) -> float:
        return len(group) / len(df) * gini_ratio(group[label_col])

    for col in feature_cols:
        summed_gini = df.groupby(col).apply(weighted_gini).sum()
        purities[col] = summed_gini
    return purities
