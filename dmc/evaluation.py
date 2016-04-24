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
    _, counts = np.unique(arr, return_counts=True)
    squared_ratio = np.vectorize(lambda count: np.square(np.divide(count, len(arr))))
    return 1.0 - np.sum(squared_ratio(counts))


def feature_purities(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Returns a dictionary of dictionaries where
        col_purities['col']['val']
    is the gini coefficient of the label column taken only
    the elements in col equal to val.
    """
    col_purities = {}
    for col in df.columns:
        if not col == label_col:
            col_purities[col] = (df.groupby(col)[label_col]
                                 .apply(gini_ratio).to_dict())
    return col_purities
