import numpy as np


def dmc_cost(predicted: np.array, ground_truth: np.array) -> int:
    """Cost function defined for the DMC"""
    diff = np.abs(predicted - ground_truth)
    return np.sum(diff)


def dmc_cost_relative(predicted: np.array, ground_truth: np.array) -> float:
    cost = dmc_cost(predicted, ground_truth)
    return cost/len(predicted)


def precision(predicted: np.array, ground_truth: np.array) -> int:
    size = len(predicted)
    diff = predicted - ground_truth
    return 1 - np.count_nonzero(diff)/size
