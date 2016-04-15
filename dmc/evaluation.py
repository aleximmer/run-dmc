import numpy as np


def dmc_cost(predicted: np.array, ground_truth: np.array) -> int:
    """Cost function defined for the DMC"""
    diff = np.abs(predicted - ground_truth)
    return np.sum(diff)
