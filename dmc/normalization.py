from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np


def scale_features(X: np.array) -> np.array:
    """Scale features to mean 0 and unit variance (1)"""
    scaler = StandardScaler().fit(X)
    return scaler.transform(X)


def normalize_features(X: np.array) -> np.array:
    """Normalize features by scaling to [0,1]"""
    scaler = MinMaxScaler().fit(X)
    return scaler.transform(X)
