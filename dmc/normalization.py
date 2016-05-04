from sklearn.preprocessing import StandardScaler, MaxAbsScaler
import numpy as np
from scipy.sparse import csr_matrix


def scale_features(X: np.array) -> np.array:
    """Scale features to mean 0 and unit variance (1)"""
    scaler = StandardScaler(with_mean=False).fit(X)
    return scaler.transform(X)


def scale_raw_features(X: np.array) -> np.array:
    """Scale features to mean 0 and unit variance if
    column was not OneHot encoded"""
    for col in range(X.shape[1]):
        dense_col = X[:, col].todense()
        if (dense_col > 1.).any() or (dense_col < 0.).any():
            scaler = StandardScaler().fit(dense_col)
            X[:, col] = csr_matrix(scaler.transform(dense_col))
    return X


def normalize_features(X: np.array) -> np.array:
    """Normalize features by scaling to [0,1]"""
    scaler = MaxAbsScaler().fit(X)
    return scaler.transform(X)


def normalize_raw_features(X: np.array) -> np.array:
    """Normalize features if column was not OneHot encoded"""
    for col in range(X.shape[1]):
        dense_col = X[:, col].todense()
        if (dense_col > 1.).any() or (dense_col < 0.).any():
            scaler = MaxAbsScaler().fit(dense_col)
            X[:, col] = csr_matrix(scaler.transform(dense_col))
    return X
