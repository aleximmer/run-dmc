from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np


def scale_features(X: np.array) -> np.array:
    print('scale')
    scaler = StandardScaler().fit(X)
    return scaler.transform(X)


def normalize_features(X: np.array) -> np.array:
    print('normalize')
    scaler = MinMaxScaler().fit(X)
    return scaler.transform(X)
