import numpy as np
import pandas as pd

from dmc.encoding import encode_features


target_feature = 'returnQuantity'
default_ignore_features = ['returnQuantity', 'orderID',
                           'orderDate', 'customerID']


def transform_feature_matrix(df: pd.DataFrame, ignore_features: list) -> np.array:
    """Used to transform the full data space in order to get all categories"""
    assert target_feature in ignore_features
    X = np.empty((len(df), 0))
    for ft in [ft for ft in df.columns if ft not in ignore_features]:
        X = np.append(X, encode_features(df, ft), axis=1)
    return X.astype(np.float32)


def transform_target_vector(df: pd.DataFrame) -> np.array:
    """Only used on data with known labels otherwise it will fail"""
    return np.squeeze(df.as_matrix(columns=['returnQuantity'])).astype(np.int32)


def transform(df: pd.DataFrame, ignore_features=None, scaler=None) -> (np.array, np.array):
    ignore_features = ignore_features if ignore_features is not None \
        else default_ignore_features
    X = transform_feature_matrix(df, ignore_features)
    if scaler is not None:
        X = scaler(X)
    Y = transform_target_vector(df)
    return X, Y
