import numpy as np
from scipy.sparse import csr_matrix, hstack
import pandas as pd

from dmc.encoding import encode_features, encode_features_np


target_feature = 'returnQuantity'
default_ignore_features = ['returnQuantity', 'orderID',
                           'orderDate', 'customerID']


def transform_preserving_headers(df: pd.DataFrame) -> \
        (np.array, np.array):
    """Transform specific data space and return list of tuples indicating where features lie"""
    X, ft_list = np.empty((len(df), 0)), []
    for ft in [ft for ft in df.columns if ft not in default_ignore_features]:
        X_enc = encode_features_np(df, ft)
        X = np.append(X, encode_features_np(df, ft), axis=1)
        ft_list.extend([ft] * len(X_enc.T))
    return X.astype(np.float32), np.array(ft_list)


def transform_feature_matrix(df: pd.DataFrame, ignore_features: list) -> csr_matrix:
    """Used to transform the full data space in order to get all categories"""
    assert target_feature in ignore_features
    X = None
    for ft in [ft for ft in df.columns if ft not in ignore_features]:
        X = encode_features(df, ft) if X is None else hstack([X, encode_features_np(df, ft)])
    return X.astype(np.float32)


def transform_target_vector(df: pd.DataFrame, binary=False) -> np.array:
    """Only used on data with known labels otherwise it will fail"""
    if binary:
        df.returnQuantity = df.returnQuantity.apply(lambda x: 1 if x > 0 else 0)
    return np.squeeze(df.as_matrix(columns=['returnQuantity'])).astype(np.int32)


def transform(df: pd.DataFrame, ignore_features=None, scaler=None, binary_target=False) \
        -> (csr_matrix, np.array):
    ignore_features = ignore_features if ignore_features is not None \
        else default_ignore_features
    X = transform_feature_matrix(df, ignore_features)
    if scaler is not None:
        X = scaler(X)
    Y = transform_target_vector(df, binary_target)
    return X, Y
