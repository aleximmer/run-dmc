import numpy as np
from scipy.sparse import csr_matrix, hstack
import pandas as pd

from dmc.encoding import encode_features


target_feature = 'returnQuantity'
default_ignore_features = ['returnQuantity', 'orderID',
                           'orderDate', 'customerID']


def transform_feature_matrix_ph(df: pd.DataFrame, ignore_features=None) -> \
        (np.array, np.array):
    """Transform specific data space and return list of tuples indicating where features lie"""
    ignore_features = default_ignore_features if ignore_features is None else ignore_features
    assert target_feature in ignore_features
    X, ft_list = None, []
    for ft in [ft for ft in df.columns if ft not in ignore_features]:
        X_enc = encode_features(df, ft)
        X = X_enc if X is None else hstack([X, X_enc])
        ft_list.extend([ft] * X_enc.shape[1])
    return X.astype(np.float32), np.array(ft_list)


def transform_feature_matrix(df: pd.DataFrame, ignore_features: list) -> csr_matrix:
    """Used to transform the full data space in order to get all categories"""
    assert target_feature in ignore_features
    X = None
    for ft in [ft for ft in df.columns if ft not in ignore_features]:
        X = encode_features(df, ft) if X is None else hstack([X, encode_features(df, ft)])
    return X.astype(np.float32)


def transform_target_vector(df: pd.DataFrame, binary=False) -> np.array:
    """Only used on data with known labels otherwise it will fail"""
    if binary:
        df.returnQuantity = df.returnQuantity.apply(lambda x: 1 if x > 0 else 0)
    return np.squeeze(df.as_matrix(columns=['returnQuantity'])).astype(np.int32)


def transform_preserving_header(df: pd.DataFrame, ignore_features=None, scaler=None,
                                binary_target=False) -> (csr_matrix, np.array, list):
    ignore_features = ignore_features if ignore_features is not None \
        else default_ignore_features
    X, fts = transform_feature_matrix_ph(df, ignore_features)
    if scaler is not None:
        X = scaler(X)
    Y = transform_target_vector(df, binary_target)
    return X, Y, fts


def transform(df: pd.DataFrame, ignore_features=None, scaler=None, binary_target=False) \
        -> (csr_matrix, np.array):
    ignore_features = ignore_features if ignore_features is not None \
        else default_ignore_features
    X = transform_feature_matrix(df, ignore_features)
    if scaler is not None:
        X = scaler(X)
    Y = transform_target_vector(df, binary_target)
    return X, Y
