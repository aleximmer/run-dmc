import numpy as np
from scipy.sparse import csr_matrix, hstack
import pandas as pd
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, OneHotEncoder, LabelEncoder
from sklearn.decomposition import PCA, SparsePCA


encode_label = ['paymentMethod', 'sizeCode', 't_customer_preferredPayment']
encode_int = ['deviceID', 'productGroup', 'articleID', 'customerID', 'orderID',
              'voucherID', 'orderYear', 'orderMonth', 'orderDay', 'orderWeekDay',
              'orderWeek', 'orderSeason', 'orderQuarter', 'customerAvgUnisize', 'binnedColorCode']


def encode_features(df: pd.DataFrame, ft: str) -> csr_matrix:
    """Encode categorical features"""
    if ft not in set(encode_label + encode_int):
        return csr_matrix(df.as_matrix(columns=[ft]))

    label_enc = LabelEncoder()
    one_hot_enc = OneHotEncoder(sparse=True)

    if ft in encode_label:
        V = df[ft].as_matrix().T
        V_lab = label_enc.fit_transform(V).reshape(-1, 1)
        V_enc = one_hot_enc.fit_transform(V_lab)
        return V_enc

    if ft in encode_int:
        V = df[ft].as_matrix().reshape(-1, 1)
        V_enc = one_hot_enc.fit_transform(V)
        return V_enc


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
        X_enc = encode_features(df, ft)
        X = X_enc if X is None else hstack([X, X_enc], format='csr', dtype=np.float32)
    return X.astype(np.float32)


def transform_target_vector(df: pd.DataFrame, binary=False) -> np.array:
    """Only used on data with known labels otherwise it will fail"""
    binarize = lambda x: 1 if x > 0 else 0
    detect = lambda x: x if np.isnan(x) else binarize(x)
    if binary:
        df.returnQuantity = df.returnQuantity.apply(detect)
    return np.squeeze(df.as_matrix(columns=['returnQuantity'])).astype(np.float32)


def transform_preserving_header(df: pd.DataFrame, ignore_features=None, scaler=None,
                                binary_target=False, dim_reduction=False, dims=None) \
        -> (csr_matrix, np.array, list):
    ignore_features = ignore_features if ignore_features is not None \
        else default_ignore_features
    X, fts = transform_feature_matrix_ph(df, ignore_features)
    X = csr_matrix(X)
    if scaler is not None:
        X = scaler(X)
    if dim_reduction:
        assert dims

    Y = transform_target_vector(df, binary_target)
    return X, Y, fts


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


def transform(df: pd.DataFrame, ignore_features=None, scaler=None, binary_target=False,
              dim_reduction=False, dimensions=None) \
        -> (csr_matrix, np.array):
    ignore_features = ignore_features if ignore_features is not None \
        else default_ignore_features
    X = csr_matrix(transform_feature_matrix(df, ignore_features))
    if scaler is not None:
        X = scaler(X)
    if dim_reduction:
        assert dimensions > 0
        # also try pca = SparsePCA()
        pca = PCA(n_components=dimensions, whiten=True)
        X = pca.fit_transform(X.toarray())
    Y = transform_target_vector(df, binary_target)
    return X, Y
