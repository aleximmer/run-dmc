import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


encode_label = ['paymentMethod', 'sizeCode', 't_customer_preferredPayment']
encode_int = ['deviceID', 'productGroup', 'articleID', 'orderYear', 'orderMonth',
              'voucherID', 'customerID', 'orderDay', 'orderWeekDay', 'orderWeek',
              'orderSeason', 'orderQuarter']


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


def encode_features_np(df: pd.DataFrame, ft: str) -> np.array:
    """Encode categorical features"""
    if ft not in set(encode_label + encode_int):
        return df.as_matrix(columns=[ft])

    label_enc = LabelEncoder()
    one_hot_enc = OneHotEncoder(sparse=False)

    if ft in encode_label:
        V = df[ft].as_matrix().T
        V_lab = label_enc.fit_transform(V).reshape(-1, 1)
        V_enc = one_hot_enc.fit_transform(V_lab)
        return V_enc

    if ft in encode_int:
        V = df[ft].as_matrix().reshape(-1, 1)
        V_enc = one_hot_enc.fit_transform(V)
        return V_enc
