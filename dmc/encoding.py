import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


def encode_features(df: pd.DataFrame, ft: str) -> np.array:
    """Encode categorical features"""
    encode_label = ['paymentMethod', 'sizeCode']
    encode_int = ['deviceID', 'productGroup', 'articleID',
                  'voucherID', 'customerID']

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
