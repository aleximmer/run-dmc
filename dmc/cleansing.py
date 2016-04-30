import pandas as pd
import numpy as np


def assert_constraints(df: pd.DataFrame) -> pd.DataFrame:
    """
    1. Define constraints here
    2. Enforce later
    """
    # Column values
    assert (df.quantity != 0).all()
    assert (df.quantity >= df.returnQuantity).all()

    # Column types
    assert df.orderDate.dtype == np.dtype('<M8[ns]')
    assert df.orderID.dtype == np.int
    assert df.articleID.dtype == np.int
    assert df.customerID.dtype == np.int
    assert df.voucherID.dtype == np.float


def enforce_constraints(df: pd.DataFrame) -> pd.DataFrame:
    """Drop data which doesn't comply with constraints
    Dropped rows would be """
    df = df[df.quantity > 0]
    df = df[df.quantity >= df.returnQuantity]
    # nans in these rows definitely have returnQuantity == 0
    df = df.dropna(subset=['voucherID', 'rrp', 'productGroup'])
    return df


def parse_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Convert to float and integer types"""
    df.orderDate = pd.to_datetime(df.orderDate)
    df.orderID = df.orderID.apply(lambda x: x.replace('a', '')).astype(np.int)
    df.articleID = df.articleID.apply(lambda x: x.replace('i', '')).astype(np.int)
    df.customerID = df.customerID.apply(lambda x: x.replace('c', '')).astype(np.int)
    df.voucherID = df.voucherID.apply(lambda x: str(x).replace('v', '')).astype(np.float)
    return df


def handle_blacklisted_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop or encode specific features of group B"""
    blacklist = ['id', 't_orderDate', 't_orderDateWOYear', 't_season', 't_atLeastOneReturned']
    for key in blacklist:
        if key in df.columns:
            df = df.drop(key, 1)
    if ('t_voucher_lastUsedDate_A' not in df.columns and
       't_voucher_firstUsedDate_A' not in df.columns):
        return df
    df.t_voucher_firstUsedDate_A = pd.to_datetime(df.t_voucher_firstUsedDate_A)
    df.t_voucher_lastUsedDate_A = pd.to_datetime(df.t_voucher_lastUsedDate_A)
    df.t_voucher_firstUsedDate_A = df.t_voucher_firstUsedDate_A.apply(
        lambda x: x.dayofyear if x.year == 2014 else x.dayofyear + 365)
    df.t_voucher_lastUsedDate_A = df.t_voucher_lastUsedDate_A.apply(
        lambda x: x.dayofyear if x.year == 2014 else x.dayofyear + 365)

    if 't_singleItemPrice_per_rrp' in df.columns:
        df.t_singleItemPrice_per_rrp = np.nan_to_num(df.t_singleItemPrice_per_rrp)

    return df


def cleanse(df: pd.DataFrame) -> pd.DataFrame:
    df = handle_blacklisted_features(df)
    df = parse_strings(df)
    df = enforce_constraints(df)
    assert_constraints(df)
    return df
