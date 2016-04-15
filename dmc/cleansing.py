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
    """Drop data which doesn't comply with constraints"""
    df = df[df.quantity > 0]
    df = df[df.quantity >= df.returnQuantity]
    return df


def parse_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Convert to float and integer types"""
    df.orderDate = pd.to_datetime(df['orderDate'])
    df.orderID = df.orderID.apply(lambda x: x.replace('a', '')).astype(np.int)
    df.articleID = df.articleID.apply(lambda x: x.replace('i', '')).astype(np.int)
    df.customerID = df.customerID.apply(lambda x: x.replace('c', '')).astype(np.int)
    df.voucherID = df.voucherID.apply(lambda x: str(x).replace('v', '')).astype(np.float)
    return df


def cleanse(df: pd.DataFrame) -> pd.DataFrame:
    df = parse_strings(df)
    df = enforce_constraints(df)
    assert_constraints(df)
    return df
