import pandas as pd
import numpy as np

from dmc.features import add_dependent_features, add_independent_features


def apply_features(data: dict) -> dict:
    """Add features and drop unused ones.
    """
    train = add_dependent_features(train)
    return {'train': train, 'test': test}


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
    df.voucherID = np.nan_to_num(df.voucherID)
    return df


def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns which are either duplicate or will be added within our framework

    - date features since we create all possible of them using pandas
    - binary target is an option for benchmarking later
    - last six are dropped because of amateurish feature engineering
    """
    blacklist = {'id', 't_orderDate', 't_orderDateWOYear', 't_season', 't_dayOfWeek',
                 't_dayOfMonth', 't_isWeekend', 't_singleItemPrice_per_rrp', 't_atLeastOneReturned',
                 't_voucher_usedOnlyOnce_A', 't_voucher_stdDevDiscount_A', 't_voucher_OrderCount_A',
                 't_voucher_hasAbsoluteDiscountValue_A', 't_voucher_firstUsedDate_A',
                 't_voucher_lastUsedDate_A', 't_customer_avgUnisize'}
    return df.drop(blacklist & set(df.columns), 1)


def remove_features(df: pd.DataFrame) -> pd.DataFrame:
    blacklist = ['t_customer_avgUnisize']  # t_voucher_firstUsedDate_A, t_voucher_lastUsedDate_A
    df = df.drop(blacklist, 1)
    return df


def cleanse(df: pd.DataFrame) -> pd.DataFrame:
    df = drop_columns(df)
    df = parse_strings(df)
    df = remove_features(df)
    df = enforce_constraints(df)
    return df


def clean_ids(id_list: list) -> list:
    return {int(x.replace('a', '')) for x in id_list}


def split_train_test(data: pd.DataFrame, train_ids: list, test_ids: list) \
        -> (pd.DataFrame, pd.DataFrame):
    train_ids = clean_ids(train_ids)
    test_ids = clean_ids(test_ids)
    train = data[data.orderID.isin(train_ids)].copy()
    test = data[data.orderID.isin(test_ids)].copy()
    return train, test
