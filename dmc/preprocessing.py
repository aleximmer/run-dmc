import pandas as pd
import numpy as np

from dmc.features import SelectedFeatures


def enforce_constraints(df: pd.DataFrame) -> pd.DataFrame:
    """Drop data which doesn't comply with constraints
    Dropped rows would be """
    df.voucherID.fillna(0, inplace=True)
    df_train = df[~np.isnan(df.returnQuantity)]
    df_class = df[np.isnan(df.returnQuantity)]
    print(len(df_train), len(df_class))
    df_train = df_train[df_train.quantity > 0]
    df_train = df_train[(df_train.quantity >= df_train.returnQuantity)]
    # nans in these rows definitely have returnQuantity == 0
    df_train = df_train.dropna(subset=['voucherID', 'rrp', 'productGroup'])
    df = pd.concat([df_train, df_class], ignore_index=True)
    df = fill_nas(df)
    return df


def fill_nas(df: pd.DataFrame) -> pd.DataFrame:
    if 't_order_daysToNextOrder' in df.columns.tolist():
        df['t_order_daysToNextOrder'].fillna(0, inplace=True)
    if 't_order_daysToPreviousOrder' in df.columns.tolist():
        df['t_order_daysToPreviousOrder'].fillna(0, inplace=True)
    if 't_customer_avgUnisize' in df.columns.tolist():
        df['t_customer_avgUnisize'].fillna(df['t_customer_avgUnisize'].mean(),
                                           inplace=True)
        df['t_customer_avgUnisize'] = df['t_customer_avgUnisize'].astype(np.int)
    if 't_unisize' in df.columns.tolist():
        df['t_unisize'].fillna(df['t_unisize'].mean(), inplace=True)
    if 't_unisizeOffset' in df.columns.tolist():
        df['t_unisizeOffset'].fillna(0, inplace=True)
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
    new_features = set(df.columns.tolist()) - SelectedFeatures.get_all_features()
    if len(new_features):
        print('>>> New features found in df: {}'.format(new_features))
    whitelist = SelectedFeatures.get_whitelist()
    for key in [k for k in df.columns if k not in whitelist]:
        df = df.drop(key, 1)
    return df


def cleanse(df: pd.DataFrame) -> pd.DataFrame:
    df = drop_columns(df)
    df = parse_strings(df)
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
