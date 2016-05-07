import pandas as pd
import numpy as np

from dmc.features import SelectedFeatures


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


def add_recognition_vector(train: pd.DataFrame, test: pd.DataFrame, columns: list) \
        -> (pd.DataFrame, list):
    """Create a mask of test values seen in training data.
    """
    known_mask = test[columns].copy().apply(lambda column: column.isin(train[column.name]))
    known_mask.columns = ('known_' + c for c in columns)
    return known_mask


def split_known_unknown(train: pd.DataFrame, test: pd.DataFrame) -> list():
    """For each permutation of known and unknown categories return the cropped train DataFrame and
    the test subset for evaluation.
    """
    potentially_unknown = ['articleID', 'customerID', 'voucherID', 'productGroup']
    known_mask = add_recognition_vector(train, test, potentially_unknown)
    test = pd.concat([test, known_mask], axis=1)
    splitters = list(known_mask.columns)
    result = []
    for mask, group in test.groupby(splitters):
        specifier = '-'.join('known_' + col if known else 'unknown_' + col
                             for known, col in zip(mask, potentially_unknown))
        unknown_columns = [col for known, col in zip(mask, potentially_unknown) if not known]
        nan_columns = [col for col in group.columns
                       if group[col].dtype == float and np.isnan(group[col]).any()]
        train_crop = train.copy().drop(unknown_columns + nan_columns, axis=1)
        test_group = group.copy().drop(unknown_columns + nan_columns + splitters, axis=1)
        result.append((train_crop, test_group, specifier))
    return result
