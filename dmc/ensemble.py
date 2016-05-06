import pandas as pd
import numpy as np


def unknown_categories(column, train: pd.DataFrame, test: pd.DataFrame):
    unknown = set(test[column]) - set(train[column])
    return unknown


def add_recognition_vector(train: pd.DataFrame, test: pd.DataFrame, columns: list) \
        -> (pd.DataFrame, list):
    """For each permutation of known and unknown for specific categories
    return the cropped train DataFrame and the test subset for evaluation"""
    recognition_columns = []
    for col in columns:
        unknown = unknown_categories(col, train, test)
        column_name = 'known' + col
        recognition_columns.append(column_name)
        test[column_name] = test[col].apply(lambda x: x not in unknown)
    return test, recognition_columns


def split(train: pd.DataFrame, test: pd.DataFrame) -> list():
    """For each permutation of known and unknown for specific categories
    return the cropped train DataFrame and the test subset for evaluation"""
    potentially_unknown = ['articleID', 'customerID', 'voucherID', 'productGroup']
    test, splitters = add_recognition_vector(train, test, potentially_unknown)
    result = []
    for vec, group in test.groupby(splitters):
        specifier = '-'.join(['known' + str(e[1]) if e[0] else 'unknown' + str(e[1])
                              for e in zip(vec, potentially_unknown)])
        unknown = [e[1] for e in zip(vec, potentially_unknown) if not e[0]]
        nans = [col for col in group.columns
                if group[col].dtype == float and np.isnan(group[col]).any()]
        train_crop = train.copy().drop(unknown + nans, axis=1)
        test_group = group.copy().drop(unknown + nans + splitters, axis=1)
        result.append((train_crop, test_group, specifier))
    return result
