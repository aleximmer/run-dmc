import pandas as pd
import numpy as np


def add_recognition_vector(train: pd.DataFrame, test: pd.DataFrame, columns: list) \
        -> (pd.DataFrame, list):
    """Create a mask of test values seen in training data.
    """
    known_mask = test[columns].copy().apply(lambda column: column.isin(train[column.name]))
    known_mask.columns = ('known_' + c for c in columns)
    return known_mask


def split(train: pd.DataFrame, test: pd.DataFrame) -> list():
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
