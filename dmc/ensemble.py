import pandas as pd


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
        test[column_name] = test.apply(lambda x: x not in unknown)
    return test, recognition_columns


def split(train: pd.DataFrame, test: pd.DataFrame) -> list():
    """For each permutation of known and unknown for specific categories
    return the cropped train DataFrame and the test subset for evaluation"""
    test, splitters = add_recognition_vector(train, test)
    result = []
    for split, group in test.groupby(splitters):
        result.append((train, group, split))
    return result
