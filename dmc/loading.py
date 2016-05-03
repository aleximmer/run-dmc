import pandas as pd


def data_train():
    return pd.read_csv('data/datacup-out-manual.csv', sep=',', na_values='\\N')


def data_full():
    return pd.read_csv('data/datacup-out-unified.csv', sep=',', na_values='\\N')
