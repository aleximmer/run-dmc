import pandas as pd
from os import listdir


def data_train():
    return pd.read_csv('data/datacup-out-manual.csv', sep=',', na_values='\\N')


def data_full():
    return pd.read_csv('data/datacup-out-unified.csv', sep=',', na_values='\\N')
