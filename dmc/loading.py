import pandas as pd
from os import listdir


def data_train():
    return pd.read_csv('data/orders_train.txt', sep=';')


def data_class():
    return pd.read_csv('data/orders_class.txt', sep=';')


def data_features():
    PATH = 'data/features/'
    filenames = listdir(PATH)
    return [pd.read_csv(PATH + f, sep=',') for f in filenames if f.endswith('.csv')]
