import pandas as pd


def data_train():
    return pd.read_csv('data/orders_train.txt', sep=';')


def data_class():
    return pd.read_csv('data/orders_class.txt', sep=';')
