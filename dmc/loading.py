import pandas as pd


def data_train():
    return pd.read_csv('data/datacup-out-manual.csv', sep=',', na_values='\\N')


def data_full():
    return pd.read_csv('data/datacup-out-unified.csv', sep=',', na_values='\\N')


def load_ids(id_file_prefix: str) -> (list, list):
    """Load the test and train ids of a given file prefix
    (e.g. 'rawFirstOrders[Test.txt|Training.txt]').
    """
    train_path = 'data/idLists/' + id_file_prefix + 'Training.txt'
    test_path = 'data/idLists/' + id_file_prefix + 'Test.txt'
    with open(train_path, 'r') as f:
        train_ids = [line.strip().replace("\"", '') for line in f]
    with open(test_path, 'r') as f:
        test_ids = [line.strip().replace("\"", '') for line in f]
    return train_ids, test_ids
