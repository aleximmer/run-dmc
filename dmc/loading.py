import pandas as pd
import os


processed_file = '/data/processed.csv'
processed_full_file = '/data/processed_full.csv'
rel_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)) + processed_file)
rel_file_path_full = os.path.join(os.path.dirname(os.path.realpath(__file__)) + processed_full_file)


def data_train():
    return pd.read_csv('data/datacup-out-training_test_nosplit_transformed.csv',
                       sep=',', na_values='\\N')


def data_class():
    return pd.read_csv('data/datacup-out-class_nosplit_transformed.csv',
                       sep=',', na_values='\\N')


def preprocessed_data():
    if os.path.isfile(rel_file_path):
        return pd.DataFrame.from_csv(rel_file_path)


def preprocessed_full_data():
    if os.path.isfile(rel_file_path):
        return pd.DataFrame.from_csv(rel_file_path_full)


def dump_data(df, full=False):
    if full:
        df.to_csv(rel_file_path_full)
    df.to_csv(rel_file_path)


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
