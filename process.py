import os.path
import argparse
import pandas as pd
import numpy as np

import dmc
from dmc.classifiers import DecisionTree, Forest, NaiveBayes, SVM, TheanoNeuralNetwork, \
    TensorFlowNeuralNetwork
from dmc.classifiers import TreeBag, SVMBag
from dmc.classifiers import AdaTree, AdaBayes, AdaSVM, GradBoost


processed_file = '/data/processed.csv'
processed_full_file = '/data/processed_full.csv'

# Remove classifiers which you don't want to run and add new ones here
basic = [DecisionTree, Forest, NaiveBayes, SVM, TheanoNeuralNetwork, TensorFlowNeuralNetwork]
bag = [TreeBag, SVMBag, GradBoost]
ada = [AdaTree, AdaBayes, AdaSVM]


def eval_classifiers(df: pd.DataFrame, split: int, tune_parameters: bool, clas=None):
    X, Y = dmc.transformation.transform(df, scaler=dmc.transformation.scale_features,
                                        binary_target=True)
    print('classifier', clas)
    train = X[:split], Y[:split]
    test = X[split:], Y[split:]
    clf = clas(train[0], train[1], tune_parameters)
    res = clf(test[0])
    precision = dmc.evaluation.precision(res, test[1])
    print('precision', precision)


def eval_features(df: pd.DataFrame):
    ft_importance = dmc.evaluation.evaluate_features_by_ensemble(df)
    print(ft_importance.sort_values('tree', ascending=False))


def processed_data(load_full=False) -> pd.DataFrame:
    """Create or read DataFrame with all features that are independent"""
    rel_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)) + processed_file)
    rel_file_path_full = os.path.join(os.path.dirname(os.path.realpath(__file__))
            + processed_full_file)
    if os.path.isfile(rel_file_path) and not load_full:
        return pd.DataFrame.from_csv(rel_file_path)
    if os.path.isfile(rel_file_path_full) and load_full:
        df = pd.DataFrame.from_csv(rel_file_path_full)
        df.sizeCode = df.sizeCode.astype(str)
        return df
    if load_full:
        df = dmc.loading.data_full()
    else:
        df = dmc.loading.data_train()
    df = dmc.preprocessing.cleanse(df)
    print('len after preprocessing', len(df))
    df = dmc.features.add_independent_features(df)
    if load_full:
        df.to_csv(rel_file_path_full, sep=',')
        print('Finished processing. Dumped results to {}.'.format(rel_file_path_full))
    else:
        df.to_csv(rel_file_path, sep=',')
        print('Finished processing. Dumped results to {}.'.format(rel_file_path))
    return df


def split_data_by_id(df: pd.DataFrame, id_file_prefix: str) -> (pd.DataFrame, int):
    train_ids, test_ids = dmc.loading.load_ids(id_file_prefix)
    train, test = dmc.preprocessing.split_train_test(df, train_ids, test_ids)
    train, test = dmc.features.add_dependent_features(train, test)
    return train, test


def split_data_at_id(df: pd.DataFrame, orderID: int) -> (pd.DataFrame, pd.DataFrame):
    train = df[df.orderID < orderID]
    test = df[df.orderID >= orderID]
    #train, test = dmc.features.add_dependent_features(train, test)
    return train, test


def shuffle(df: pd.DataFrame) -> pd.DataFrame:
    return df.reindex(np.random.permutation(df.index))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', action='store_true', help='load the unified dataset (train + class)')
    args = parser.parse_args()
    load_full = args.f

    data = processed_data(True)
    data = data[~data.returnQuantity.isnull()]
    #train_ids, test_ids = dmc.loading.load_ids(id_prefix)
    train, test = split_data_at_id(data, 1527394)
    split_point = len(train)

    #train = shuffle(train[:split_point])[:3000]
    data = pd.concat([train, test])


    #eval_ensemble(train, test)
    print(split_point, len(data), len(train))
    print('start evaluation')
    eval_classifiers(data, 1000000, tune_parameters=True, clas=TensorFlowNeuralNetwork)
    #eval_features(data[:split_point])
