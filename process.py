import os.path
import argparse
import pandas as pd
import numpy as np
import sys

import dmc
from dmc.classifiers import DecisionTree, Forest, NaiveBayes, SVM, TheanoNeuralNetwork, \
    TensorFlowNeuralNetwork
from dmc.classifiers import TreeBag, SVMBag
from dmc.classifiers import AdaTree, AdaBayes, AdaSVM, GradBoost
from dmc.ensemble import Ensemble


processed_file = '/data/processed.csv'
processed_full_file = '/data/processed_full.csv'

# Remove classifiers which you don't want to run and add new ones here
basic = [DecisionTree, Forest, NaiveBayes, SVM, TheanoNeuralNetwork, TensorFlowNeuralNetwork]
bag = [TreeBag, SVMBag, GradBoost]
ada = [AdaTree, AdaBayes, AdaSVM]


def eval_classifiers(df: pd.DataFrame, split: int, tune_parameters: bool, clas=None):
    X, Y = dmc.transformation.transform(df, scaler=dmc.transformation.scale_raw_features,
                                        binary_target=True)
    print('classifier', clas)
    train = X[:split], Y[:split]
    test = X[split:], Y[split:]
    clf = clas(train[0], train[1], tune_parameters)
    res = clf(test[0])
    precision = dmc.evaluation.precision(res, test[1])
    print('precision', precision)


def eval_ensemble(train: pd.DataFrame, test: pd.DataFrame):
    ensemble = Ensemble(train, test)
    ensemble.transform(binary_target=True)
    ensemble.classify()


def eval_features(df: pd.DataFrame):
    ft_importance = dmc.evaluation.evaluate_features_by_ensemble(df)
    print(ft_importance.sort_values('tree', ascending=False))


def processed_data(load_train=True) -> pd.DataFrame:
    if load_train:
        df = dmc.loading.data_train()
    else:
        df = dmc.loading.data_class()
    df = dmc.preprocessing.cleanse(df)
    df = dmc.features.add_independent_features(df)
    return df


def load_data(load_full=False) -> pd.DataFrame:
    if load_full:
        print('Load full data')
        data = dmc.loading.preprocessed_full_data()
    data = dmc.loading.preprocessed_data()

    if not data:
        data = processed_data(load_train=True)
        if load_full:
            class_data = processed_data(load_train=False)
            data = pd.concat([data, class_data])
            print('concat')
    dmc.loading.dump_data(load_full)
    return data


def split_data_by_id(df: pd.DataFrame, id_file_prefix: str) -> (pd.DataFrame, int):
    train_ids, test_ids = dmc.loading.load_ids(id_file_prefix)
    train, test = dmc.preprocessing.split_train_test(df, train_ids, test_ids)
    train, test = dmc.features.add_dependent_features(train, test)
    return train, test


def shuffle(df: pd.DataFrame) -> pd.DataFrame:
    return df.reindex(np.random.permutation(df.index))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('id_prefix', help='prefix of the id file to use')
    parser.add_argument('-f', action='store_true', help='load the unified dataset (train + class)')
    args = parser.parse_args()
    id_prefix = args.id_prefix
    load_full = args.f

    data = load_data(load_full)

    sys.exit()
    train_ids, test_ids = dmc.loading.load_ids(id_prefix)
    train, test = dmc.preprocessing.split_train_test(data, train_ids, test_ids)
    split_point = len(train)

    train = shuffle(train[:split_point])[:3000]
    data = pd.concat([train, test])

    # eval_ensemble(train, test)
    print(split_point, len(data), len(train))
    print('start evaluation')
    eval_classifiers(data, 3000, tune_parameters=True, clas=SVM)
    # eval_features(data[:split_point])
