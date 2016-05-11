import os.path
import argparse
import pandas as pd

import dmc
from dmc.classifiers import DecisionTree, Forest, NaiveBayes, SVM, TheanoNeuralNetwork, \
    TensorFlowNeuralNetwork
from dmc.classifiers import TreeBag, SVMBag
from dmc.classifiers import AdaTree, AdaBayes, AdaSVM, GradBoost
from dmc.ensemble import Ensemble


processed_file = '/data/processed.csv'

# Remove classifiers which you don't want to run and add new ones here
basic = [DecisionTree, Forest, NaiveBayes, SVM, TheanoNeuralNetwork, TensorFlowNeuralNetwork]
bag = [TreeBag, SVMBag, GradBoost]
ada = [AdaTree, AdaBayes, AdaSVM]


def eval_classifiers(df: pd.DataFrame, split: int, tune_parameters: bool):
    for scaler in [dmc.transformation.normalize_raw_features]:
        X, Y = dmc.transformation.transform(df, scaler=scaler,
                                            binary_target=True)
        train = X[:split], Y[:split]
        test = X[split:], Y[split:]
        clf = NaiveBayes(train[0], train[1], tune_parameters)
        res = clf(test[0])
        precision = dmc.evaluation.precision(res, test[1])
        print(precision, ' using ', 'Naive Bayes and', scaler)


def eval_ensemble(train: pd.DataFrame, test: pd.DataFrame):
    ensemble = Ensemble(train, test)
    ensemble.transform(binary_target=True)
    ensemble.classify()


def eval_features(df: pd.DataFrame):
    ft_importance = dmc.evaluation.evaluate_features_by_ensemble(df)
    print(ft_importance.sort_values('tree', ascending=False))


def processed_data() -> pd.DataFrame:
    """Create or read DataFrame with all features that are independent"""
    rel_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)) + processed_file)
    if os.path.isfile(rel_file_path):
        return pd.DataFrame.from_csv(rel_file_path)
    df = dmc.loading.data_train()
    df = dmc.preprocessing.cleanse(df)
    df = dmc.features.add_independent_features(df)
    print('Finished processing. Dumping results to {}.'.format(rel_file_path))
    df.to_csv(rel_file_path, sep=',')
    return df


def split_data_by_id(df: pd.DataFrame, id_file_prefix: str) -> (pd.DataFrame, int):
    train_ids, test_ids = dmc.loading.load_ids(id_file_prefix)
    train, test = dmc.preprocessing.split_train_test(df, train_ids, test_ids)
    train, test = dmc.features.add_dependent_features(train, test)
    return train, test


if __name__ == '__main__':

    id_prefix = 'rawSummerSale'

    data = processed_data()
    train, test = split_data_by_id(data, id_prefix)
    split_point = len(train)

    #eval_ensemble(train, test)
    print('start evaluation')
    eval_classifiers(data, split_point, tune_parameters=False)
    #eval_features(data[:split_point])
