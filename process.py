import os.path
import argparse
import pandas as pd

import dmc
from dmc.classifiers import DecisionTree, Forest, NaiveBayes, SVM, NeuralNetwork
from dmc.classifiers import TreeBag, BayesBag, SVMBag
from dmc.classifiers import AdaTree, AdaBayes, AdaSVM
from dmc.ensemble import Ensemble


processed_file = '/data/processed.csv'

# Remove classifiers which you don't want to run and add new ones here
basic = [DecisionTree, Forest, NaiveBayes, SVM, NeuralNetwork]
bag = [TreeBag, BayesBag, SVMBag]
ada = [AdaTree, AdaBayes, AdaSVM]


def eval_classifiers(df: pd.DataFrame, split: int):
    X, Y = dmc.transformation.transform(df, scaler=dmc.transformation.scale_features,
                                        binary_target=True)
    train = X[:split], Y[:split]
    test = X[split:], Y[split:]
    for classifier in (basic + bag + ada):
        clf = classifier(train[0], train[1])
        res = clf(test[0])
        precision = dmc.evaluation.precision(res, test[1])
        print(precision, ' using ', str(classifier))


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
        return pd.read_csv(rel_file_path)
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
    parser = argparse.ArgumentParser()
    parser.add_argument('id_prefix', help='prefix of the id file to use')
    args = parser.parse_args()
    id_prefix = args.id_prefix

    data = processed_data()
    train, test = split_data_by_id(data, id_prefix)
    split_point = len(train)

    eval_ensemble(train, test)
    #eval_classifiers(data, split_point)
    #eval_features(data[:split_point])
