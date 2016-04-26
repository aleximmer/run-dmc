import dmc
import numpy as np
import os.path
import pandas as pd
from dmc.classifiers import AdaTree
from dmc.classifiers import DecisionTree, Forest
from dmc.classifiers import TreeBag

processed_file = '/data/processed_train.csv'

# Remove classifiers which you don't want to run and add new ones here
basic = [DecisionTree, Forest]
bag = [TreeBag]
ada = [AdaTree]


def shuffle(df: pd.DataFrame) -> pd.DataFrame:
    return df.reindex(np.random.permutation(df.index))


def eval_classifiers(df: pd.DataFrame, tr_size, te_size):
    df = shuffle(df)
    df = df[:te_size + tr_size]
    X, Y = dmc.transformation.transform(df, scaler=dmc.normalization.normalize_features,
                                        binary_target=True)
    train = X[:tr_size], Y[:tr_size]
    test = X[tr_size:tr_size + te_size], Y[tr_size:tr_size + te_size]
    for classifier in (basic + bag + ada):
        clf = classifier(train[0], train[1])
        res = clf(test[0])
        precision = dmc.evaluation.precision(res, test[1])
        print(precision, ' using ', str(classifier))


def eval_features(df: pd.DataFrame, size):
    df = shuffle(df)
    ft_importance = dmc.evaluation.eval_features_by_ensemble(df[:size])
    print(ft_importance.sort_values('tree', ascending=False))


def processed_data() -> pd.DataFrame:
    rel_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)) +
                                 processed_file)
    if os.path.isfile(rel_file_path):
        return pd.DataFrame.from_csv(rel_file_path)
    data = dmc.data_train()
    data = dmc.cleansing.cleanse(data, unproven=True)
    data = dmc.preprocessing.preprocess(data)
    feature_dfs = dmc.data_features()
    feature_dfs = [dmc.cleansing.cleanse(df, unproven=True) for df in feature_dfs]
    data = dmc.preprocessing.merge_features(data, feature_dfs)
    print('Finished processing. Dumping results to {}.'.format(rel_file_path))
    data.to_csv(rel_file_path, sep=',')
    return data


if __name__ == '__main__':
    data = processed_data()
    eval_classifiers(data, 5000, 5000)
    # eval_features(data, 5000)
