import pandas as pd
import numpy as np
import dmc
import os.path

processed_file = '/data/processed_train.csv'


def eval_classifiers(df: pd.DataFrame, tr_size, te_size):
    # shuffle Dataframe
    df = df.reindex(np.random.permutation(df.index))
    df = df[:te_size + tr_size]
    X, Y = dmc.transformation.transform(df, scaler=dmc.normalization.normalize_features)
    train = X[:tr_size], Y[:tr_size]
    test = X[tr_size:tr_size + te_size], Y[tr_size:tr_size + te_size]
    for classifier in dmc.classifiers.DMCClassifier.__subclasses__():
        clf = classifier(train[0].copy(), train[1].copy())
        res = clf(test[0])
        precision = dmc.evaluation.precision(res, test[1])
        print(precision, ' using ', str(classifier))


def processed_data() -> pd.DataFrame:
    rel_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)) + processed_file)
    if os.path.isfile(rel_file_path):
        return pd.DataFrame.from_csv(rel_file_path)
    data = dmc.data_train()
    data = dmc.cleansing.cleanse(data, unproven=True)
    data = dmc.preprocessing.preprocess(data)
    data.to_csv(rel_file_path, sep=',')
    return data


if __name__ == '__main__':
    data = processed_data()
    eval_classifiers(data, 10000, 50000)
