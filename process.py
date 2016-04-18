import pandas as pd
import numpy as np
import dmc


def eval_classifiers(df: pd.DataFrame, tr_size, te_size):
    # shuffle Dataframe
    df = df.reindex(np.random.permutation(df.index))
    train = df[:tr_size]
    test = df[tr_size:tr_size + te_size]
    for classifier in dmc.classifiers.DMCClassifier.__subclasses__():
        clf = classifier(train)
        res = clf(test)
        precision = dmc.evaluation.precision(res, clf.label_vector(test).T)
        print(precision, ' using ', str(classifier))


def processed_data() -> pd.DataFrame:
    data = dmc.data_train()
    data = dmc.cleansing.cleanse(data, unproven=True)
    data = dmc.preprocessing.preprocess(data)
    return data


if __name__ == '__main__':
    data = processed_data()
    eval_classifiers(data, 1000000, 500000)
