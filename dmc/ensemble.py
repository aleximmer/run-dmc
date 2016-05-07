import pandas as pd
import numpy as np

from dmc.transformation import normalize_features
from dmc.transformation import transform
from dmc.classifiers import Forest
from dmc.evaluation import precision, dmc_cost


def add_recognition_vector(train: pd.DataFrame, test: pd.DataFrame, columns: list) \
        -> (pd.DataFrame, list):
    """Create a mask of test values seen in training data.
    """
    known_mask = test[columns].copy().apply(lambda column: column.isin(train[column.name]))
    known_mask.columns = ('known_' + c for c in columns)
    return known_mask


def split(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """For each permutation of known and unknown categories return the cropped train DataFrame and
    the test subset for evaluation.
    """
    potentially_unknown = ['articleID', 'customerID', 'voucherID', 'productGroup']
    known_mask = add_recognition_vector(train, test, potentially_unknown)
    test = pd.concat([test, known_mask], axis=1)
    splitters = list(known_mask.columns)
    result = dict()
    for mask, group in test.groupby(splitters):
        specifier = '-'.join('known_' + col if known else 'unknown_' + col
                             for known, col in zip(mask, potentially_unknown))
        unknown_columns = [col for known, col in zip(mask, potentially_unknown) if not known]
        nan_columns = [col for col in group.columns
                       if group[col].dtype == float and np.isnan(group[col]).any()]
        train_crop = train.copy().drop(unknown_columns + nan_columns, axis=1)
        test_group = group.copy().drop(unknown_columns + nan_columns + splitters, axis=1)
        result[specifier] = (train_crop, test_group)
    return result


class Ensemble:
    def __init__(self, train: pd.DataFrame, test: pd.DataFrame):
        self.train = train
        self.test = test
        self.splits = split(train, test)

    def transform(self, binary_target=True, scalers=None):
        scalers = [normalize_features] * len(self.splits) if scalers is None else scalers
        for s, scaler in zip(self.splits, scalers):
            offset = len(self.splits[s][0])
            data = pd.concat([self.splits[s][0], self.splits[s][1]])
            X, Y = transform(data, binary_target=binary_target, scaler=scaler)
            self.splits[s] = ({
                'train': (X[:offset], Y[:offset]),
                'test': (X[offset:], Y[offset:])
            })

    def classify(self, classifiers=None):
        results = []
        classifiers = [Forest] * len(self.splits) if classifiers is None else classifiers
        for s, classifier in zip(self.splits, classifiers):
            clf = classifier(*self.splits[s]['train'])
            pred = clf(self.splits[s]['test'][0])
            prec = precision(pred, self.splits[s]['test'][1])
            cost = dmc_cost(pred, self.splits[s]['test'][1])
            results.append((len(self.splits[s]['test'][1]), prec, cost))
        all_prec, all_cost, full_size = 0, 0, len(self.test)
        for size, prec, cost in results:
            all_prec += size / full_size * prec
            all_cost += cost
        print(all_prec, all_cost)
