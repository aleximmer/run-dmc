from multiprocessing import Pool, cpu_count
import pandas as pd
import numpy as np

from dmc.transformation import normalize_features
from dmc.transformation import transform
from dmc.classifiers import DecisionTree
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
        self.pool = Pool(processes=3)

    def transform(self, binary_target=True, scalers=None, ignore_features=None):
        scalers = [normalize_features] * len(self.splits) if scalers is None else scalers
        ignore_features = ignore_features if ignore_features else [None] * len(self.splits)
        binary = [binary_target] * len(self.splits)

        transformation_tuples = zip(self.splits.items(), scalers, ignore_features, binary)
        trans = self.pool.map(self._transform_split, transformation_tuples)
        for res in trans:
            print(res[0], res[0] in self.splits.keys())
            self.splits[res[0]] = res[1]

    @staticmethod
    def _transform_split(args):
        item, scaler, rm_features, binary_target = args
        offset = len(item[1][0])
        data = pd.concat([item[1][0], item[1][1]])
        if rm_features:
            data = data.drop(rm_features, 1)
        X, Y = transform(data, binary_target=binary_target, scaler=scaler)
        return item[0], {
            'train': (X[:offset], Y[:offset]),
            'test': (X[offset:], Y[offset:])}

    def classify(self, classifiers=None, hyper_param=False, verbose=True):
        classifiers = [DecisionTree] * len(self.splits) if classifiers is None else classifiers
        split_apply = zip(self.splits.items(), classifiers, [hyper_param] * len(self.splits))

        results = self.pool.map(self._classify_split, split_apply)

        all_prec, all_cost, full_size = 0, 0, len(self.test)
        for size, prec, cost, name, importances in results:
            all_prec += size / full_size * prec
            all_cost += cost
            if verbose:
                print(name, ': size ', size, ' prec ', prec)
                print('--------------------------------------')

        print('Overall :', all_prec, all_cost)

    @staticmethod
    def _classify_split(args):
        split, classifier, hyper_param = args
        clf = classifier(*split[1]['train'], tune_parameters=hyper_param)
        pred = clf(split[1]['test'][0])
        prec = precision(pred, split[1]['test'][1])
        cost = dmc_cost(pred, split[1]['test'][1])
        try:
            importances = clf.clf.feature_importances_
        except:
            importances = None
        return len(split[1]['test'][1]), prec, cost, split[0], importances
