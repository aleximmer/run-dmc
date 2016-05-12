from multiprocessing import Pool
from collections import OrderedDict
import pandas as pd
import numpy as np

from dmc.transformation import normalize_features
from dmc.transformation import transform, transform_preserving_header
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
    result = OrderedDict()
    for mask, group in test.groupby(splitters):
        key = ''.join('k' if known else 'u' for known,col in zip(mask, potentially_unknown))
        specifier = ''.join('k' + col if known else 'u' + col
                             for known, col in zip(mask, potentially_unknown))
        unknown_columns = [col for known, col in zip(mask, potentially_unknown) if not known]
        nan_columns = [col for col in group.columns
                       if group[col].dtype == float and np.isnan(group[col]).any()]
        train_crop = train.copy().drop(unknown_columns + nan_columns, axis=1)
        test_group = group.copy().drop(unknown_columns + nan_columns + splitters, axis=1)
        result[key] = {'train': train_crop, 'test': test_group, 'name': specifier}
    return result


class ECEnsemble:
    def __init__(self, train: pd.DataFrame, test: pd.DataFrame, params: dict):
        self.test_size = len(test)
        self.splits = split(train, test)
        self._enrich_splits(params)
        # TODO: nans in productGroup, voucherID, rrp result in prediction = 0

    def _enrich_splits(self, params):
        """Each split needs parameters, no defaults exist"""
        for k in self.splits:
            self.splits[k] = {**self.splits[k], **params[k]}

    def transform(self):
        for k in self.splits:
            self.splits[k] = self._transform_split(self.splits[k])

    @staticmethod
    def _subsample(train: pd.DataFrame, size: int):
        size = min(len(train), size)
        return train.reindex(np.random.permutation(train.index))[:size]

    @staticmethod
    def transform_target_frame(test: pd.DataFrame):
        return pd.DataFrame(test, columns=['returnQuantity'])

    @classmethod
    def _transform_split(cls, splinter: dict) -> dict:
        if splinter['sample']:
            splinter['train'] = cls._subsample(splinter['train'], splinter['sample'])
        offset = len(splinter['train'])
        data = pd.concat([splinter['train'], splinter['test']])
        X, Y = transform(data, binary_target=True, scaler=splinter['scaler'],
                         ignore_features=splinter['ignore_features'])
        splinter['target'] = cls.transform_target_frame(splinter['test'])
        splinter['train'] = (X[:offset], Y[:offset])
        splinter['test'] = (X[offset:], Y[offset:])
        return splinter

    def classify(self):
        for k in self.splits:
            self.splits[k] = self._classify_split(self.splits[k])
        self.report()
        self.dump_results()

    @staticmethod
    def _classify_split(splinter: dict) -> dict:
        clf = splinter['classifier'](*splinter['train'])
        ypr = clf(splinter['test'][0])
        # TODO: add predict proba
        splinter['target']['prediction'] = ypr
        splinter['target']['returnQuantity'] = splinter['test'][1]
        return splinter

    def report(self):
        precs = []
        for k in self.splits:
            if not np.isnan(self.splits[k]['target'].returnQuantity).any():
                prec = precision(self.splits[k]['target'].returnQuantity,
                                 self.splits[k]['target'].prediction)
                print(k, 'precision', prec, 'size', len(self.splits[k]['target']))
                precs.append(prec)
        partials = np.array([len(self.splits[k]['target']) for k in self.splits])/self.test_size
        if precs:
            precs = np.array(precs)
            print('OVERALL:', np.sum(np.multiply(precs, partials)))

    def dump_results(self):
        pass


"""
class Ensemble:
    def __init__(self, train: pd.DataFrame, test: pd.DataFrame):
        self.test_size = len(test)
        self.splits = split(train, test)
        self.pool = Pool(processes=1)

    def transform(self, binary_target=True, scalers=None, ignore_features=None):
        scalers = [normalize_features] * len(self.splits) if scalers is None else scalers
        ignore_features = ignore_features if ignore_features else [None] * len(self.splits)
        binary = [binary_target] * len(self.splits)

        transformation_tuples = zip(self.splits.items(), scalers, ignore_features, binary)
        # trans = self.pool.map(self._transform_split, transformation_tuples)
        trans = []
        for elem in transformation_tuples:
            trans.append(self._transform_split(elem[]))
        for res in trans:
            self.splits[res[0]] = res[1]

    @staticmethod
    def _transform_split(args):
        item, scaler, rm_features, binary_target = args
        offset = len(item[1][0])
        data = pd.concat([item[1][0], item[1][1]])
        X, Y, fts = transform_preserving_header(data, binary_target=binary_target,
                                                scaler=scaler, ignore_features=rm_features)
        return item[0], {
            'features': fts,
            'train': (X[:offset], Y[:offset]),
            'test': (X[offset:], Y[offset:])}

    def classify(self, classifiers=None, hyper_param=False, verbose=True):
        classifiers = [DecisionTree] * len(self.splits) if classifiers is None else classifiers
        split_apply = zip(self.splits.items(), classifiers, [hyper_param] * len(self.splits))

        results = self.pool.map(self._classify_split, split_apply)

        all_prec, all_cost, full_size = 0, 0, self.test_size
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
"""
