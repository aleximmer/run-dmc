from collections import OrderedDict
from multiprocessing import Pool
import pandas as pd
import numpy as np

from dmc.transformation import transform
from dmc.evaluation import precision


def add_recognition_vector(train: pd.DataFrame, test: pd.DataFrame, columns: list) \
        -> (pd.DataFrame, list):
    """Create a mask of test values seen in training data.
    """
    known_mask = test[columns].copy().apply(lambda column: column.isin(train[column.name]))
    known_mask.columns = ('known_' + c for c in columns)
    return known_mask


def split(train: pd.DataFrame, test: pd.DataFrame, categorical_splits: list) -> dict:
    """For each permutation of known and unknown categories return the cropped train DataFrame and
    the test subset for evaluation.
    """
    known_mask = add_recognition_vector(train, test, categorical_splits)
    test = pd.concat([test, known_mask], axis=1)
    splitters = list(known_mask.columns)
    result = OrderedDict()
    for mask, group in test.groupby(splitters):
        key = ''.join('k' if known else 'u' for known, col in zip(mask, categorical_splits))
        specifier = ''.join('k' + col if known else 'u' + col
                            for known, col in zip(mask, categorical_splits))
        unknown_columns = [col for known, col in zip(mask, categorical_splits) if not known]
        nan_columns = [col for col in group.columns if col != 'returnQuantity'
                       and group[col].dtype == float and np.isnan(group[col]).any()]
        train_crop = train.copy().drop(unknown_columns + nan_columns, axis=1)
        test_group = group.copy().drop(unknown_columns + nan_columns + splitters, axis=1)
        result[key] = {'train': train_crop, 'test': test_group, 'name': specifier}
    return result


class ECEnsemble:
    def __init__(self, train: pd.DataFrame, test: pd.DataFrame, params: dict,
                 categorical_splits=None):
        """
        :param train: train DF
        :param test: test DF
        :param params: dict with the following structure
        Template for params:
        params = {
            'uuuu': {'sample': None, 'scaler': sc, 'ignore_features': None, 'classifier': Bayes},
            'uuku': {'sample': None, 'scaler': sc, 'ignore_features': None, 'classifier': Bayes},
            'ukuu': {'sample': None, 'scaler': sc, 'ignore_features': None, 'classifier': Bayes},
            'ukku': {'sample': None, 'scaler': sc, 'ignore_features': None, 'classifier': Bayes},
            'kuuu': {'sample': None, 'scaler': sc, 'ignore_features': None, 'classifier': Bayes},
            'kuuk': {'sample': None, 'scaler': sc, 'ignore_features': None, 'classifier': Bayes},
            'kuku': {'sample': None, 'scaler': sc, 'ignore_features': None, 'classifier': Bayes},
            'kukk': {'sample': None, 'scaler': sc, 'ignore_features': None, 'classifier': Bayes},
            'kkuu': {'sample': None, 'scaler': sc, 'ignore_features': None, 'classifier': Bayes},
            'kkuk': {'sample': None, 'scaler': sc, 'ignore_features': None, 'classifier': Bayes},
            'kkku': {'sample': None, 'scaler': sc, 'ignore_features': None, 'classifier': Bayes},
            'kkkk': {'sample': None, 'scaler': sc, 'ignore_features': None, 'classifier': Bayes}
        }
        u = unknown, k = known, scaler = None for Trees and else something like scale_features from
        dmc.transformation, ignore_features are the features which should be ignored for the split,

        :return:
        """
        if categorical_splits is None:
            categorical_splits = ['articleID', 'customerID', 'voucherID', 'productGroup']
        self.processes = 2
        self.test = test.copy()
        test = test.dropna(subset=['rrp'])
        self.test_size = len(test)
        self.splits = split(train, test, categorical_splits)
        self._enrich_splits(params)
        # TODO: nans in productGroup, voucherID, rrp result in prediction = 0

    def _enrich_splits(self, params):
        """Each split needs parameters, no defaults exist"""
        for k in self.splits:
            self.splits[k] = {**self.splits[k], **params[k]}

    def transform(self):
        tuples = zip([k for k in self.splits if self.splits],
                     [self.splits[k] for k in self.splits])
        pool = Pool(self.processes)
        splits = pool.map(self._transform_split, tuples, 1)
        for k, d in splits:
            self.splits[k] = d

    @staticmethod
    def _subsample(train: pd.DataFrame, size: int):
        size = min(len(train), size)
        return train.reindex(np.random.permutation(train.index))[:size]

    @staticmethod
    def transform_target_frame(test: pd.DataFrame):
        return pd.DataFrame(test, index=test.index, columns=['returnQuantity'])

    @classmethod
    def _transform_split(cls, splinter: tuple) -> dict:
        key, splinter = splinter
        if splinter['sample']:
            splinter['train'] = cls._subsample(splinter['train'], splinter['sample'])
        offset = len(splinter['train'])
        data = pd.concat([splinter['train'], splinter['test']])
        X, Y = transform(data, binary_target=True, scaler=splinter['scaler'],
                         ignore_features=splinter['ignore_features'])
        splinter['target'] = cls.transform_target_frame(splinter['test'])
        splinter['train'] = (X[:offset], Y[:offset].astype(np.int32))
        if np.isnan(Y[offset:]).any():
            splinter['test'] = (X[offset:], Y[offset:])
        else:
            splinter['test'] = (X[offset:], Y[offset:])
        return key, splinter

    def classify(self, dump_results=False, dump_name=None):
        tuples = zip(self.splits, [self.splits[k] for k in self.splits])
        pool = Pool(self.processes)
        splits = pool.map(self._classify_split, tuples, 1)
        for k, d in splits:
            self.splits[k] = d
        self.report()
        if dump_results:
            self.dump_results(dump_name)

    @staticmethod
    def _classify_split(splinter: tuple) -> dict:
        key, splinter = splinter
        clf = splinter['classifier'](splinter['train'][0], splinter['train'][1], splinter['optimize'], key)
        ypr = clf(splinter['test'][0])
        try:
            probs = np.max(clf.predict_proba(splinter['test'][0]), 1)
            splinter['target']['confidence'] = np.squeeze(probs)
        except Exception as e:
            print('Classifier offers no predict_proba method', e)
            splinter['target']['confidence'] = np.nan
        splinter['target']['prediction'] = ypr
        # returnQuantity can be nan for class data
        splinter['target']['returnQuantity'] = splinter['test'][1]
        # free memory
        splinter['train'] = None
        splinter['test'] = None
        print('Finished fitting', key, 'with', splinter['classifier'])
        return key, splinter

    def report(self):
        precs = []
        for k in self.splits:
            # If no returnQuantity (target-set) is given, we cannot compute precisions
            if not np.isnan(self.splits[k]['target'].returnQuantity).any():
                prec = precision(self.splits[k]['target'].returnQuantity,
                                 self.splits[k]['target'].prediction)
                print(k, 'precision', prec, 'size', len(self.splits[k]['target']))
                precs.append(prec)
        partials = np.array([len(self.splits[k]['target']) for k in self.splits]) / self.test_size
        if precs:
            precs = np.array(precs)
            print('OVERALL:', np.sum(np.multiply(precs, partials)))
        else:
            print('Target set has no evaluation labels')

    def dump_results(self, file_name):
        test = self.test
        predicted = pd.concat([self.splits[k]['target'] for k in self.splits])
        test['prediction'] = predicted.prediction.astype(int)
        test['confidence'] = predicted.confidence

        test.loc[test['rrp'].isnull(), 'prediction'] = 0
        test.loc[test['rrp'].isnull(), 'confidence'] = 1.0

        res = pd.DataFrame(test, test.index, columns=['orderID', 'articleID', 'colorCode',
                                                      'sizeCode', 'quantity', 'confidence',
                                                      'prediction'])
        res.to_csv('data/' + file_name + '.csv', sep=';')
