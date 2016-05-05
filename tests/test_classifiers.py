import unittest
import pandas as pd

from dmc.preprocessing import cleanse, feature
from dmc.transformation import transform, normalize_features
from dmc.evaluation import precision
from dmc.classifiers import DecisionTree, Forest, NaiveBayes, SVM
from dmc.classifiers import TreeBag, BayesBag, SVMBag
from dmc.classifiers import AdaTree, AdaBayes, AdaSVM


# Neural Network would be required to have other layers
basic = [DecisionTree, Forest, NaiveBayes, SVM]
bag = [TreeBag, BayesBag, SVMBag]
ada = [AdaTree, AdaBayes, AdaSVM]


class PrimitiveClassifierTest(unittest.TestCase):
    def setUp(self):
        df = pd.read_csv('tests/test_data.txt', delimiter=';')
        df = cleanse(df)
        df = feature(df)
        X, Y = transform(df, scaler=normalize_features)
        self.X_tr, self.Y_tr = X[:6], Y[:6]
        self.X_te, self.Y_te = X[6:], Y[6:]

    def testClassifers(self):
        for classifier in (basic + bag + ada):
            clf = classifier(self.X_tr, self.Y_tr)
            res = clf(self.X_te)
            pred_precision = precision(res, self.Y_te)
            self.assertEqual(pred_precision, 1.0)
