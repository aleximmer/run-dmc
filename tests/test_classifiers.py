import unittest
import pandas as pd
from dmc.preprocessing import cleanse
from dmc.features import add_independent_features
from dmc.transformation import transform, normalize_features
from dmc.evaluation import precision
from dmc.classifiers import DecisionTree, Forest, NaiveBayes, SVM
from dmc.classifiers import TreeBag, SVMBag
from dmc.classifiers import AdaTree, AdaBayes, AdaSVM


# Neural Network would be required to have other layers
basic = [DecisionTree, Forest, NaiveBayes, SVM]
bag = [TreeBag, SVMBag]
ada = [AdaTree, AdaBayes, AdaSVM]


class PrimitiveClassifierTest(unittest.TestCase):
    def setUp(self):
        raw_data = pd.read_csv('tests/test_data_old.txt', delimiter=';')
        raw_data = raw_data.head(50)
        clean_data = cleanse(raw_data)
        data = add_independent_features(clean_data)
        X, Y = transform(data, scaler=normalize_features)
        self.X_tr, self.Y_tr = X[:6], Y[:6]
        self.X_te, self.Y_te = X[6:], Y[6:]

    def testClassifers(self):
        for classifier in (basic + bag + ada):
            clf = classifier(self.X_tr, self.Y_tr, False)
            res = clf(self.X_te)
            pred_precision = precision(res, self.Y_te)
            self.assertEqual(pred_precision, 1.0)
