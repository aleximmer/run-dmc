import unittest
import pandas as pd
import dmc
from dmc.classifiers import DecisionTree, Forest, NaiveBayes, SVM, NeuralNetwork
from dmc.classifiers import TreeBag, BayesBag, SVMBag
from dmc.classifiers import AdaTree, AdaBayes, AdaSVM


# Neural Network would be required to have other layers
basic = [DecisionTree, Forest, NaiveBayes, SVM]
bag = [TreeBag, BayesBag, SVMBag]
ada = [AdaTree, AdaBayes, AdaSVM]


class PrimitiveClassifierTest(unittest.TestCase):
    def setUp(self):
        df = pd.read_csv('tests/test_data.txt', delimiter=';')
        df = dmc.cleansing.cleanse(df, unproven=True)
        df = dmc.preprocessing.preprocess(df)
        X, Y = dmc.transformation.transform(df, scaler=dmc.normalization.normalize_features)
        self.X_tr, self.Y_tr = X[:6], Y[:6]
        self.X_te, self.Y_te = X[6:], Y[6:]

    def testClassifers(self):
        for classifier in (basic + bag + ada):
            clf = classifier(self.X_tr, self.Y_tr)
            res = clf(self.X_te)
            precision = dmc.evaluation.precision(res, self.Y_te)
            self.assertEqual(precision, 1.0)
