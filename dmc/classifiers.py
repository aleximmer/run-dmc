from copy import deepcopy
import pandas as pd
import numpy as np
import theanets as tn
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from dmc.encoding import encode_features


target_feature = 'returnQuantity'
default_ignore_features = ['returnQuantity', 'orderID',
                           'orderDate', 'customerID']


class DMCClassifier:
    classifier = None

    def __init__(self, df: pd.DataFrame, ignore_features=None, X=None, Y=None):
        self.ignore_features = ignore_features if ignore_features is not None \
            else deepcopy(default_ignore_features)
        assert target_feature in self.ignore_features
        self.X = X if X is not None else self.feature_matrix(df)
        self.Y = Y if Y is not None else self.label_vector(df)
        assert len(self.Y) == len(self.X)
        self.clf = self.classifier() if self.classifier else None

    def __call__(self, df: pd.DataFrame) -> np.array:
        self.fit()
        return self.predict(df)

    def fit(self):
        self.clf.fit(self.X, self.Y)

    def predict(self, df: pd.DataFrame) -> np.array:
        X = self.feature_matrix(df)
        return self.clf.predict(X)

    def feature_matrix(self, df: pd.DataFrame) -> np.array:
        X = np.empty((len(df), 0))
        for ft in [ft for ft in df.columns if ft not in self.ignore_features]:
            X = np.append(X, encode_features(df, ft), axis=1)
        return X.astype(np.float32)

    @classmethod
    def label_vector(cls, df: pd.DataFrame) -> np.array:
        return np.squeeze(df.as_matrix(columns=['returnQuantity'])).astype(np.int32)


class DecisionTree(DMCClassifier):
    classifier = DecisionTreeClassifier


class Forest(DMCClassifier):
    def __init__(self, df: pd.DataFrame):
        super().__init__(df)
        self.clf = RandomForestClassifier(n_estimators=100, n_jobs=8)


class NaiveBayes(DMCClassifier):
    classifier = BernoulliNB


class SVM(DMCClassifier):
    def __init__(self, df: pd.DataFrame):
        super().__init__(df)
        self.clf = SVC(decision_function_shape='ovo')


class NeuralNetwork(DMCClassifier):
    def __init__(self, df: pd.DataFrame):
        super().__init__(df)
        input_layer, output_layer = len(self.X.T), 6
        self.clf = tn.Classifier([input_layer, 100, 70, 50, 20, output_layer])

    def fit(self):
        self.clf.train((self.X, self.Y), algo='sgd', learning_rate=1e-4, momentum=0.9)
