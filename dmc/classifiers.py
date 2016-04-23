import pandas as pd
import numpy as np
import theanets as tn
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, \
    BaggingClassifier, AdaBoostClassifier


class DMCClassifier:
    classifier = None

    def __init__(self, X: np.array, Y: np.array):
        assert len(Y) == len(X)
        self.X = X
        self.Y = Y
        self.clf = self.classifier if self.classifier else None

    def __call__(self, df: pd.DataFrame) -> np.array:
        self.fit()
        return self.predict(df)

    def fit(self):
        self.clf.fit(self.X, self.Y)
        return self

    def predict(self, X: np.array) -> np.array:
        return self.clf.predict(X)


class DecisionTree(DMCClassifier):
    classifier = DecisionTreeClassifier()


class Forest(DMCClassifier):
    def __init__(self, X: np.array, Y: np.array):
        super().__init__(X, Y)
        self.clf = RandomForestClassifier(n_estimators=100, n_jobs=8)


class NaiveBayes(DMCClassifier):
    classifier = BernoulliNB()


class SVM(DMCClassifier):
    def __init__(self, X: np.array, Y: np.array):
        super().__init__(X, Y)
        self.clf = SVC(decision_function_shape='ovo')


class NeuralNetwork(DMCClassifier):
    def __init__(self, X: np.array, Y: np.array):
        super().__init__(X, Y)
        input_layer, output_layer = len(self.X.T), 6
        self.clf = tn.Classifier([input_layer, 100, 70, 50, 20, output_layer])

    def fit(self):
        self.clf.train((self.X, self.Y), algo='sgd', learning_rate=1e-4, momentum=0.9)
        return self


class BagEnsemble(DMCClassifier):
    estimators = 10
    max_features = .5
    max_samples = .5

    def __init__(self, X: np.array, Y: np.array):
        super().__init__(X, Y)
        self.clf = BaggingClassifier(self.classifier, n_estimators=self.estimators, n_jobs=8,
                                     max_samples=self.max_samples, max_features=self.max_features)


class TreeBag(BagEnsemble):
    classifier = DecisionTreeClassifier()


class BayesBag(BagEnsemble):
    classifier = BernoulliNB()


class SVMBag(BagEnsemble):
    def __init__(self, X: np.array, Y: np.array):
        self.classifier = SVC(decision_function_shape='ovo')
        super().__init__(X, Y)


class AdaBoostEnsemble(DMCClassifier):
    estimators = 50
    learning_rate = .5
    algorithm = 'SAMME.R'

    def __init__(self, X: np.array, Y: np.array):
        super().__init__(X, Y)
        self.clf = AdaBoostClassifier(self.classifier, n_estimators=self.estimators,
                                      learning_rate=self.learning_rate, algorithm=self.algorithm)


class AdaTree(AdaBoostEnsemble):
    classifier = DecisionTreeClassifier()


class AdaBayes(AdaBoostEnsemble):
    classifier = BernoulliNB()


class AdaSVM(AdaBoostEnsemble):
    algorithm = 'SAMME'

    def __init__(self, X: np.array, Y: np.array):
        self.classifier = SVC(decision_function_shape='ovo')
        super().__init__(X, Y)
