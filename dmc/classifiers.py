import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import theanets as tn
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, \
    BaggingClassifier, AdaBoostClassifier
import tensorflow.contrib.learn as skflow


class DMCClassifier:
    clf = None

    def __init__(self, X: csr_matrix, Y: np.array):
        assert len(Y) == X.shape[0]
        self.X = X
        self.Y = Y

    def __call__(self, df: pd.DataFrame) -> np.array:
        self.fit()
        return self.predict(df)

    def fit(self):
        self.clf.fit(self.X, self.Y)
        return self

    def predict(self, X: csr_matrix) -> np.array:
        return self.clf.predict(X)


class DecisionTree(DMCClassifier):
    clf = DecisionTreeClassifier()


class Forest(DMCClassifier):
    def __init__(self, X: csr_matrix, Y: np.array):
        super().__init__(X, Y)
        self.clf = RandomForestClassifier(n_estimators=100, n_jobs=8)


class NaiveBayes(DMCClassifier):
    clf = BernoulliNB(binarize=True)


class SVM(DMCClassifier):
    def __init__(self, X: csr_matrix, Y: np.array):
        super().__init__(X, Y)
        self.clf = SVC(decision_function_shape='ovo')


class TheanoNeuralNetwork(DMCClassifier):
    def __init__(self, X: csr_matrix, Y: np.array):
        super().__init__(X, Y)
        input_layer, output_layer = self.X.shape[1], len(np.unique(Y))
        inp = tn.layers.base.Input(size=input_layer, sparse='csr')
        self.clf = tn.Classifier([inp, 100, 50, output_layer])

    def fit(self):
        self.clf.train((self.X, self.Y), algo='sgd', learning_rate=1e-4, momentum=0.9)
        return self


class BagEnsemble(DMCClassifier):
    classifier = None
    estimators = 10
    max_features = .5
    max_samples = .5

    def __init__(self, X: csr_matrix, Y: np.array):
        super().__init__(X, Y)
        self.clf = BaggingClassifier(self.classifier, n_estimators=self.estimators, n_jobs=8,
                                     max_samples=self.max_samples, max_features=self.max_features)


class TreeBag(BagEnsemble):
    classifier = DecisionTreeClassifier()


class BayesBag(BagEnsemble):
    classifier = BernoulliNB()


class SVMBag(DMCClassifier):
    classifier = None
    estimators = 10
    max_features = .5
    max_samples = .5

    def __init__(self, X: csr_matrix, Y: np.array):
        super().__init__(X, Y)
        self.X, self.Y = X.toarray(), Y
        self.classifier = SVC(decision_function_shape='ovo')
        self.clf = BaggingClassifier(self.classifier, n_estimators=self.estimators, n_jobs=8,
                                     max_samples=self.max_samples, max_features=self.max_features)

    def predict(self, X: csr_matrix):
        X = X.toarray()
        return self.clf.predict(X)


class AdaBoostEnsemble(DMCClassifier):
    classifier = None
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


class TensorFlowNeuralNetwork(DMCClassifier):
    steps = 2000
    learning_rate = 0.05
    hidden_units = [100, 100, 100]

    def __init__(self, X: np.array, Y: np.array):
        super().__init__(X.todense(), Y)  # TensorFlow/Skflow doesn't support sparse matrices
        n_input, n_classes = self.X.shape[1], len(np.unique(Y))
        self.clf = skflow.TensorFlowDNNClassifier(hidden_units=self.hidden_units, n_classes=n_classes, steps=self.steps,
                                                  learning_rate=self.learning_rate, verbose=0)

    def predict(self, X: csr_matrix):
        X = X.todense()  # TensorFlow/Skflow doesn't support sparse matrices
        return self.clf.predict(X)