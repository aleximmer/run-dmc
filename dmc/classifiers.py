import numpy as np
from scipy.sparse import csr_matrix
import theanets as tn
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, \
    BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier


class DMCClassifier:
    clf = None

    def __init__(self, X: csr_matrix, Y: np.array):
        assert len(Y) == X.shape[0]
        self.X = X
        self.Y = Y

    def __call__(self, X: csr_matrix) -> np.array:
        self.fit()
        return self.predict(X)

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


class NeuralNetwork(DMCClassifier):
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
    estimators = 20
    max_features = .9
    max_samples = .9

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
    estimators = 20
    max_features = .9
    max_samples = .9

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
    estimators = 200
    learning_rate = 1
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


class GradBoost(DMCClassifier):
    estimators = 2000
    learning_rate=1
    max_depth = 1
    max_features = 0.97

    def __init__(self, X: np.array, Y: np.array):
        super().__init__(X, Y)
        self.clf = GradientBoostingClassifier(n_estimators=self.estimators,
                                      learning_rate=self.learning_rate, max_depth=self.max_depth, max_features=self.max_features)

    def predict(self, X: csr_matrix) -> np.array:
        return self.clf.predict(X.toarray())

