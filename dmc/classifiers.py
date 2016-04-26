import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import theanets as tn
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, \
    BaggingClassifier, AdaBoostClassifier

from time import time
from operator import itemgetter
from scipy.stats import randint as sp_randint

from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier


class DMCClassifier:
    clf = None

    def __init__(self, X: csr_matrix, Y: np.array):
        assert len(Y) == X.shape[0]
        self.X = X
        self.Y = Y

    def __call__(self, df: pd.DataFrame) -> np.array:
        self.estimate_parameters_with_grid_search_cv()
        self.fit()
        return self.predict(df)

    def estimate_parameters_with_random_search(self):
        # specify parameters and distributions to sample from
        param_dist = {"max_depth": [3, None],
                      "max_features": sp_randint(1, 11),
                      "min_samples_split": sp_randint(1, 11),
                      "min_samples_leaf": sp_randint(1, 11),
                      "criterion": ["gini", "entropy"]}

        # run randomized search
        n_iter_search = 100
        random_search = RandomizedSearchCV(self.clf, param_distributions=param_dist,
                                           n_iter=n_iter_search)

        start = time()
        random_search.fit(self.X, self.Y)
        print("RandomizedSearchCV took %.2f seconds for %d candidates"
              " parameter settings." % ((time() - start), n_iter_search))
        self.report(random_search.grid_scores_)


    def estimate_parameters_with_grid_search_cv(self):
        # use a full grid over all parameters
        param_grid = {"max_depth": [None, 10, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50],
                      "max_features": [None, 1, 10, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50],
                      "min_samples_split": [1, 10, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50],
                      "min_samples_leaf": [1, 10, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50],
                      "criterion": ["gini", "entropy"]}

        # run grid search
        grid_search = GridSearchCV(self.clf, param_grid=param_grid)
        start = time()
        grid_search.fit(self.X, self.Y)

        print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
              % (time() - start, len(grid_search.grid_scores_)))
        self.report(grid_search.grid_scores_)

    def fit(self):
        self.clf.fit(self.X, self.Y)
        return self


    def predict(self, X: csr_matrix) -> np.array:
        return self.clf.predict(X)

    # Utility function to report best scores
    def report(self, grid_scores, n_top=3):
        top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
        for i, score in enumerate(top_scores):
            print("Model with rank: {0}".format(i + 1))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  score.mean_validation_score,
                  np.std(score.cv_validation_scores)))
            print("Parameters: {0}".format(score.parameters))
            print("")


class DecisionTree(DMCClassifier):
    clf = DecisionTreeClassifier()


class Forest(DMCClassifier):
    def __init__(self, X: csr_matrix, Y: np.array):
        super().__init__(X, Y)
        self.clf = RandomForestClassifier(n_estimators=100, n_jobs=8)


class NaiveBayes(DMCClassifier):
    clf = BernoulliNB()


class SVM(DMCClassifier):
    def __init__(self, X: csr_matrix, Y: np.array):
        super().__init__(X, Y)
        self.clf = SVC(decision_function_shape='ovo')


class NeuralNetwork(DMCClassifier):
    def __init__(self, X: csr_matrix, Y: np.array):
        super().__init__(X, Y)
        input_layer, output_layer = self.X.shape[1], len(np.unique(Y))
        inp = tn.layers.base.Input(size=input_layer, sparse='csr')
        self.clf = tn.Classifier([inp, 800, 600, 300, 150, 20, output_layer])

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
