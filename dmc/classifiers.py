import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import theanets as tn
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, \
    BaggingClassifier, AdaBoostClassifier

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
        self.param_dist = { "max_features": sp_randint(1, self.X.shape[1]) }

    def __call__(self, df: pd.DataFrame) -> np.array:
        print(self.clf.get_params().keys())
        self.estimate_parameters_with_random_search()
        self.fit()
        return self.predict(df)


    def report(self, grid_scores, n_top=3):
        top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
        for i, score in enumerate(top_scores):
            print("Model with rank: {0}".format(i + 1))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  score.mean_validation_score,
                  np.std(score.cv_validation_scores)))
            print("Parameters: {0}".format(score.parameters))
            print("")


    def estimate_parameters_with_random_search(self):
        random_search = RandomizedSearchCV(self.clf, param_distributions=self.param_dist,
                                           n_iter=100)
        random_search.fit(self.X, self.Y)
        self.report(random_search.grid_scores_)


    def estimate_parameters_with_grid_search_cv(self):
        grid_search = GridSearchCV(self.clf, param_grid=self.param_dist)
        grid_search.fit(self.X, self.Y)
        self.report(grid_search.grid_scores_)


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
        self.param_dist = {'max_depth': sp_randint(1,200), 'min_samples_leaf': 100, "max_features": sp_randint(1, 2400), 'criterion': ['entropy', 'gini']}
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
        self.param_dist = {'base_estimator__max_features': sp_randint(1, self.X.shape[1])}
        self.clf = BaggingClassifier(self.classifier, n_estimators=self.estimators, n_jobs=8,
                                     max_samples=self.max_samples, max_features=self.max_features)


class TreeBag(BagEnsemble):
    classifier = DecisionTreeClassifier()

    def __init__(self, X: np.array, Y: np.array):
        super().__init__(X, Y)
        param_dist = {'base_estimator__max_features': sp_randint(1, self.X.shape[1])}



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
        self.param_dist={'base_estimator__max_features': sp_randint(1, X.shape[1])}
        self.clf = AdaBoostClassifier(self.classifier, n_estimators=self.estimators,
                                      learning_rate=self.learning_rate, algorithm=self.algorithm)


class AdaTree(AdaBoostEnsemble):
    classifier = DecisionTreeClassifier()

    def __init__(self, X: np.array, Y: np.array):
        super().__init__(X, Y)
        param_dist = {'max_features': sp_randint(1, self.X.shape[1])}


class AdaBayes(AdaBoostEnsemble):
    classifier = BernoulliNB()


class AdaSVM(AdaBoostEnsemble):
    algorithm = 'SAMME'

    def __init__(self, X: np.array, Y: np.array):
        self.classifier = SVC(decision_function_shape='ovo')
        super().__init__(X, Y)
