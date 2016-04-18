import pandas as pd
import numpy as np
import theanets as tn
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


class DMCClassifier:
    classifier = None

    def __init__(self, df: pd.DataFrame):
        self.X = self.feature_matrix(df)
        self.Y = self.label_vector(df)
        self.clf = self.classifier() if self.classifier else None

    def __call__(self, df: pd.DataFrame) -> np.array:
        self.fit()
        return self.predict(df)

    def fit(self):
        self.clf.fit(self.X, self.Y)

    def predict(self, df: pd.DataFrame) -> np.array:
        Y = self.feature_matrix(df)
        return self.clf.predict(Y)

    @classmethod
    def feature_matrix(cls, df: pd.DataFrame) -> np.array:
        cols = ['colorCode', 'quantity', 'price', 'rrp', 'voucherAmount',
                'productPrice', 'totalSavings',
                'orderYear', 'orderMonth', 'orderDay', 'orderWeekDay', 'orderDayOfYear',
                'orderWeek', 'orderWeekOfYear', 'orderQuarter', 'orderSeason',
                'surplusArticleQuantity', 'surplusArticleSizeQuantity',
                'surplusArticleColorQuantity', '0paymentMethod', '1paymentMethod',
                '2paymentMethod', '3paymentMethod', '4paymentMethod', '5paymentMethod',
                '6paymentMethod', '7paymentMethod', '8paymentMethod', '0sizeCode',
                '1sizeCode', '2sizeCode', '3sizeCode', '4sizeCode', '5sizeCode',
                '6sizeCode', '7sizeCode', '8sizeCode', '9sizeCode', '10sizeCode',
                '11sizeCode', '12sizeCode', '13sizeCode', '14sizeCode', '15sizeCode',
                '16sizeCode', '17sizeCode', '18sizeCode', '19sizeCode', '20sizeCode',
                '21sizeCode', '22sizeCode', '23sizeCode', '24sizeCode', '25sizeCode',
                '26sizeCode', '27sizeCode', '0deviceID', '1deviceID', '2deviceID',
                '3deviceID', '4deviceID', '0productGroup', '1productGroup',
                '2productGroup', '3productGroup', '4productGroup', '5productGroup',
                '6productGroup', '7productGroup', '8productGroup', '9productGroup',
                '10productGroup', '11productGroup', '12productGroup', '13productGroup',
                '14productGroup', '15productGroup', '16productGroup']
        return df.as_matrix(columns=cols).astype(np.float32)

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
