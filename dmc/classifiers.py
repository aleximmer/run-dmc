import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier


class DMCClassifier:
    classifier = None

    def __init__(self, df: pd.DataFrame):
        self.X = self.feature_matrix(df)
        self.Y = self.label_vector(df)
        self.clf = self.classifier()

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
        return df.as_matrix()

    @classmethod
    def label_vector(cls, df: pd.DataFrame) -> np.array:
        return df.as_matrix(columns=['returnQuantity'])


class DecisionTree(DMCClassifier):
    classifier = DecisionTreeClassifier

    @classmethod
    def feature_matrix(cls, df: pd.DataFrame) -> np.array:
        return np.nan_to_num(df.as_matrix(
            columns=['quantity', 'deviceID', 'voucherID',
                     'colorCode', 'rrp', 'productGroup']))
