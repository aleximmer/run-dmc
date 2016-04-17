import pandas as pd
import numpy as np


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
        Y = self.label_vector(df)
        return self.clf.predict(Y)

    @classmethod
    def feature_matrix(cls, df: pd.DataFrame) -> np.array:
        return df.as_matrix()

    @classmethod
    def label_vector(cls, df: pd.DataFrame) -> np.array:
        return df.as_matrix()
