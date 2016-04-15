import pandas as pd
import numpy as np


class DMCClassifier:
    def __init__(self, df: pd.DataFrame):
        self.X, self.Y = self.feature_matrix(df)
        self.classifier = None

    def __call__(self, df: pd.DataFrame) -> np.array:
        self.fit()
        return self.predict(df)

    def fit(self):
        raise NotImplementedError

    def predict(self, df: pd.DataFrame) -> np.array:
        raise NotImplementedError

    @classmethod
    def feature_matrix(cls, df: pd.DataFrame) -> (np.array, np.array):
        return np.array(), np.array()
