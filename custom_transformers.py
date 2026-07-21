import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
def winsorize(X, lower=0.25, upper=0.75):
    Q1 = np.percentile(X, lower * 100, axis=0)
    Q3 = np.percentile(X, upper * 100, axis=0)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return np.clip(X, lower_bound, upper_bound)

class NamedWinsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lower=0.05, upper=0.95):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        self.feature_names_ = list(X.columns)
        return self

    def transform(self, X):
        return pd.DataFrame(winsorize(X, self.lower, self.upper),
                            columns=self.feature_names_, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_)
