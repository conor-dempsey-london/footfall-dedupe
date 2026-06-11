import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from scipy.optimize import curve_fit

from hs_models.utils import evaluate_dedupe_mape_threshold


class Hill2DRegressor(BaseEstimator, RegressorMixin):
    def __init__(
            self, 
            area_col='area_hex',
            dup_col='duplicated',
            p0=[200000, 15000, 100, 0.9, 0.8],
            bounds=(0.1, [np.inf, np.inf, np.inf, 10, 10])):
        self.p0 = p0
        self.bounds = bounds
        self.area_col   = area_col
        self.dup_col    = dup_col

    @staticmethod
    def _hill_2d_centered(coords, L, x0, a0, n, k):
        x, a = coords
        term = ((x / x0)**n) * ((a / a0)**k)
        return L * term / (1 + term)

    def fit(self, X, y):
        X = X[[self.dup_col, self.area_col]]
        # Validate inputs
        X, y = check_X_y(X, y)
        x, a = X[:, 0], X[:, 1]
        self.popt, _ = curve_fit(self._hill_2d_centered, (x, a), y, p0=self.p0, bounds=self.bounds)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = X[[self.dup_col, self.area_col]]
        X = check_array(X)
        x, a = X[:, 0], X[:, 1]
        y_pred = self._hill_2d_centered((x, a), *self.popt)
        return y_pred  

    def score(self, X, y):
        check_is_fitted(self)
        return evaluate_dedupe_mape_threshold(y, self.predict(X)) 
    