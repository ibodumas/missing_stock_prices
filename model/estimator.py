"""
Base class for Customized estimator, to be used for grid search cross-validation.
"""

from abc import abstractmethod
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.interpolate import UnivariateSpline as spline
from numpy import polyfit, poly1d
import fbprophet


class BaseCustomEstimator(BaseEstimator, RegressorMixin):
    def __init__(self, param1=None, param2=None, param3=None):
        """
        :param (param1, param2, param2): are hyperparameters -
                there values depend on the underlying model. The default value is 0.05.
        """
        self.param1 = param1
        self.param2 = param2
        self.param3 = param3

    @abstractmethod
    def fit(self):
        raise NotImplementedError("abstractmethod fit is yet to be implemented")

    def predict(self, X):
        is_trained = getattr(self, "trained_model", False)
        if is_trained:
            return self.trained_model(X)
        else:
            raise RuntimeError("the model has not been trained; call fit.")


class SplineEstimator(BaseCustomEstimator):
    def __init__(self, param1=None, param2=None):
        """
        :param param1: degree_spline(int) - Degree of the smoothing spline. Must be <= 5.
                        Default is k=3, a cubic spline.
        :param param2: smoothing_factor(float) - Smoothing factor used to choose the
                        no. of knots. No. of knots will be increased until the smoothing
                        condition is satisfied.
        """
        super().__init__(param1, param2)

    def fit(self, X, y):
        self.trained_model = spline(X, y, k=self.param1, s=self.param2)
        return self


class PolyInterpEstimator(BaseCustomEstimator):
    def __init__(self, param1=None):
        """
        :param param1: degree(int) - degree of the polynomial
        """
        super().__init__(param1)

    def fit(self, X, y):
        self.trained_model = poly1d(polyfit(X, y, deg=self.param1))
        return self


class FBProphetEstimator(BaseCustomEstimator):
    def __init__(self, param1=None):
        """
        :param param1: changepoint_prior_scale(int) - for controlling how sensitive the trend is to changes
        """
        super().__init__(param1)

    def fit(self, X_y, y):
        model = fbprophet.Prophet(changepoint_prior_scale=self.param1)
        self.trained_model = model.fit(df=X_y)
        return self

    def predict(self, X):
        is_trained = getattr(self, "trained_model", False)
        if is_trained:
            X = pd.DataFrame({"ds": X.ds})
            return self.trained_model.predict(X).yhat
        else:
            raise RuntimeError("the model has not been trained; call fit.")
