import unittest
import util
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import numpy as np


class TestRidgePoly(unittest.TestCase):
    def test1(self):
        X = np.arange(1, 50).reshape(-1, 1)
        y = np.arange(1, 50)
        alphas = [0]
        degree = [2]
        param_tune = {"poly__degree": degree, "ridgeRregr__alpha": alphas}

        ridgeRregr = Ridge()
        poly = PolynomialFeatures()
        pipe_model = Pipeline(steps=[("poly", poly), ("ridgeRregr", ridgeRregr)])

        ridge_poly = GridSearchCV(
            pipe_model, param_tune, cv=10, scoring="neg_mean_squared_error"
        )
        ridge_poly.fit(X, y)

        x_test = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)

        pred = ridge_poly.predict(x_test)
        pred = [round(i, 1) for i in pred]

        util.plot_actual_predicted(X, y, ridge_poly.predict(X))

        self.assertEqual(pred, [1, 2, 3, 4, 5])
        self.assertEqual(ridge_poly.best_params_, {'poly__degree': 2, 'ridgeRregr__alpha': 0})
