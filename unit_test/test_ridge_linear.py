import unittest
import util
from sklearn.linear_model import Ridge as ridge_linear_regr
from sklearn.model_selection import GridSearchCV
import numpy as np


class TestRidgeLinear(unittest.TestCase):
    def test1(self):
        X = np.arange(1, 50).reshape(-1, 1)
        y = np.arange(1, 50)
        alpha = [0, 10]
        param_tune = {"alpha": alpha}

        ridge_linear = GridSearchCV(
            ridge_linear_regr(), param_tune, cv=2, scoring="neg_mean_squared_error"
        )

        ridge_linear.fit(X, y)

        x_test = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)

        pred = ridge_linear.predict(x_test)
        pred = [round(i, 1) for i in pred]

        util.plot_actual_predicted(X, y, ridge_linear.predict(X))

        self.assertEqual(pred, [1, 2, 3, 4, 5])
        self.assertEqual(ridge_linear.best_params_, {"alpha": 0})

    def test2(self):
        X = np.arange(50, 100).reshape(-1, 1)
        y = np.arange(50, 100)
        alpha = [0, 10]
        param_tune = {"alpha": alpha}

        ridge_linear = GridSearchCV(
            ridge_linear_regr(), param_tune, cv=2, scoring="neg_mean_squared_error"
        )

        ridge_linear.fit(X, y)

        x_test = np.array([55, 60, 65, 70]).reshape(-1, 1)

        pred = ridge_linear.predict(x_test)
        pred = [round(i, 1) for i in pred]

        util.plot_actual_predicted(X, y, ridge_linear.predict(X))

        self.assertEqual(pred, [55, 60, 65, 70])
        self.assertEqual(ridge_linear.best_params_, {"alpha": 0})
