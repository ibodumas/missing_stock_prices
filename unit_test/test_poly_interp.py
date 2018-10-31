import unittest
import util
from model.interpolations.poly_interp import PolyInterpEstimator
from sklearn.model_selection import GridSearchCV


class TestSpline(unittest.TestCase):
    def test1(self):
        X = list(range(1, 50))
        y = list(range(1, 50))
        degree = [2]
        param_tune = {"param1": degree}
        poly = GridSearchCV(
            PolyInterpEstimator(), param_tune, cv=2, scoring="neg_mean_squared_error"
        )
        poly.fit(X, y)

        pred = poly.predict([11, 25, 5, 40])
        pred = [round(i, 1) for i in pred]

        util.plot_actual_predicted(X, y, poly.predict(X))

        self.assertEqual(pred, [11, 25, 5, 40])
        self.assertEqual(poly.best_params_, {"param1": 2})

    def test2(self):
        X = list(range(51, 100))
        y = list(range(51, 100))
        degree = [2]
        param_tune = {"param1": degree}
        poly = GridSearchCV(
            PolyInterpEstimator(), param_tune, cv=2, scoring="neg_mean_squared_error"
        )
        poly.fit(X, y)

        pred = poly.predict([99, 81, 70, 56])
        pred = [round(i, 1) for i in pred]

        util.plot_actual_predicted(X, y, poly.predict(X))

        self.assertEqual(pred, [99, 81, 70, 56])
        self.assertEqual(poly.best_params_, {"param1": 2})
