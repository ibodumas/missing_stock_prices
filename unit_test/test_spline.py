import unittest
import util
from model.interpolations.spline_interp import SplineEstimator
from sklearn.model_selection import GridSearchCV


class TestSpline(unittest.TestCase):
    def test1(self):
        X = list(range(1, 50))
        y = list(range(1, 50))
        degree_spline = [2]
        smoothing_factor = [2]
        param_tune = {"param1": degree_spline, "param2": smoothing_factor}
        spline = GridSearchCV(SplineEstimator(), param_tune, cv=2)
        spline.fit(X, y)

        pred = spline.predict([1, 2, 3, 4, 5])
        pred = [round(i, 1) for i in pred]

        util.plot_actual_predicted(X, y, spline.predict(X))

        self.assertEqual(pred, [1, 2, 3, 4, 5])
        self.assertEqual(spline.best_params_, {"param1": 2, "param2": 2})

    def test2(self):
        X = list(range(51, 100))
        y = list(range(51, 100))
        degree_spline = [2]
        smoothing_factor = [2]
        param_tune = {"param1": degree_spline, "param2": smoothing_factor}
        spline = GridSearchCV(SplineEstimator(), param_tune, cv=2)
        spline.fit(X, y)

        pred = spline.predict([55, 60, 65, 70])
        pred = [round(i, 1) for i in pred]

        util.plot_actual_predicted(X, y, spline.predict(X))

        self.assertEqual(pred, [55, 60, 65, 70])
        self.assertEqual(spline.best_params_, {"param1": 2, "param2": 2})
