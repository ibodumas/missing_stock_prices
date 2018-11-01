"""
Spline Interpolation:
This uses low-degree polynomials, and selects polynomial pieces in order to optimally fit the data.
Univariate-Spline: It's a form of spline involving one variable.
"""
##
from model import data_processing
import util
from model.estimator import SplineEstimator
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.externals import joblib
import numpy as np


x_train = data_processing.X_TRAIN
y_train = data_processing.Y_TRAIN
x_test = data_processing.X_TEST
y_test = data_processing.Y_TEST

degree_splines = [2, 3, 4, 5]
smoothing_factors = np.arange(3, 13, 0.5)
param_tune = {"param1": degree_splines, "param2": smoothing_factors}
spline = GridSearchCV(
    SplineEstimator(), param_tune, cv=util.CV, scoring=util.MERIC_SCORING
)
spline.fit(x_train, y_train)

util.plot_actual_predicted(x_train, y_train, spline.predict(x_train))

spline.best_params_

sp_test_pred_y = spline.predict(x_test)

test_spline_err = metrics.mean_absolute_error(sp_test_pred_y, y_test)


if util.SAVE_MODEL:
    joblib.dump(spline, util.MODEL_PATH)
