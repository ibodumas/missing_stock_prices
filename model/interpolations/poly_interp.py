"""
Polynomial Interpolation:
This is a generalization of linear interpolation.
"""
##
from model import data_processing
import util
from model.estimator import PolyInterpEstimator
from sklearn.model_selection import GridSearchCV
from sklearn import metrics


x_train = data_processing.X_TRAIN
y_train = data_processing.Y_TRAIN
x_test = data_processing.X_TEST
y_test = data_processing.Y_TEST

degrees = [2, 3, 4, 5, 6, 7, 8, 9, 10]
param_tune = {"param1": degrees}
poly = GridSearchCV(
    PolyInterpEstimator(), param_tune, cv=util.CV, scoring=util.MERIC_SCORING
)
poly.fit(x_train, y_train)

util.plot_actual_predicted(x_train, y_train, poly.predict(x_train))

poly.best_params_

py_test_pred_y = poly.predict(x_test)

test_poly_err = metrics.mean_absolute_error(py_test_pred_y, y_test)
