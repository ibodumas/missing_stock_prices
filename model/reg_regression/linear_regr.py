"""
Implementation of a grid search cross-validation linear ridge regression.
Linear least squares with l2 regularization.
Minimizes the objective function:
||y - Xw||^2_2 + alpha * ||w||^2_2
"""
###
from model import data_processing
import util
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.linear_model import Ridge
import numpy as np


x_train = np.array(data_processing.X_TRAIN).reshape(-1, 1)
y_train = data_processing.Y_TRAIN
x_test = np.array(data_processing.X_TEST).reshape(-1, 1)
y_test = data_processing.Y_TEST

alphas = np.logspace(0, 10, 11)
param_tune = {"alpha": alphas}

ridge_linear = GridSearchCV(Ridge(), param_tune, cv=util.CV, scoring=util.MERIC_SCORING)

ridge_linear.fit(x_train, y_train)

util.plot_actual_predicted(x_train, y_train, ridge_linear.predict(x_train))

ridge_linear.best_params_

rl_test_pred_y = ridge_linear.predict(x_test)

test_rl_err = metrics.mean_absolute_error(rl_test_pred_y, y_test)
