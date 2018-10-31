"""
Implementation of a grid search cross-validation polynomial ridge regression.
"""
###
from model import data_processing
import util
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline


x_train = np.array(data_processing.X_TRAIN).reshape(-1, 1)
y_train = data_processing.Y_TRAIN
x_test = np.array(data_processing.X_TEST).reshape(-1, 1)
y_test = data_processing.Y_TEST


alphas = np.logspace(0, 10, 11)
degree = [2, 3, 4, 5, 6, 7, 8]
param_tune = {"poly__degree": degree, "ridgeRregr__alpha": alphas}

ridgeRregr = Ridge()
poly = PolynomialFeatures()
pipe_model = Pipeline(steps=[("poly", poly), ("ridgeRregr", ridgeRregr)])

ridge_poly = GridSearchCV(
    pipe_model, param_tune, cv=10, scoring="neg_mean_squared_error"
)
ridge_poly.fit(x_train, y_train)

util.plot_actual_predicted(x_train, y_train, ridge_poly.predict(x_train))

ridge_poly.best_params_

rp_test_pred_y = ridge_poly.predict(x_test)

test_rp_mse = metrics.mean_squared_error(rp_test_pred_y, y_test)
