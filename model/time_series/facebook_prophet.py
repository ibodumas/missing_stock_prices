##
from model import data_processing
import util
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from model.estimator import FBProphetEstimator


x_train = pd.DataFrame({"ds": data_processing.X_TRAIN})
y_train = pd.DataFrame({"y": data_processing.Y_TRAIN})
x_test = pd.DataFrame({"ds": data_processing.X_TEST})
y_test = data_processing.Y_TEST

x_y_train = pd.concat([x_train, y_train], axis=1)

changepoint_prior_scale = [0.4, 0.45, 0.5, 0.55, 0.6]
param_tune = {"param1": changepoint_prior_scale}
fb_proph = GridSearchCV(
    FBProphetEstimator(), param_tune, cv=10, scoring="neg_mean_squared_error"
)
fb_proph.fit(x_y_train, y_train)

fb_proph.best_params_

util.plot_actual_predicted(x_train, y_train, fb_proph.predict(x_train))

fb_y_pred = fb_proph.predict(x_test)

test_fb_mse = metrics.mean_squared_error(fb_y_pred, y_test)
