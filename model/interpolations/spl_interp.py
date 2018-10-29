"""
Spline Interpolation:
This uses low-degree polynomials, and selects polynomial pieces in order to optimally fit the data.
Univariate-Spline: It's a form of spline involving one variable.
"""
from model import util
from model import data_processing
from scipy.interpolate import UnivariateSpline as spline
from sklearn.model_selection import ParameterGrid
from sklearn import metrics


def spline_interpolation(train, **kwargs):
    """
    :param train: Pandas DataFrame - Stock prices with some missing values. dim = (n, 2)
    :param kwargs: dict with keys degree_spline and smoothing_factor
                1. degree_spline: int - Degree of the smoothing spline. Must be <= 5.
                Default is k=3, a cubic spline.
                2. smoothing_factor: float - Smoothing factor used to choose the
                no. of knots. No. of knots will be increased until the smoothing
                condition is satisfied.
    :return: A function that holds the trained model.
    """

    spline_model = spline(
        train.x, train.y, k=kwargs["degree_spline"], s=kwargs["smoothing_factor"]
    )

    return spline_model


param_grid = ParameterGrid(
    dict(degree_spline=[1, 2, 3], smoothing_factor=[2, 3, 4, 5, 6])
)

grid_search_result = util.grid_search(
    data_processing.DATA_TRAINING, model=spline_interpolation, param_grid=param_grid
)
grid_search_result

# get the optimal parameters
best_degree_spline = grid_search_result.degree_spline[0]
best_smoothing_factor = grid_search_result.smoothing_factor[0]

# use the optimal parameters to train the model on the entire data (excluding missing price)
# And predict the missing prices
pred_missing_prices = spline_interpolation(
    data_processing.DATA_TRAINING,
    data_processing.DATA_MISSING.x,
    degree_spline=3,
    smoothing_factor=2,
)

mse_pred_missing = metrics.mean_squared_error(
    pred_missing_prices, data_processing.ACTUAL_PRICES
)
