"""
Spline Interpolation:
This uses low-degree polynomials, and selects polynomial pieces in order to optimally fit the data.
Univariate-Spline: It's a form of spline involving one variable.
"""
from model import util
from model.data_processing import data_preprocessing
from scipy.interpolate import UnivariateSpline as spline
from sklearn.model_selection import ParameterGrid
from sklearn import metrics
import pandas as pd


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


# Specify the data to be loaded.
# Change the input data to work through other a specific file.
input_file = "input00.txt"
output_file = "output00.txt"
DATA_TRAINING, DATA_MISSING, ACTUAL_PRICES = data_preprocessing(input_file, output_file)


# Find optimal Parameters via grid search
degree_spline = [2, 3, 4, 5]
smoothing_factor = [2, 3, 4, 5]

param_grid = ParameterGrid(
    dict(degree_spline=degree_spline, smoothing_factor=smoothing_factor)
)

grid_search_result = util.grid_search(
    DATA_TRAINING, model=spline_interpolation, param_grid=param_grid
)

grid_search_result.sort_values(['Train_MSE', 'Validation_MSE']).head()

util.heatmap(
    grid_search_result.Train_MSE,
    degree_spline,
    smoothing_factor,
    "Heatmap of Training MSE",
)

util.heatmap(
    grid_search_result.Validation_MSE,
    degree_spline,
    smoothing_factor,
    "Heatmap of Validation MSE",
)

# Predict the missing stock prices
# - while choosing the set of parameters needed for bias-variance tradeoff.
# - degree_spline=3 and smoothing_factor=3 will be used because it performs better across the 3 input files.

pred_missing_prices = spline_interpolation(
    DATA_TRAINING,
    degree_spline=3,
    smoothing_factor=3
)(DATA_MISSING.x)

pd.DataFrame(
    {
        "predicted missing prices": pred_missing_prices,
        "Actual": ACTUAL_PRICES.iloc[:, 0].tolist(),
    },
    index=DATA_MISSING.x.tolist()
).head()

# The Mean Square Error:
round(metrics.mean_squared_error(pred_missing_prices, ACTUAL_PRICES), 6)

# Visualization
# - Using an interactive plot

# Combine actual data with the predicted prices
x_y_predictd = DATA_MISSING.copy()
x_y_predictd.y = pred_missing_prices
all_with_pred = pd.concat([DATA_TRAINING, x_y_predictd])
all_with_pred = all_with_pred.sort_values('x')

# Combine actual data
all_x_y_actual = DATA_MISSING.copy()
all_x_y_actual.y = ACTUAL_PRICES.iloc[:, 0].tolist()
all_with_actual = pd.concat([DATA_TRAINING, all_x_y_actual])

# plot 1
fig = figure(
    title="Viewing the closeness of Actual vs. Predicting values of the Missing Stock Prices",
    width=950,
    height=700,
)

fig.title.text_font_size = "20px"
source = ColumnDataSource(
    data=dict(x=all_x_y_actual.x, y=all_x_y_actual.y, pointer=range(1, 21))
)
fig.scatter(x="x", y="y", color="navy", legend=["Actual"], source=source)
fig.scatter(x_y_predictd.x, x_y_predictd.y, color="firebrick", legend=["Predicted"])
fig.xaxis[0].axis_label = "Ordered Period (x)"
fig.yaxis[0].axis_label = "Stock Prices (y)"
labels = LabelSet(
    x="x",
    y="y",
    text="pointer",
    level="glyph",
    x_offset=5,
    y_offset=5,
    source=source,
    render_mode="canvas",
)

fig.add_layout(labels)
show(fig)
