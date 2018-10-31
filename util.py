"""
This houses all general purpose objects.
"""
import os
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from bokeh.plotting import figure, show
from bokeh.models import LabelSet, ColumnDataSource
from bokeh.layouts import row
from bokeh.io import output_file

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ACTUAL_COLOR = "navy"
PREDICTED_COLOR = "firebrick"
SAVE_MODEL = False
MODEL_PATH = os.path.join(ROOT_DIR, "spline.joblib")


def grid_search(data_training, model, param_grid):
    """
    :param data_training:
    :param model: the function that defines the model.
    :param param_grid: Sklearn ParameterGrid
    :return: sorted Pandas DataFrame of the gridsearch and the corresponding MSE.
    """

    x_train, x_val, y_train, y_val = train_test_split(
        data_training.x, data_training.y, test_size=0.1, random_state=42
    )

    # sort the data because the x value of spline must be increasing.
    train_data = pd.DataFrame({"x": x_train, "y": y_train}).sort_values("x")
    val_data = pd.DataFrame({"x": x_val, "y": y_val}).sort_values("x")

    param_grid_result = pd.DataFrame.from_dict(list(param_grid))
    val_MSE = []
    tr_MES = []

    for ind, grid in enumerate(param_grid):
        trained_model = model(train_data, **grid)
        y_predicted = trained_model(val_data.x)
        val_error = metrics.mean_squared_error(y_predicted, val_data.y)
        val_MSE.append(round(val_error, 6))

        training_pred = trained_model(train_data.x)
        tr_error = metrics.mean_squared_error(training_pred, train_data.y)
        tr_MES.append(round(tr_error, 6))

    param_grid_result["Train_MSE"] = tr_MES
    param_grid_result["Validation_MSE"] = val_MSE
    return param_grid_result


def heatmap(metric, x_axis, y_axis, title):
    """
    :param metric: the computed value - e.g MSE, RMSE
    :param x_axis: values on the x-axis
    :param y_axis: values on the y-axis
    :param title: Plot title
    :return: None - Renders the plot.
    """
    plt.figure(figsize=(4, 4))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(
        np.reshape(list(metric), (4, 4)), interpolation="nearest", cmap=plt.cm.hot
    )
    plt.xlabel("Degree of Spline")
    plt.ylabel("Smoothing Factor")
    plt.colorbar()
    plt.xticks(np.arange(4), x_axis, rotation=45)
    plt.yticks(np.arange(4), y_axis)
    plt.title(title)
    plt.show()


def plot_actual_predicted(X, y_actual, y_pred):
    plt.scatter(
        X, y_actual, marker="o", label="Actual points", color=ACTUAL_COLOR, s=30
    )
    plt.plot(X, y_pred, "g-", lw=2.5, label="Fitted Curve", color=PREDICTED_COLOR)
    plt.title("Plot of Actual Points vs. Fitted Curve")
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    plt.legend(loc="upper left")
    plt.show()


def visualizations(x_y_actual, x_y_predict, all_with_actual, all_with_pred):
    """
    plot 1: Viewing the closeness of Actual vs. Predicting values of the Missing Stock Prices

    plot 2: side-by-side
     - All data with the actual values of the missing prices
     - All data with the estimated values of the missing prices

    plot 3: Positioning the side-by-side plots above in a single view

    :param x_y_actual: (x, y) Pandas DataFrame of the actual missing stock prices
    :param x_y_predict: (x, y) Pandas DataFrame of the predicted missing stock prices
    :param all_with_actual: (x, y) Pandas DataFrame of all data with the actual missing stock prices
    :param all_with_pred: (x, y) Pandas DataFrame of all data with the predicted missing stock prices
    :return: None. Render the plots.
    """

    # plot 1
    output_file("plot1.html")
    fig = figure(
        title="Viewing the closeness of Actual vs. Predicting values of the Missing Stock Prices",
        width=950,
        height=700,
    )

    fig.title.text_font_size = "20px"
    source = ColumnDataSource(
        data=dict(x=x_y_actual.x, y=x_y_actual.y, pointer=range(1, 21))
    )
    fig.scatter(x="x", y="y", color="navy", legend=["Actual"], source=source)
    fig.scatter(x_y_predict.x, x_y_predict.y, color="firebrick", legend=["Predicted"])
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

    # plot 2
    output_file("plot2.html")
    fig_act = figure(
        plot_width=475,
        plot_height=300,
        title="All data with the actual values of the missing prices",
    )
    fig_act.scatter(all_with_actual.x, all_with_actual.y, color="navy")
    fig_act.xaxis[0].axis_label = "Ordered Period (x)"
    fig_act.yaxis[0].axis_label = "Stock Prices (y)"

    fig_pred = figure(
        plot_width=475,
        plot_height=300,
        title="All data with the estimated values of the missing prices",
    )
    fig_pred.scatter(all_with_pred.x, all_with_pred.y, color="firebrick")
    fig_pred.xaxis[0].axis_label = "Ordered Period (x)"
    fig_pred.yaxis[0].axis_label = "Stock Prices (y)"

    show(row(fig_act, fig_pred))

    # plot 3
    output_file("plot3.html")
    fig = figure(
        title="Positioning the side-by-side plots above in a single view", width=900
    )
    fig.scatter(
        all_with_pred.x, all_with_pred.y, color="firebrick", legend=["Predicted"]
    )
    fig.scatter(all_with_actual.x, all_with_actual.y, color="navy", legend=["Actual"])
    fig.xaxis[0].axis_label = "Ordered Period (x)"
    fig.yaxis[0].axis_label = "Stock Prices (y)"
    show(fig)
