"""
This houses all general purpose objects.
"""
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd


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

