"""
This houses all general purpose objects.
"""
import os
import matplotlib.pyplot as plt


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ACTUAL_COLOR = "navy"
PREDICTED_COLOR = "firebrick"
SAVE_MODEL = True
MODEL_PATH = os.path.join(ROOT_DIR, "spline.joblib")
MERIC_SCORING = "neg_mean_absolute_error"
CV = 10


def plot_actual_predicted(X, y_actual, y_pred):
    """
    :param X: List, Array or Series signifying the x-axis
    :param y_actual: List, Array or Series of the actual
    :param y_pred: List, Array or Series of the predicted
    :return: Renders a plot.
    """
    plt.figure(figsize=(15, 5))
    plt.scatter(X, y_actual, marker="o", label="Actual Price", color=ACTUAL_COLOR, s=30)
    plt.plot(X, y_pred, "g-", lw=2.5, label="Fitted Curve", color=PREDICTED_COLOR)
    plt.title("Plot of Actual Points vs. Fitted Curve")
    plt.xlabel("Ordered Period (x)")
    plt.ylabel("Stock Prices (y)")
    plt.legend(loc="upper left")
    plt.show()


def visualization_dist(x_test, y_test, y_pred):
    x_test = x_test.tolist()
    y_test = y_test.iloc[:, 0].tolist()

    plt.figure(figsize=(15, 5))
    plt.scatter(
        x_test, y_test, marker="o", label="Actual Price", color=ACTUAL_COLOR, s=30
    )
    plt.scatter(
        x_test, y_pred, marker="o", label="Predicted Price", color=PREDICTED_COLOR
    )
    plt.title(
        "Viewing the closeness of Actual vs. Predicting values of the Missing Stock Prices"
    )
    for i in range(1, len(x_test) + 1):
        plt.annotate(i, (x_test[i - 1], y_test[i - 1]), fontsize=15, color=ACTUAL_COLOR)
        plt.annotate(i, (x_test[i - 1], y_pred[i - 1]), fontsize=8)

    plt.xlabel("Ordered Period (x)")
    plt.ylabel("Stock Prices (y)")
    plt.legend(loc="lower left")
    plt.show()
