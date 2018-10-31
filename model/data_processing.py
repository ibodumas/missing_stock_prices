import os
import util
import pandas as pd


def data_preprocessing(input_name: str, output_name: str):
    data = pd.read_csv(os.path.join(util.ROOT_DIR, "jupy_note", input_name), sep="\t")
    data.index = range(data.shape[0])
    data.columns = ["y"]
    data["x"] = range(data.shape[0])

    # position of the missing prices
    pos_missing = data.y.str.startswith("Missing")

    data_missing = data[pos_missing]

    # training data set
    data_training = data[pos_missing == False]

    actual_prices = pd.read_csv(
        os.path.join(util.ROOT_DIR, "jupy_note", output_name), header=None
    )

    data_training = data_training.astype("float")
    actual_prices = actual_prices.astype("float")

    return data_training.x, data_training.y, data_missing.x, actual_prices


input_file = "input00.txt"
output_file = "output00.txt"
X_TRAIN, Y_TRAIN, X_TEST, Y_TEST = data_preprocessing(input_file, output_file)
