import os
import settings
import pandas as pd


def data_preprocessing(input_name: str, output_name: str):
    _data = pd.read_csv(
        os.path.join(settings.ROOT_DIR, "jupy_note", input_name), sep="\t"
    )
    _data.index = range(_data.shape[0])
    _data.columns = ["y"]
    _data["x"] = range(_data.shape[0])

    # position of the missing prices
    _pos_missing = _data.y.str.startswith("Missing")

    DATA_MISSING = _data[_pos_missing]

    # training data set
    DATA_TRAINING = _data[_pos_missing == False]

    ACTUAL_PRICES = pd.read_csv(
        os.path.join(settings.ROOT_DIR, "jupy_note", output_name), header=None
    )
    return DATA_TRAINING, DATA_MISSING, ACTUAL_PRICES
