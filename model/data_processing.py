import settings
import pandas as pd

_data = pd.read_csv(settings.DIR_INPUT_2, sep="\t")
_data.index = range(_data.shape[0])
_data.columns = ["y"]
_data["x"] = range(_data.shape[0])
_pos_missing = _data.y.str.startswith("Missing")  # position of the missing prices

DATA_MISSING = _data[_pos_missing]
DATA_TRAINING = _data[_pos_missing == False]  # training data set
del _data, _pos_missing

ACTUAL_PRICES = pd.read_csv(settings.DIR_OUTPUT_2, header=None)
