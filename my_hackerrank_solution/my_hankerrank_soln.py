import pandas as pd
from scipy.interpolate import UnivariateSpline as spline

data = pd.read_csv("input", sep="\t")
data.index = range(data.shape[0])
data.columns = ["y"]
data["x"] = range(data.shape[0])

# position of the missing price
pos_missing = data.y.str.startswith("Missing")
data_missing = data[pos_missing]
data_training = data[pos_missing == False]

spline_model = spline(data_training.x, data_training.y, k=3, s=3)
pred_price = spline_model(data_missing.x)
print(*pred_price, sep="\n")
