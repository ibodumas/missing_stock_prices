import ast
import util
from sklearn.externals import joblib
import numpy as np


SAVED_MODEL = joblib.load(util.MODEL_PATH)


def predict(x):
    try:
        x = ast.literal_eval(x)
        pred = SAVED_MODEL.predict(np.array(x))
        return {"prices": pred.tolist()}
    except (ValueError, SyntaxError):
        return (
            "Invalid request format. Sample input: 2, 4",
            404,
            {"x-error": "Invalid request"},
        )
    except Exception:
        return "Not Found", 404, {"x-error": "not found"}
