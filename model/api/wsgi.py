import ast
import util
from sklearn.externals import joblib
import numpy as np
import logging.config
import connexion


SAVED_MODEL = joblib.load(util.MODEL_PATH)


def predictor(x):
    try:
        x = ast.literal_eval(x)
        pred = SAVED_MODEL.predict(np.array(x))
        return {"prices": pred.tolist()}
    except (ValueError, SyntaxError):
        return 'Invalid request format. Sample input: 2, 4', 404, {'x-error': 'Invalid request'}
    except Exception:
        return 'Not Found', 404, {'x-error': 'not found'}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = connexion.App(__name__)
    app.add_api("api_spec.yml")
    application = app.app
    app.run(port=8080)