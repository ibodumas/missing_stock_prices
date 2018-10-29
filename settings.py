import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    _path = [
        os.path.join(ROOT_DIR, "model"),
        os.path.join(ROOT_DIR, "model", "timeseries"),
        os.path.join(ROOT_DIR, "model", "reg_regression"),
        os.path.join(ROOT_DIR, "model", "interpolations"),
    ]

    for pth in _path:
        if pth not in sys.path:
            sys.path.append(pth)
