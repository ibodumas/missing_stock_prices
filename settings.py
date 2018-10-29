import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DIR_INPUT_0 = os.path.join(ROOT_DIR, "jupy_note", "input00.txt")
DIR_INPUT_1 = os.path.join(ROOT_DIR, "jupy_note", "input01.txt")
DIR_INPUT_2 = os.path.join(ROOT_DIR, "jupy_note", "input02.txt")
DIR_OUTPUT_0 = os.path.join(ROOT_DIR, "jupy_note", "output00.txt")
DIR_OUTPUT_1 = os.path.join(ROOT_DIR, "jupy_note", "output01.txt")
DIR_OUTPUT_2 = os.path.join(ROOT_DIR, "jupy_note", "output02.txt")


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
