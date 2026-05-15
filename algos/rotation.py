"""Standalone rotation predictor — delegates to pre_annotate."""
import numpy as np
from . import pre_annotate


def predict_yaw(points):
    return pre_annotate.predict_yaw(points)


if __name__ == "__main__":
    # warmup
    result = predict_yaw(np.random.random([1000, 3]))
    print(result)
