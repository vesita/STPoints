import os
import warnings
import numpy as np
import glob
import math
import json

from . import util

util.config_gpu()

RESAMPLE_NUM = 10
NUM_POINT = 512
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "deep_annotation_inference.h5")

_rotation_model = None

def _get_model():
    global _rotation_model
    if _rotation_model is None:
        import tensorflow as tf
        if not os.path.isfile(MODEL_PATH):
            warnings.warn(f"auto-annotate model not found: {MODEL_PATH}")
            return None
        _rotation_model = tf.keras.models.load_model(MODEL_PATH)
        _rotation_model.summary()
    return _rotation_model


def sample_one_obj(points, num):
    if points.shape[0] < NUM_POINT:
        return np.concatenate([points, np.zeros((NUM_POINT - points.shape[0], 3), dtype=np.float32)], axis=0)
    else:
        idx = np.arange(points.shape[0])
        np.random.shuffle(idx)
        return points[idx[0:num]]


def predict_yaw(points):
    model = _get_model()
    if model is None:
        return [0, 0, 0]
    points = np.array(points).reshape((-1, 3))
    input_data = np.stack([sample_one_obj(points, NUM_POINT) for _ in range(RESAMPLE_NUM)], axis=0)
    pred_val = model.predict(input_data, verbose=0)
    pred_cls = np.argmax(pred_val, axis=-1)

    ret = (pred_cls[0] * 3 + 1.5) * np.pi / 180.
    ret = [0, 0, ret]
    return ret


def annotate_file(input, output=None):
    """Stub: auto-annotate not available in this build (pipeline removed)."""
    warnings.warn("auto_annotate is disabled — the clustering pipeline has been removed")
    return []
