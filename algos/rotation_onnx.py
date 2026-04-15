"""
Rotation prediction using ONNX runtime (no TensorFlow dependency).

Drop-in replacement for rotation.py. Uses rotation_model.onnx converted from
the original deep_annotation_inference.h5 (PointNet classifier, 120 rotation bins).

Requires: onnxruntime, numpy
"""

import os
import numpy as np
import onnxruntime as ort

RESAMPLE_NUM = 10
NUM_POINT = 512

_model_dir = os.path.dirname(os.path.abspath(__file__))
_onnx_path = os.path.join(_model_dir, "models", "rotation_model.onnx")

_session = ort.InferenceSession(_onnx_path)
_input_name = _session.get_inputs()[0].name


def _sample_one_obj(points, num):
    if points.shape[0] < NUM_POINT:
        pad = np.zeros((NUM_POINT - points.shape[0], 3), dtype=np.float32)
        return np.concatenate([points, pad], axis=0)
    else:
        idx = np.arange(points.shape[0])
        np.random.shuffle(idx)
        return points[idx[:num]]


def predict(points):
    points = np.array(points, dtype=np.float32).reshape((-1, 3))
    input_data = np.stack(
        [_sample_one_obj(points, NUM_POINT) for _ in range(RESAMPLE_NUM)],
        axis=0
    )
    pred_val = _session.run(None, {_input_name: input_data})[0]
    pred_cls = np.argmax(pred_val, axis=-1)

    ret = (pred_cls[0] * 3 + 1.5) * np.pi / 180.0
    return [0, 0, float(ret)]


# Warmup
predict(np.random.random((1000, 3)))


if __name__ == "__main__":
    result = predict(np.random.random((1000, 3)))
    print(f"Predicted rotation: {result}")
