import numpy as np

import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '..'))
import scene_reader


class MAFilter:
    def __init__(self, init_x):
        self.x = init_x
        self.v = np.zeros(9)
        self.step = 0

    def update(self, x):
        if self.step == 0:
            self.v = x - self.x
        else:
            self.v = self.v * 0.5 + (x - self.x) * 0.5
        self.x[0:9] = x
        self.step += 1

    def predict(self):
        self.x += self.v
        self.step += 1
        return self.x


def get_my_filter(init_x):
    return MAFilter(init_x)


def get_obj_ann(scene, frame, id):
    ann = scene_reader.read_annotations(scene, frame)
    target_ann = list(filter(lambda a: a["obj_id"] == id, ann))
    if len(target_ann) == 1:
        return target_ann[0]
    elif len(target_ann) > 1:
        print("Warning: duplicate object id found!")
        return target_ann[0]
    else:
        return None


def ann_to_kalman_state(ann):
    return np.array([
        ann["psr"]["position"]["x"],
        ann["psr"]["position"]["y"],
        ann["psr"]["position"]["z"],
        ann["psr"]["scale"]["x"],
        ann["psr"]["scale"]["y"],
        ann["psr"]["scale"]["z"],
        ann["psr"]["rotation"]["x"],
        ann["psr"]["rotation"]["y"],
        ann["psr"]["rotation"]["z"],
    ])


def kalman_state_to_ann(proto, state):
    state = np.reshape(state, -1)
    return {
        "psr": {
            "position": {"x": state[0], "y": state[1], "z": state[2]},
            "scale": {"x": state[3], "y": state[4], "z": state[5]},
            "rotation": {"x": state[6], "y": state[7], "z": state[8]},
        },
        "obj_type": proto["obj_type"],
        "obj_id": proto["obj_id"],
    }


def interpolate_segment(start_ann, end_ann, insert_number):
    end = ann_to_kalman_state(end_ann)
    start = ann_to_kalman_state(start_ann)
    linear_delta = (end - start) / (insert_number + 1)
    return list(map(
        lambda i: kalman_state_to_ann(start_ann, start + linear_delta * (i + 1)),
        range(insert_number)
    ))


def interpolate(annotations):
    N = len(annotations)
    i = 0
    num_interpolate = 0
    while i + 1 < N:
        start = None
        end = None
        while (i + 1) < N and not (annotations[i] and (annotations[i + 1] is None)):
            i = i + 1
        start = i
        i = i + 2
        while i < N and annotations[i] is None:
            i = i + 1
        if i < len(annotations):
            end = i
        if start is not None and end is not None:
            predicted = interpolate_segment(annotations[start], annotations[end], end - start - 1)
            for p in predicted:
                p["annotator"] = "I"
            annotations[(start + 1):end] = predicted
            num_interpolate += end - start - 1
    return num_interpolate


def kalmanfilter_pred(annotations):
    i = 0
    while annotations[i] is None or annotations[i].get("annotator") == "K":
        i += 1
    if i >= len(annotations):
        return 0
    state = ann_to_kalman_state(annotations[i])
    ref_ann = annotations[i]
    kalmanfilter = get_my_filter(state)
    i += 1
    while i < len(annotations) and annotations[i] is not None:
        state = ann_to_kalman_state(annotations[i])
        kalmanfilter.predict()
        kalmanfilter.update(state)
        ref_ann = annotations[i]
        i += 1
    pred_num = 0
    while i < len(annotations) and annotations[i] is None:
        kalmanfilter.predict()
        ann = kalman_state_to_ann(ref_ann, kalmanfilter.x)
        ann["annotator"] = "K"
        annotations[i] = ann
        i += 1
        pred_num += 1
    return pred_num


def predict(scene_id, obj_id, current_frame, predict_frame):
    scene = scene_reader.get_one_scene(scene_id)
    frames = scene["frames"]
    annotations = list(map(lambda f: get_obj_ann(scene_id, f, obj_id), frames))
    for i, a in enumerate(annotations):
        if a and a.get("annotator") and (a["annotator"] == "K" or a["annotator"] == "I"):
            annotations[i] = None
    num_pred = 0
    num_pred += interpolate(annotations)
    num_pred += kalmanfilter_pred(annotations)
    annotations.reverse()
    num_pred += kalmanfilter_pred(annotations)
    annotations.reverse()
    updated_frames = []
    for i, a in enumerate(annotations):
        if a is None:
            continue
        if a.get("annotator") and (a["annotator"] == "K" or a["annotator"] == "I"):
            write_annotation_back(scene_id, frames[i], a)
            updated_frames.append(frames[i])
    return updated_frames


def write_annotation_back(scene_id, frame, new_ann):
    ann = scene_reader.read_annotations(scene_id, frame)
    ann = list(filter(lambda a: a["obj_id"] != new_ann["obj_id"], ann))
    ann.append(new_ann)
    scene_reader.save_annotations(scene_id, frame, ann)


if __name__ == "__main__":
    predict("rider_2hz", "1", None, None)
