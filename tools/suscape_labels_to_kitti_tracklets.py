


import utils

import numpy as np
from scipy.interpolate import interp1d

def psr_to_np(psr):
    return np.array([
        psr['position']['x'],
        psr['position']['y'],
        psr['position']['z'],
        psr['rotation']['x'],
        psr['rotation']['y'],
        psr['rotation']['z'],
        psr['scale']['x'],
        psr['scale']['y'],
        psr['scale']['z'],
    ])
def np_to_psr(np):
    return {
        "position": {
            "x": np[0],
            "y": np[1],
            "z": np[2],
        },
        "rotation": {
            "x": np[3],
            "y": np[4],
            "z": np[5],
        },
        "scale": {
            "x": np[6],
            "y": np[7],
            "z": np[8],
        }
    }

def interpolate_one_obj(scene, obj_id):
    
    # frames are sorted
    frames = scene.meta['frames']
    
    xs = []  # frame index
    ys = []
    bs = []

    for i, f in enumerate(frames):
        b = scene.find_box_in_frame(f, obj_id)
        if b is not None:
            xs.append(i)
            ys.append(psr_to_np(b['psr']))
            bs.append(b)

    # unwarp ys "rotation" from [0, 2pi] to continuous, using numpy.unwrap
    ys = np.array(ys)
    ys[:,3:6] = np.unwrap(ys[:,3:6], axis=0)
    
    # 
    f = interp1d(xs, ys, axis=0, kind='linear', fill_value="extrapolate")

    # extrapolate 4 frames before and 4 frames after
    start  = xs[0]
    end = xs[-1]

    xnew = np.arange(start, end+1, 1)
    # ynew = f(xnew)
    ynew = f(xnew)

    # write back
    ynew = np.array(ynew)
    ynew[:,3:6] = np.mod(ynew[:,3:6]+np.pi, 2*np.pi) - np.pi  # wrap back to [-pi, pi]
    

    return ynew



def parse_suscape_trackets(rootdir, scene):
    s = utils.SuscapeScene(rootdir, scene)

    s.load_labels()

    objs = s.list_objs()

    tracklets = {}

    for obj in objs:
        id, typename = obj
        # availalbe_boxes = s.get_boxes_of_obj(id)
        boxes = interpolate_one_obj(s, id)
        # print(id, boxes.shape[0], len(availalbe_boxes.keys()))
        tracklets[id] = {
            "type": typename,
            "boxes": boxes
        }
    
    return tracklets


def write_kitti_tracklets(tracklets, output_file):

    def fmt(v):
        # format floats similar to existing file
        try:
            return "{:.16g}".format(float(v))
        except Exception:
            return str(v)

    items = list(tracklets.items())

    with open(output_file, 'w') as fh:
        fh.write('<?xml version="1.0" encoding="UTF-8" standalone="yes" ?>\n')
        fh.write('<!DOCTYPE boost_serialization>\n')
        fh.write('<boost_serialization signature="serialization::archive" version="9">\n')
        fh.write('<tracklets class_id="0" tracking_level="0" version="0">\n')
        fh.write('\t<count>{}</count>\n'.format(len(items)))
        fh.write('\t<item_version>1</item_version>\n')

        for idx, (obj_id, info) in enumerate(items):
            obj_type = info.get('type', 'Unknown')
            boxes = info.get('boxes')
            start_frame = info.get('start_frame', 0)

            # default size (h,w,l) from first box's scale if available
            h = w = l = 0.0
            if boxes is not None and len(boxes) > 0:
                # boxes columns: tx,ty,tz,rx,ry,rz,sx,sy,sz
                s = boxes[0, 6:9]
                # map scales to h,w,l as (sz, sy, sx) or fallback
                try:
                    sx, sy, sz = float(s[0]), float(s[1]), float(s[2])
                    # assume sx=length, sy=width, sz=height
                    l, w, h = sx, sy, sz
                except Exception:
                    h = w = l = 0.0

            fh.write('\t<item class_id="1" tracking_level="0" version="1">\n')
            fh.write('\t\t<objectType>{}</objectType>\n'.format(obj_type))
            fh.write('\t\t<h>{}</h>\n'.format(fmt(h)))
            fh.write('\t\t<w>{}</w>\n'.format(fmt(w)))
            fh.write('\t\t<l>{}</l>\n'.format(fmt(l)))
            fh.write('\t\t<first_frame>{}</first_frame>\n'.format(int(start_frame)))

            # poses
            fh.write('\t\t<poses class_id="2" tracking_level="0" version="0">\n')
            nposes = 0 if boxes is None else int(boxes.shape[0])
            fh.write('\t\t\t<count>{}</count>\n'.format(nposes))
            fh.write('\t\t\t<item_version>2</item_version>\n')

            for i in range(nposes):
                row = boxes[i]
                tx, ty, tz = row[0], row[1], row[2]
                rx, ry, rz = row[3], row[4], row[5]

                # write pose item
                fh.write('\t\t\t<item>\n')
                fh.write('\t\t\t\t<tx>{}</tx>\n'.format(fmt(tx)))
                fh.write('\t\t\t\t<ty>{}</ty>\n'.format(fmt(ty)))
                fh.write('\t\t\t\t<tz>{}</tz>\n'.format(fmt(tz)))
                fh.write('\t\t\t\t<rx>{}</rx>\n'.format(fmt(rx)))
                fh.write('\t\t\t\t<ry>{}</ry>\n'.format(fmt(ry)))
                fh.write('\t\t\t\t<rz>{}</rz>\n'.format(fmt(rz)))

                # default metadata values (following KITTI tracklet fields)
                fh.write('\t\t\t\t<state>1</state>\n')
                fh.write('\t\t\t\t<occlusion>0</occlusion>\n')
                fh.write('\t\t\t\t<occlusion_kf>0</occlusion_kf>\n')
                fh.write('\t\t\t\t<truncation>0</truncation>\n')
                fh.write('\t\t\t\t<amt_occlusion>0</amt_occlusion>\n')
                fh.write('\t\t\t\t<amt_occlusion_kf>0</amt_occlusion_kf>\n')
                fh.write('\t\t\t\t<amt_border_l>0</amt_border_l>\n')
                fh.write('\t\t\t\t<amt_border_r>1</amt_border_r>\n')
                fh.write('\t\t\t\t<amt_border_kf>-1</amt_border_kf>\n')
                fh.write('\t\t\t</item>\n')

            fh.write('\t\t</poses>\n')
            fh.write('\t\t<finished>1</finished>\n')
            fh.write('\t</item>\n')

        fh.write('</tracklets>\n')
        fh.write('</boost_serialization>\n')





if __name__ == "__main__":
    parse_suscape_trackets("../data", "scene-000002")