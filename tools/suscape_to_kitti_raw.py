


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
    

    return {
        "boxes": ynew,
        "start_frame": start
    }



def parse_suscape_trackets(s):  # s: scene
    

    s.load_labels()

    objs = s.list_objs()

    tracklets = {}

    for obj in objs:
        id, typename = obj
        # availalbe_boxes = s.get_boxes_of_obj(id)
        tracklet = interpolate_one_obj(s, id)
        # print(id, boxes.shape[0], len(availalbe_boxes.keys()))
        tracklets[id] = tracklet
        tracklets[id]['type'] = typename
    
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

def write_calib_files(scene, output_folder):
    
    """
    example
    calib_time: 15-Mar-2012 11:37:16
    R: 7.533745e-03 -9.999714e-01 -6.166020e-04 1.480249e-02 7.280733e-04 -9.998902e-01 9.998621e-01 7.523790e-03 1.480755e-02
    T: -4.069766e-03 -7.631618e-02 -2.717806e-01
    delta_f: 0.000000e+00 0.000000e+00
    delta_c: 0.000000e+00 0.000000e+00


    """
    static_calib = scene.meta['calib']["camera"]["front"]
    lidar_to_camera = np.array(static_calib["lidar_to_camera"]).reshape([4,4])
    intrinsic = np.array(static_calib["intrinsic"]).reshape([3,3])

    with open(os.path.join(output_folder, "calib_velo_to_cam.txt"), 'w') as f:
        f.write("calib_time: 15-Mar-2025 11:37:16\n")
        R = lidar_to_camera[0:3,0:3]
        T = lidar_to_camera[0:3,3]
        R_flat = R.flatten()
        R_str = " ".join(["{:.6e}".format(v) for v in R_flat])
        T_str = " ".join(["{:.6e}".format(v) for v in T])
        f.write("R: {}\n".format(R_str))
        f.write("T: {}\n".format(T_str))
        f.write("delta_f: 0.000000e+00 0.000000e+00\n")
        f.write("delta_c: 0.000000e+00 0.000000e+00\n")


    """
    cam to cam

    calib_time: 09-Jan-2012 13:57:47
    corner_dist: 9.950000e-02
    S_00: 1.392000e+03 5.120000e+02
    K_00: 9.842439e+02 0.000000e+00 6.900000e+02 0.000000e+00 9.808141e+02 2.331966e+02 0.000000e+00 0.000000e+00 1.000000e+00
    D_00: -3.728755e-01 2.037299e-01 2.219027e-03 1.383707e-03 -7.233722e-02
    R_00: 1.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00
    T_00: 2.573699e-16 -1.059758e-16 1.614870e-16
    S_rect_00: 1.242000e+03 3.750000e+02
    R_rect_00: 9.999239e-01 9.837760e-03 -7.445048e-03 -9.869795e-03 9.999421e-01 -4.278459e-03 7.402527e-03 4.351614e-03 9.999631e-01
    P_rect_00: 7.215377e+02 0.000000e+00 6.095593e+02 0.000000e+00 0.000000e+00 7.215377e+02 1.728540e+02 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00

    
    calib_cam_to_cam.txt: Camera-to-camera calibration
    --------------------------------------------------

    - S_xx: 1x2 size of image xx before rectification
    - K_xx: 3x3 calibration matrix of camera xx before rectification
    - D_xx: 1x5 distortion vector of camera xx before rectification
    - R_xx: 3x3 rotation matrix of camera xx (extrinsic)
    - T_xx: 3x1 translation vector of camera xx (extrinsic)
    - S_rect_xx: 1x2 size of image xx after rectification
    - R_rect_xx: 3x3 rectifying rotation to make image planes co-planar
    - P_rect_xx: 3x4 projection matrix after rectification

    """

    # front image width and height
    width = scene.meta['camera']['front']['width']
    height = scene.meta['camera']['front']['height']

    with open(os.path.join(output_folder, "calib_cam_to_cam.txt"), 'w') as f:
        f.write("calib_time: 09-Jan-2025 13:57:47\n")
        f.write("corner_dist: 9.950000e-02\n")
        
        camera_ids = ["00", "01", "02", "03"]  # only front camera
        for id in camera_ids:
            
            f.write("S_{}: {} {}\n".format(id, int(width), int(height)))
            # k_xx use identiy for simplicity
            identiy33 = np.eye(3).flatten()
            identiy33_str = " ".join(["{:.6e}".format(v) for v in identiy33])
            f.write("K_{}: {}\n".format(id, identiy33_str))
            # d_xx use zeros for simplicity
            f.write("D_{}: 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00\n".format(id))
            # r_xx use identity for simplicity
            f.write("R_{}: {}\n".format(id, identiy33_str))
            # t_xx use zeros for simplicity
            f.write("T_{}: 0.000000e+00 0.000000e+00 0.000000e+00\n".format(id))
            f.write("S_rect_{}: {} {}\n".format(id, int(width), int(height)))
            f.write("R_rect_{}: {}\n".format(id, identiy33_str))
            # p_rect_00
            p_rect_00 = np.zeros((3,4))
            p_rect_00[0:3,0:3] = intrinsic
            
            p_rect_00_flat = p_rect_00.flatten()
            p_rect_00_str = " ".join(["{:.6e}".format(v) for v in p_rect_00_flat])
            f.write("P_rect_{}: {}\n".format(id, p_rect_00_str))

        

def write_image02_files(scene, output_folder):
    """
    link files to output_folder
    images to output_folder/data
    timestamps to output_folder/timestamps.txt
    """
    
    frames = scene.meta['frames']
    os.makedirs(os.path.join(output_folder, "data"), exist_ok=True)
    
    # frames are timestamps, target filenames use 10-digit zero-padded frame index
    timestamps_file = os.path.join(output_folder, "timestamps.txt")
    with open(timestamps_file, 'w') as tf:
        for i, f in enumerate(frames):
            # link image file, use relative file path
            
            tgt_image_file = os.path.join(output_folder, "data", "{:010d}.png".format(i))
            src_image_file = os.path.join(scene.scene_path, "camera", "front", f + ".jpg")

            link_src = os.path.relpath(src_image_file, os.path.dirname(tgt_image_file))
            
            os.symlink(link_src, tgt_image_file)

            # write timestamp line
            tf.write("{}\n".format(f))

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--rootdir', type=str, required=True, help='Root directory of Suscape dataset')
    parser.add_argument('--scene', type=str, required=True, help='Scene name to process')
    parser.add_argument('--output', type=str, required=True, help='Output KITTI tracklet XML file')

    args = parser.parse_args()

    print(args)

 

    os.makedirs(os.path.join(args.output, args.scene, args.scene), exist_ok=True)

    s = utils.SuscapeScene(args.rootdir, args.scene)

    tracklets = parse_suscape_trackets(s)
    write_kitti_tracklets(tracklets, os.path.join(args.output, args.scene, args.scene, "tracklet_labels.xml"))

    write_calib_files(s, os.path.join(args.output, args.scene))

    write_image02_files(s, os.path.join(args.output, args.scene, args.scene, "image_02"))

# cd tools
# python suscape_labels_to_kitti_raw.py --rootdir ../data --scene scene-000002 --output ../out/test