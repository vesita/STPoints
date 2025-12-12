
# cd ./suscape_scenes_10hz
# sed -i 's/2hz/10hz/g' scene-000002/desc.json
# python3 ~/code2/dataset_tools/crop_scene.py regen ./scene-000002 '' '000,100,200,300,400,500,600,700,800,900'
# cp -H -r ../suscape_scenes/scene-000002/label ./scene-000002/label_2hz 
# python ~/code2/SUSTechPoints-be-dev/tools/interpolate_10hz_labels.py --data . --scenes 'scene-000000' 


from utils import SuscapeScene
import numpy as np
import argparse
import os
import re
import json

from scipy.interpolate import interp1d

parser = argparse.ArgumentParser(description='interpolate 10hz labels')
parser.add_argument('--data', type=str, default="/home/lie/nas/suscape_scenes_10hz", help="")
parser.add_argument('--scenes', type=str, default="scene-000000", help="")


args = parser.parse_args()


# 1. use egovehicle pose (coarse) to do interpolation
# 2. remove all dynamic objects
# 3. finely-adjust lidar-pose
# 4. redo interpolation with lidar-pose


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
def interpolate_one_obj(scene, obj_id, output_lables):
    boxes  = scene.get_boxes_of_obj(o[0])
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
    start  = max(xs[0]-4, 0)
    end = min(xs[-1]+4, len(frames)-1)

    xnew = np.arange(start, end+1, 1)
    ynew = f(xnew)

    # wrap back
    ynew = np.array(ynew)
    ynew[:,3:6] = np.mod(ynew[:,3:6]+np.pi, 2*np.pi) - np.pi  # wrap back to [-pi, pi]
    
    last_existent_box = bs[0]

    for i,x in enumerate(xnew):
        frame = frames[x]
        if x in xs:
            last_existent_box = bs[xs.index(x)]
        
        # shallow copy, we only replace 'psr'
        b = last_existent_box.copy()
        b["psr"]= np_to_psr(ynew[i])

        if frame in output_lables:
            output_lables[frame].append(b)
        else:
            output_lables[frame] = [b]

   


if __name__ == "__main__":

    # read all scene names
    all_scenes = os.listdir(args.data)
    scenes = list(filter(lambda s: re.fullmatch(args.scenes, s), all_scenes))
    scenes.sort()
    
    print(scenes)


    for scene_name in scenes:
        scene = SuscapeScene(args.data, scene_name, label_folder="label_2hz")
        scene.load_labels()
        
        objs = scene.list_objs()
        print(objs)

        # ego_poses = scene.load_ego_pose()
        # print(ego_poses)

        output_lables = {}
        
        for o in objs:
            print(o)
            # print(scene.get_obj(o))
            # boxes = scene.get_boxes_of_obj(o[0])
            # print(boxes)

            interpolate_one_obj(scene, o[0], output_lables)
    

        for k in output_lables:
            # print(k, len(output_lables[k]))
            # write to file

            if len(output_lables[k]) == 0:
                continue

            file = f"{args.data}/{args.scenes}/label/{k}.json"
            # if not os.path.exists(file):            
            with open(file, "w") as f:
                    l = {
                        "frame": k,
                        "objs": output_lables[k],
                        "scene": scene_name,
                    }
                    f.write(json.dumps(l, indent=4))
            