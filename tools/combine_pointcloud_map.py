import os
import re
import json
from utils import remove_box, SuscapeScene
import numpy as np
import argparse
import copy

parser = argparse.ArgumentParser(description='adjust ego pose')
parser.add_argument('--data', type=str, help="", default="/home/lie/nas/suscape_scenes_10hz")
parser.add_argument('--lidar', type=str, default="", help="")
parser.add_argument('--scenes', type=str, default="scene-000000", help="")
parser.add_argument('--boxscale', type=float, default=1.3, help="scale to remove box")
args = parser.parse_args()



def remove_objects(pts, objs):
    
    # remove egocar head & tail
    filter = (pts[:,1] > 4) | (pts[:,1] < -4) | (pts[:,0] > 2) | (pts[:,0] < -2)

    for obj in objs:
        filter = filter & remove_box(pts[:, :3], obj, 0, args.boxscale) 
    
    return pts[filter]


def combine_and_save_lidars(lidars, poses, file):
     
    map = []
    for i, lidar in enumerate(lidars):
         pts = np.matmul(np.concatenate([lidar[:, 0:3], np.ones([lidar.shape[0],1])], axis=1), poses[i].T)
         l = np.concatenate([pts[:, 0:3], lidar[:, 3:]], axis=1)
         map.append(l)
    
    map = np.concatenate(map, axis=0)
    map = map.astype(np.float32)

    color = (map[:, 4:7]*256.0).astype(np.uint8).astype(np.int32)

    color = (color[:,0] * 0x100  + color[:,1])*0x100 + color[:,2]
    color = color.astype(np.int32)

    size = map.shape[0]
    with open(file, 'wb') as f:
        header = f"""# .PCD v.7 - Point Cloud Data file format
VERSION .7
FIELDS x y z intensity rgb
SIZE 4 4 4 4 4
TYPE F F F F I
COUNT 1 1 1 1 1
WIDTH {size}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {size}
DATA binary
"""
            
            
        f.write(header.encode('utf-8'))
        for i,d in enumerate(map[:, :4]): 
            f.write(d.tobytes())
            f.write(color[i].tobytes())
        
ego_vehicle_box = {
            "obj_id": "-1",
            "obj_type": "EgoVehicle",
            "psr": {
                "position": {
                    "x": 0.0,
                    "y": 0.0,
                    "z": -1.0
                },
                "rotation": {
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.0
                },
                "scale": {
                    "x": 5.0,
                    "y": 3.0,
                    "z": 2.0
                }
            }
        }

def proc_scene(scene):
    scene = SuscapeScene(args.data, scene, args.lidar)
    frames = scene.meta['frames']
    lidar_pose_path = os.path.join(args.data, scene.name, 'lidar_pose')

    lidars = []
    poses = []

    for i in range(0, len(frames)):

        next_frame = frames[i]
        next_lidar = scene.read_lidar(next_frame)

        objs = scene.get_boxes_by_frame(next_frame)
        # objs.append(ego_vehicle_box)  # add ego vehicle box
        # egovehicle is handled in remove_objects
        next_lidar = remove_objects(next_lidar, objs)
        lidars.append(next_lidar)


        with open(os.path.join(lidar_pose_path, next_frame+'.json'), 'r') as f:
            data = json.load(f)
            lidar_pose = data['lidarPose']
            lidar_pose = np.array(lidar_pose).reshape(4, 4)
            poses.append(lidar_pose)
                    

    
    os.makedirs(os.path.join(args.data, scene.name, 'map'), exist_ok=True)
    combine_and_save_lidars(lidars, poses, os.path.join(args.data, scene.name, 'map', 'map2.pcd'))



scenes = os.listdir(args.data)
scenes.sort()
for s in scenes:
    if re.fullmatch(args.scenes, s):
        print('processing', s)
        proc_scene(s)