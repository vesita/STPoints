import random
import string
import re

import cherrypy
import os
import json
import toml
import cv2
import numpy as np
from jinja2 import Environment, FileSystemLoader
env = Environment(loader=FileSystemLoader('./'))

import sys
import scene_reader
from tools import check_labels  as check
from calibpy.calib_pnp import solve_pnp_ippe


# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(BASE_DIR)

#sys.path.append(os.path.join(BASE_DIR, './algos'))
#import algos.rotation as rotation
from algos import pre_annotate


#sys.path.append(os.path.join(BASE_DIR, '../tracking'))
#import algos.trajectory as trajectory

# extract_object_exe = "~/code/pcltest/build/extract_object"
# registration_exe = "~/code/go_icp_pcl/build/test_go_icp"

# sys.path.append(os.path.join(BASE_DIR, './tools'))
# import tools.dataset_preprocess.crop_scene as crop_scene

def _euler_to_rotmat(rx, ry, rz):
    """ZYX 欧拉角 → 3×3 旋转矩阵（与 JS euler_angle_to_rotate_matrix 一致）"""
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0],
                   [0, 0, 1]])
    return Rz @ Ry @ Rx


def _psr_to_8_corners(position, scale, rotation):
    """PSR 标注框 → 8 个世界坐标角点（与 JS psr_to_xyz 一致）"""
    R = _euler_to_rotmat(rotation["x"], rotation["y"], rotation["z"])
    t = np.array([position["x"], position["y"], position["z"]])
    sx, sy, sz = scale["x"] / 2, scale["y"] / 2, scale["z"] / 2
    local = np.array([
        [sx,  sy, -sz], [sx, -sy, -sz],  # FLB, FRB
        [sx, -sy,  sz], [sx,  sy,  sz],  # FRT, FLT
        [-sx, sy, -sz], [-sx, -sy, -sz], # RLB, RRB
        [-sx, -sy, sz], [-sx, sy,  sz],  # RRT, RLT
    ])
    return (R @ local.T).T + t


class Root(object):
    @cherrypy.expose
    def index(self, scene="", frame=""):
      tmpl = env.get_template('index.html')
      return tmpl.render()
  
    @cherrypy.expose
    def icon(self):
      tmpl = env.get_template('test_icon.html')
      return tmpl.render()

    @cherrypy.expose
    def ml(self):
      tmpl = env.get_template('test_ml.html')
      return tmpl.render()
  
    @cherrypy.expose
    def reg(self):
      tmpl = env.get_template('registration_demo.html')
      return tmpl.render()

    @cherrypy.expose
    def view(self, file):
      tmpl = env.get_template('view.html')
      return tmpl.render()

    # @cherrypy.expose
    # def saveworld(self, scene, frame):

    #   # cl = cherrypy.request.headers['Content-Length']
    #   rawbody = cherrypy.request.body.readline().decode('UTF-8')

    #   with open("./data/"+scene +"/label/"+frame+".json",'w') as f:
    #     f.write(rawbody)
      
    #   return "ok"

    @cherrypy.expose
    def saveworldlist(self):

      # cl = cherrypy.request.headers['Content-Length']
      rawbody = cherrypy.request.body.read().decode('UTF-8')
      data = json.loads(rawbody)

      for d in data:
        scene = d["scene"]
        frame = d["frame"]
        ann = d["annotation"]

        # 路径穿越防护：只允许字母/数字/下划线/连字符
        if not re.match(r'^[\w\-]+$', scene) or not re.match(r'^[\w\-]+$', frame):
            return {"success": False, "error": "invalid scene or frame name"}
        label_dir = os.path.join("./data", scene, "label")
        os.makedirs(label_dir, exist_ok=True)
        with open(os.path.join(label_dir, frame+".json"), 'w') as f:
          json.dump(ann, f, indent=2, sort_keys=True)

      return "ok"


    @cherrypy.expose
    @cherrypy.tools.json_out()
    def cropscene(self):
      rawbody = cherrypy.request.body.read().decode('UTF-8')
      data = json.loads(rawbody)

      rawdata = data["rawSceneId"]
      timestamp = rawdata.split("_")[0]

      # 路径穿越/注入防护
      if not re.match(r'^[\w\-]+$', timestamp):
          return {"code": -1, "log": ["invalid rawSceneId"]}
      safe_start = re.sub(r'[^\w\-\:\.]', '', data["startTime"])
      safe_seconds = re.sub(r'[^\w\-\.]', '', data["seconds"])
      safe_desc = re.sub(r'[^\w\-]', '', data["desc"])

      print("generate scene")
      log_file = "temp/crop-scene-"+timestamp+".log"

      cmd = "python ./tools/dataset_preprocess/crop_scene.py generate " + \
        rawdata[0:10]+"/"+timestamp + "_preprocessed/dataset_2hz " + \
        "- " + \
        safe_start + " " + \
        safe_seconds + " " + \
        "\"" + safe_desc + "\"" + \
        "> " + log_file + " 2>&1"
      print(cmd)

      code = os.system(cmd)

      with open(log_file) as f:
        log = list(map(lambda s: s.strip(), f.readlines()))

      os.remove(log_file)

      return {"code": code,
              "log": log
              }


    @cherrypy.expose
    @cherrypy.tools.json_out()
    def checkscene(self, scene):
      ck = check.LabelChecker(os.path.join("./data", scene))
      ck.check()
      print(ck.messages)
      return ck.messages


    # @cherrypy.expose
    # @cherrypy.tools.json_out()
    # def interpolate(self, scene, frame, obj_id):
    #   # interpolate_num = trajectory.predict(scene, obj_id, frame, None)
    #   # return interpolate_num
    #   return 0

    # data  N*3 numpy array
    @cherrypy.expose
    @cherrypy.tools.json_out()
    def predict_rotation(self):
      rawbody = cherrypy.request.body.read().decode('UTF-8')
      
      data = json.loads(rawbody)
      
      return {"angle": pre_annotate.predict_yaw(data["points"])}
      #return {}

    
    @cherrypy.expose    
    @cherrypy.tools.json_out()
    def auto_annotate(self, scene, frame):
      print("auto annotate ", scene, frame)
      return pre_annotate.annotate_file('./data/{}/lidar/{}.pcd'.format(scene,frame))
      


    @cherrypy.expose    
    @cherrypy.tools.json_out()
    def load_annotation(self, scene, frame):
      return scene_reader.read_annotations(scene, frame)


    @cherrypy.expose    
    @cherrypy.tools.json_out()
    def load_ego_pose(self, scene, frame):
      return scene_reader.read_ego_pose(scene, frame)


    @cherrypy.expose    
    @cherrypy.tools.json_out()
    def loadworldlist(self):
      rawbody = cherrypy.request.body.read().decode('UTF-8')
      worldlist = json.loads(rawbody)

      anns = list(map(lambda w:{
                      "scene": w["scene"],
                      "frame": w["frame"],
                      "annotation":scene_reader.read_annotations(w["scene"], w["frame"])},
                      worldlist))

      return anns
        

    # @cherrypy.expose    
    # @cherrypy.tools.json_out()
    # def auto_adjust(self, scene, ref_frame, object_id, adj_frame):
      
    #   #os.chdir("./temp")
    #   os.system("rm ./temp/src.pcd ./temp/tgt.pcd ./temp/out.pcd ./temp/trans.json")


    #   tgt_pcd_file = "./data/"+scene +"/lidar/"+ref_frame+".pcd"
    #   tgt_json_file = "./data/"+scene +"/label/"+ref_frame+".json"

    #   src_pcd_file = "./data/"+scene +"/lidar/"+adj_frame+".pcd"      
    #   src_json_file = "./data/"+scene +"/label/"+adj_frame+".json"

    #   cmd = extract_object_exe +" "+ src_pcd_file + " " + src_json_file + " " + object_id + " " +"./temp/src.pcd"
    #   print(cmd)
    #   os.system(cmd)

    #   cmd = extract_object_exe + " "+ tgt_pcd_file + " " + tgt_json_file + " " + object_id + " " +"./temp/tgt.pcd"
    #   print(cmd)
    #   os.system(cmd)

    #   cmd = registration_exe + " ./temp/tgt.pcd ./temp/src.pcd ./temp/out.pcd ./temp/trans.json"
    #   print(cmd)
    #   os.system(cmd)

    #   with open("./temp/trans.json", "r") as f:
    #     trans = json.load(f)
    #     print(trans)
    #     return trans

    #   return {}

    @cherrypy.expose    
    @cherrypy.tools.json_out()
    def datameta(self):
      return scene_reader.get_all_scenes()
    

    @cherrypy.expose    
    @cherrypy.tools.json_out()
    def scenemeta(self, scene):
      return scene_reader.get_one_scene(scene)

    @cherrypy.expose    
    @cherrypy.tools.json_out()
    def get_all_scene_desc(self):
      return scene_reader.get_all_scene_desc()

    @cherrypy.expose    
    @cherrypy.tools.json_out()
    def objs_of_scene(self, scene):
      return self.get_all_objs(os.path.join("./data",scene))

    def get_all_objs(self, path):
      label_folder = os.path.join(path, "label")
      if not os.path.isdir(label_folder):
        return []
        
      files = os.listdir(label_folder)

      files = filter(lambda x: x.split(".")[-1]=="json", files)


      def file_2_objs(f):
          with open(f) as fd:
              boxes = json.load(fd)
              objs = [x for x in map(lambda b: {"category": b.get("obj_type", "?"), "id": b.get("obj_id", "?")}, boxes)]
              return objs

      boxes = map(lambda f: file_2_objs(os.path.join(path, "label", f)), files)

      # the following map makes the category-id pairs unique in scene
      all_objs={}
      for x in boxes:
          for o in x:
              
              k = str(o["category"])+"-"+str(o["id"])

              if all_objs.get(k):
                all_objs[k]['count']= all_objs[k]['count']+1
              else:
                all_objs[k]= {
                  "category": o["category"],
                  "id": o["id"],
                  "count": 1
                }

      return [x for x in  all_objs.values()]


    # ── PnP extrinsic calibration ──────────────────────────────────────────

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @cherrypy.tools.json_in()
    def solve_pnp(self):
        """Compute extrinsic via IPPE from 4 2D–3D correspondences (no save)."""
        try:
            data = cherrypy.request.json
            scene = data["scene"]
            camera = data["camera"]
            points_3d = data["points_3d"]
            points_2d = data["points_2d"]

            print(f"[solve_pnp] scene={scene} camera={camera}")
            print(f"[solve_pnp] 3D: {points_3d}")
            print(f"[solve_pnp] 2D: {points_2d}")

            calib_file = os.path.join("./data", scene, "calib", "camera", camera + ".json")
            if not os.path.isfile(calib_file):
                return {"success": False, "error": f"calib file not found: {calib_file}"}

            with open(calib_file) as f:
                calib_data = json.load(f)

            intrinsic = calib_data.get("intrinsic")
            if not intrinsic:
                return {"success": False, "error": f"calib file missing intrinsic: {calib_file}"}
            camera_matrix = intrinsic
            dist_coeffs = calib_data.get("dist_coeffs")

            result = solve_pnp_ippe(points_3d, points_2d, camera_matrix, dist_coeffs)
            return result
        except Exception as e:
            print(f"[solve_pnp] UNHANDLED ERROR: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": f"solve_pnp 内部错误: {str(e)}"}

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @cherrypy.tools.json_in()
    def brute_force_pnp(self):
        """穷举 24 种点序排列，返回每种的重投影误差。"""
        try:
            import itertools

            data = cherrypy.request.json
            scene = data["scene"]
            camera = data["camera"]
            points_3d = data["points_3d"]
            points_2d = data["points_2d"]

            calib_file = os.path.join("./data", scene, "calib", "camera", camera + ".json")
            if not os.path.isfile(calib_file):
                return {"success": False, "error": f"calib file not found: {calib_file}"}

            with open(calib_file) as f:
                calib_data = json.load(f)

            intrinsic = calib_data.get("intrinsic")
            if not intrinsic:
                return {"success": False, "error": f"calib file missing intrinsic: {calib_file}"}
            camera_matrix = intrinsic
            dist_coeffs = calib_data.get("dist_coeffs")

            print(f"[brute] input points_3d={points_3d}")
            print(f"[brute] input points_2d={points_2d}")

            results = []
            best = {"error": float("inf"), "perm": None}

            for perm in itertools.permutations(range(4)):
                p3d = points_3d
                p2d = [points_2d[i] for i in perm]

                res = solve_pnp_ippe(p3d, p2d, camera_matrix, dist_coeffs)
                err = res.get("reprojection_error", float("inf"))
                results.append({"perm": list(perm), "error": err, "success": res.get("success", False), "extrinsic": res.get("extrinsic")})

                if err < best["error"]:
                    best["error"] = err
                    best["perm"] = list(perm)
                    best["extrinsic"] = res.get("extrinsic")

            unique_errors = set(round(r["error"], 2) for r in results)
            print(f"[brute] best={best['perm']}, error={best['error']:.2f}")
            print(f"[brute] unique error values ({len(unique_errors)}): {sorted(unique_errors)}")
            if len(unique_errors) == 1:
                print(f"[brute] 所有 24 种排列误差完全相同 = {results[0]['error']:.2f}px → 3D 点共线退化")
            return {"success": True, "results": results, "best": best}
        except Exception as e:
            print(f"[brute_force_pnp] UNHANDLED ERROR: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": f"brute_force_pnp 内部错误: {str(e)}"}

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @cherrypy.tools.json_in()
    def calib_save(self):
        """Persist extrinsic matrix to calib file on disk, also export TOML."""
        data = cherrypy.request.json
        scene = data["scene"]
        camera = data["camera"]
        extrinsic = data["extrinsic"]

        # 路径穿越防护
        if not re.match(r'^[\w\-]+$', scene) or not re.match(r'^[\w\-]+$', camera):
            return {"success": False, "error": "invalid scene or camera name"}

        calib_file = os.path.join("./data", scene, "calib", "camera", camera + ".json")
        if os.path.isfile(calib_file):
            with open(calib_file) as f:
                calib_data = json.load(f)
        else:
            # 文件不存在时自动创建
            os.makedirs(os.path.dirname(calib_file), exist_ok=True)
            calib_data = {}

        calib_data["extrinsic"] = extrinsic

        with open(calib_file, "w") as f:
            json.dump(calib_data, f, indent=2)

        # ── 同步导出 TOML ────────────────────────────────────────────────
        toml_path = os.path.join("./data", scene, "calib", "camera", camera + ".toml")
        toml_data = {"camera": {}}
        intrinsic = calib_data.get("intrinsic")
        dist_coeffs = calib_data.get("dist_coeffs")
        if intrinsic:
            # 扁平 9 元素 → 3×3 嵌套列表
            K = intrinsic
            if len(K) == 9:
                K = [K[0:3], K[3:6], K[6:9]]
            toml_data["camera"]["intrinsic"] = K
        if extrinsic:
            # 扁平 16 元素 → 4×4 嵌套列表
            E = extrinsic
            if len(E) == 16:
                E = [E[0:4], E[4:8], E[8:12], E[12:16]]
            toml_data["camera"]["extrinsic"] = E
        if dist_coeffs:
            toml_data["camera"]["dist_coeffs"] = dist_coeffs

        with open(toml_path, "w") as f:
            toml.dump(toml_data, f)
        print(f"[calib_save] TOML exported: {toml_path}")

        return {"success": True}

    @cherrypy.expose
    def undistort(self, scene="", camera="", frame=""):
        """返回去畸变后的图像。若标定文件中无 dist_coeffs 则返回原图。"""
        if not scene or not camera or not frame:
            cherrypy.response.status = 400
            return b"missing parameters"

        # 读取标定文件
        calib_file = os.path.join("./data", scene, "calib", "camera", camera + ".json")
        if not os.path.isfile(calib_file):
            cherrypy.response.status = 404
            return b"calib file not found"

        with open(calib_file) as f:
            calib = json.load(f)

        intrinsic = calib.get("intrinsic")
        dist_coeffs = calib.get("dist_coeffs")

        # 无畸变系数 → 重定向到原图
        scene_meta = scene_reader.get_one_scene(scene)
        cam_ext = scene_meta.get("camera_ext", ".jpg")
        img_path = os.path.join("./data", scene, "camera", camera, frame + cam_ext)

        if not dist_coeffs or not intrinsic:
            raise cherrypy.HTTPRedirect(img_path)

        # 读取原始图像
        img = cv2.imread(img_path)
        if img is None:
            cherrypy.response.status = 404
            return b"image not found"

        # 构建相机矩阵和畸变系数
        K = np.array(intrinsic, dtype=np.float64).reshape(3, 3)
        D = np.array(dist_coeffs, dtype=np.float64)

        # 去畸变
        undistorted = cv2.undistort(img, K, D)

        # 返回 JPEG
        _, buf = cv2.imencode('.jpg', undistorted)
        cherrypy.response.headers['Content-Type'] = 'image/jpeg'
        return buf.tobytes()

    def detect_corners(self, **kwargs):
        """检测棋盘格角点。接收 multipart form: images[], rows, cols"""
        import base64

        rows = int(kwargs.get("rows", 6))
        cols = int(kwargs.get("cols", 9))
        # CherryPy 将 multipart 文件放在 kwargs 中
        files = kwargs.get("images")
        if not files:
            # 单文件时不是列表
            files = [kwargs.get("image")] if kwargs.get("image") else []
        if not isinstance(files, list):
            files = [files]

        results = []
        for f in files:
            if not f:
                continue
            filename = getattr(f, "filename", "unknown")
            raw = f.file.read()
            arr = np.frombuffer(raw, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                results.append({"filename": filename, "success": False, "error": "无法解码图片"})
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(
                gray, (cols, rows),
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            )

            if ret:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                # 画角点预览
                preview = img.copy()
                cv2.drawChessboardCorners(preview, (cols, rows), corners, ret)
                _, buf = cv2.imencode('.jpg', preview)
                preview_b64 = base64.b64encode(buf).decode('ascii')
                # 返回角点坐标（扁平数组）
                corner_list = corners.reshape(-1, 2).tolist()
                results.append({
                    "filename": filename, "success": True,
                    "preview": preview_b64, "corners": corner_list
                })
            else:
                results.append({"filename": filename, "success": False, "error": "未检测到棋盘格角点"})

        return {"results": results}

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @cherrypy.tools.json_in()
    def calibrate_intrinsics(self):
        """使用检测到的角点计算相机内参。"""

        data = cherrypy.request.json
        rows = int(data.get("rows", 6))
        cols = int(data.get("cols", 9))
        image_data_list = data.get("images", [])  # [{filename, corners, width, height}]

        print(f"[calibrate_intrinsics] 收到请求: rows={rows}, cols={cols}, 图片数={len(image_data_list)}")

        objp = np.zeros((rows * cols, 3), np.float32)
        objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)

        obj_points = []
        img_points = []
        image_size = None

        for item in image_data_list:
            corners = np.array(item["corners"], dtype=np.float32).reshape(-1, 1, 2)
            w = int(item.get("width", 0))
            h = int(item.get("height", 0))
            if w <= 0 or h <= 0:
                print(f"[calibrate_intrinsics] 跳过 {item.get('filename')}: 尺寸无效 ({w}x{h})")
                continue
            image_size = (w, h)
            obj_points.append(objp)
            img_points.append(corners)

        print(f"[calibrate_intrinsics] 有效图片: {len(obj_points)}, image_size={image_size}")

        if len(obj_points) < 3:
            return {"success": False, "error": f"有效图片不足（{len(obj_points)}张，至少需要3张）"}

        rms, K, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, image_size, None, None
        )

        return {
            "success": True,
            "intrinsic": K.flatten().tolist(),
            "dist_coeffs": dist_coeffs.flatten().tolist(),
            "error": float(rms),
            "image_count": len(obj_points)
        }

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @cherrypy.tools.json_in()
    def save_intrinsics(self):
        """保存内参和畸变系数到标定文件。"""
        data = cherrypy.request.json
        scene = data["scene"]
        camera = data["camera"]
        intrinsic = data["intrinsic"]
        dist_coeffs = data["dist_coeffs"]

        # 路径穿越防护
        if not re.match(r'^[\w\-]+$', scene) or not re.match(r'^[\w\-]+$', camera):
            return {"success": False, "error": "invalid scene or camera name"}

        calib_file = os.path.join("./data", scene, "calib", "camera", camera + ".json")
        if not os.path.isfile(calib_file):
            # 创建新的标定文件
            os.makedirs(os.path.dirname(calib_file), exist_ok=True)
            calib_data = {"extrinsic": [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]}
        else:
            with open(calib_file) as f:
                calib_data = json.load(f)

        calib_data["intrinsic"] = intrinsic
        calib_data["dist_coeffs"] = dist_coeffs

        with open(calib_file, "w") as f:
            json.dump(calib_data, f, indent=2)

        return {"success": True}


    # ── Calibration preview ────────────────────────────────────────────────

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @cherrypy.tools.json_in()
    def render_calibration_preview(self):
        """遍历场景中所有有标注的帧，将 3D 标注框通过当前外参+内参投影到 2D 图像并保存。"""
        data = cherrypy.request.json
        scene = data["scene"]
        camera_name = data.get("camera")

        scene_meta = scene_reader.get_one_scene(scene)
        cameras = scene_meta.get("camera", [])
        if not cameras:
            return {"success": False, "error": "场景中无 camera 文件夹"}

        if camera_name not in cameras:
            camera_name = cameras[0]

        calib = scene_meta.get("calib", {}).get("camera", {}).get(camera_name)
        if not calib:
            return {"success": False, "error": f"camera {camera_name} 无标定文件"}

        intrinsic = calib.get("intrinsic")
        extrinsic = calib.get("extrinsic")
        dist_coeffs = calib.get("dist_coeffs")

        if not intrinsic or not extrinsic:
            return {"success": False, "error": "标定文件缺少 intrinsic 或 extrinsic"}

        K = np.array(intrinsic, dtype=np.float64).reshape(3, 3)
        ext = np.array(extrinsic, dtype=np.float64).reshape(4, 4)
        R_mat = ext[:3, :3]
        t_vec = ext[:3, 3]
        rvec, _ = cv2.Rodrigues(R_mat)
        dist = np.array(dist_coeffs, dtype=np.float64).reshape(-1, 1) if dist_coeffs else None

        frames = scene_meta.get("frames", [])
        camera_ext = scene_meta.get("camera_ext", ".jpg")

        out_dir = os.path.join("./temp", "calib_preview", scene)
        os.makedirs(out_dir, exist_ok=True)

        # 8 colours for up to 8 boxes per frame
        BOX_COLORS = [
            (0, 200, 0),    # green
            (0, 100, 255),  # blue
            (50, 50, 255),  # red
            (255, 200, 0),  # cyan
            (255, 0, 200),  # magenta
            (0, 200, 200),  # yellow
            (100, 200, 0),  # lime
            (200, 100, 0),  # orange
        ]

        results = []
        total_err = 0.0
        err_count = 0

        for frame in frames:
            ann = scene_reader.read_annotations(scene, frame)
            if not ann:
                continue

            img_path = os.path.join("./data", scene, "camera", camera_name, frame + camera_ext)
            img = cv2.imread(img_path)
            if img is None:
                continue

            overlay = img.copy()
            h, w = img.shape[:2]

            for bi, box in enumerate(ann[:8]):
                psr = box.get("psr", {})
                pos = psr.get("position", {})
                scl = psr.get("scale", {})
                rot = psr.get("rotation", {})

                world_corners = _psr_to_8_corners(pos, scl, rot)
                corners_2d, _ = cv2.projectPoints(world_corners, rvec, t_vec, K, dist)
                pts = corners_2d.reshape(-1, 2).astype(np.int32)

                color = BOX_COLORS[bi % len(BOX_COLORS)]
                dim_color = tuple(c // 2 for c in color)

                # front face: 0-1-2-3
                for i, j in [(0, 1), (1, 2), (2, 3), (3, 0)]:
                    cv2.line(overlay, tuple(pts[i]), tuple(pts[j]), color, 2)

                # rear face: 4-5-6-7
                for i, j in [(4, 5), (5, 6), (6, 7), (7, 4)]:
                    cv2.line(overlay, tuple(pts[i]), tuple(pts[j]), dim_color, 1)

                # connections front→rear
                for i, j in [(0, 4), (1, 5), (2, 6), (3, 7)]:
                    cv2.line(overlay, tuple(pts[i]), tuple(pts[j]), dim_color, 1, cv2.LINE_AA)

                # label
                label = f"{box.get('obj_type', '?')} #{box.get('obj_id', '?')}"
                label_pos = (pts[0][0], max(pts[0][1] - 6, 12))
                cv2.putText(overlay, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # compute projected box centre spread (pseudo-error metric)
                centre_2d = pts.mean(axis=0)
                spread = np.sqrt(np.sum((pts - centre_2d) ** 2, axis=1)).max()
                total_err += spread
                err_count += 1

            # semi-transparent overlay
            cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

            # frame info
            info = f"{frame} | {len(ann)} obj"
            cv2.putText(img, info, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(img, info, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

            out_path = os.path.join(out_dir, f"{camera_name}_{frame}.jpg")
            cv2.imwrite(out_path, img)

            results.append({
                "frame": frame,
                "num_boxes": len(ann),
                "image_url": f"/temp/calib_preview/{scene}/{camera_name}_{frame}.jpg"
            })

        return {
            "success": True,
            "scene": scene,
            "camera": camera_name,
            "total_frames": len(results),
            "frames": results
        }


if __name__ == '__main__':
    import threading, signal

    # 解决 CherryPy autoreload 重启时卡死的问题：
    # 当收到 SIGINT (Ctrl+C) 或 autoreloader 触发重启时，
    # 如果引擎在 5 秒内没有完成停止，强制退出进程。
    def _shutdown_watchdog():
        import time
        # 先等待引擎启动（最多 15 秒）
        for _ in range(30):
            if cherrypy.engine.running:
                break
            time.sleep(0.5)
        while cherrypy.engine.running:
            time.sleep(0.5)
        # engine.running 变为 False，说明正在停止
        time.sleep(5)
        # 5 秒后如果还没退出，强制退出
        os._exit(0)

    _wd = threading.Thread(target=_shutdown_watchdog, daemon=True)
    _wd.start()

    cherrypy.quickstart(Root(), '/', config="server.conf")
else:
    application = cherrypy.Application(Root(), '/', config="server.conf")
