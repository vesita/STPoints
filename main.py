import random
import string

import cherrypy
import os
import json
from jinja2 import Environment, FileSystemLoader
env = Environment(loader=FileSystemLoader('./'))

import os
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
      rawbody = cherrypy.request.body.readline().decode('UTF-8')
      data = json.loads(rawbody)

      for d in data:
        scene = d["scene"]
        frame = d["frame"]
        ann = d["annotation"]
        with open("./data/"+scene +"/label/"+frame+".json",'w') as f:
          json.dump(ann, f, indent=2, sort_keys=True)

      return "ok"


    @cherrypy.expose
    @cherrypy.tools.json_out()
    def cropscene(self):
      rawbody = cherrypy.request.body.readline().decode('UTF-8')
      data = json.loads(rawbody)
      
      rawdata = data["rawSceneId"]

      timestamp = rawdata.split("_")[0]

      print("generate scene")
      log_file = "temp/crop-scene-"+timestamp+".log"

      cmd = "python ./tools/dataset_preprocess/crop_scene.py generate "+ \
        rawdata[0:10]+"/"+timestamp + "_preprocessed/dataset_2hz " + \
        "- " +\
        data["startTime"] + " " +\
        data["seconds"] + " " +\
        "\""+ data["desc"] + "\"" +\
        "> " + log_file + " 2>&1"
      print(cmd)

      code = os.system(cmd)

      with open(log_file) as f:
        log = list(map(lambda s: s.strip(), f.readlines()))

      os.system("rm "+log_file)
      
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
      cl = cherrypy.request.headers['Content-Length']
      rawbody = cherrypy.request.body.readline().decode('UTF-8')
      
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
      rawbody = cherrypy.request.body.readline().decode('UTF-8')
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
              objs = [x for x in map(lambda b: {"category":b["obj_type"], "id": b["obj_id"]}, boxes)]
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

        camera_matrix = calib_data["intrinsic"]
        dist_coeffs = calib_data.get("dist_coeffs")

        result = solve_pnp_ippe(points_3d, points_2d, camera_matrix, dist_coeffs)
        return result

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @cherrypy.tools.json_in()
    def brute_force_pnp(self):
        """穷举 24 种点序排列，返回每种的重投影误差。"""
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

        camera_matrix = calib_data["intrinsic"]
        dist_coeffs = calib_data.get("dist_coeffs")

        print(f"[brute] input points_3d={points_3d}")
        print(f"[brute] input points_2d={points_2d}")

        results = []
        best = {"error": float("inf"), "perm": None}

        for perm in itertools.permutations(range(4)):
            # 固定 3D 点顺序，只置换 2D 点 → 测试不同 3D-2D 配对
            p3d = points_3d
            p2d = [points_2d[i] for i in perm]

            res = solve_pnp_ippe(p3d, p2d, camera_matrix, dist_coeffs)
            err = res.get("reprojection_error", float("inf"))
            results.append({"perm": list(perm), "error": err, "success": res.get("success", False), "extrinsic": res.get("extrinsic")})

            if err < best["error"]:
                best["error"] = err
                best["perm"] = list(perm)
                best["extrinsic"] = res.get("extrinsic")

        # 检查是否所有误差相同
        unique_errors = set(round(r["error"], 2) for r in results)
        print(f"[brute] best={best['perm']}, error={best['error']:.2f}")
        print(f"[brute] unique error values ({len(unique_errors)}): {sorted(unique_errors)}")
        if len(unique_errors) == 1:
            print(f"[brute] 所有 24 种排列误差完全相同 = {results[0]['error']:.2f}px → 3D 点共线退化")
        return {"success": True, "results": results, "best": best}

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @cherrypy.tools.json_in()
    def calib_save(self):
        """Persist extrinsic matrix to calib file on disk."""
        data = cherrypy.request.json
        scene = data["scene"]
        camera = data["camera"]
        extrinsic = data["extrinsic"]

        calib_file = os.path.join("./data", scene, "calib", "camera", camera + ".json")
        if not os.path.isfile(calib_file):
            return {"success": False, "error": f"calib file not found: {calib_file}"}

        with open(calib_file) as f:
            calib_data = json.load(f)

        calib_data["extrinsic"] = extrinsic

        with open(calib_file, "w") as f:
            json.dump(calib_data, f, indent=2)

        return {"success": True}


if __name__ == '__main__':
    cherrypy.quickstart(Root(), '/', config="server.conf")
else:
    application = cherrypy.Application(Root(), '/', config="server.conf")
