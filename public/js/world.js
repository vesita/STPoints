/*
世界坐标系统管理模块
用于管理3D点云标注中的坐标变换、场景数据加载等核心功能
*/

import * as THREE from './lib/three.module.js';

import {RadarManager} from "./radar.js"
import {AuxLidarManager} from "./aux_lidar.js"
import {Lidar} from "./lidar.js"
import {Annotation} from "./annotation.js"
import {EgoPose} from "./ego_pose.js"
import {logger} from "./log.js"
import { euler_angle_to_rotate_matrix, euler_angle_to_rotate_matrix_3by3, matmul, matmul2 , mat} from './util.js';

/**
 * 帧信息类 - 存储单个帧的数据信息和元数据
 * @param {*} data - 数据对象
 * @param {*} sceneMeta - 场景元数据
 * @param {*} sceneName - 场景名称
 * @param {*} frame - 帧标识
 */
function FrameInfo(data, sceneMeta, sceneName, frame){
    
    this.data = data;
    this.sceneMeta = sceneMeta;
    this.dir = ""; // 目录路径
    this.scene = sceneName; // 场景名称
    this.frame = frame; // 帧标识
    this.pcd_ext = ""; // 点云文件扩展名
    this.frame_index = this.sceneMeta.frames.findIndex(function(x){return x==frame;}), // 帧索引
    this.transform_matrix = this.sceneMeta.point_transform_matrix, // 点变换矩阵
    this.annotation_format = this.sceneMeta.boxtype, // 标注格式 (xyz或psr)

    /**
     * 获取点云文件路径
     * @returns {string} 点云文件路径
     */
    this.get_pcd_path = function(){
            return 'data/'+ this.scene + "/lidar/" + this.frame + this.sceneMeta.lidar_ext;
        };
    
    /**
     * 获取雷达数据路径
     * @param {string} name - 雷达名称
     * @returns {string} 雷达数据路径
     */
    this.get_radar_path = function(name){
        return `data/${this.scene}/radar/${name}/${this.frame}${this.sceneMeta.radar_ext}`;
    };
    
    /**
     * 获取辅助激光雷达数据路径
     * @param {string} name - 辅助激光雷达名称
     * @returns {string} 辅助激光雷达数据路径
     */
    this.get_aux_lidar_path = function(name){
        return `data/${this.scene}/aux_lidar/${name}/${this.frame}${this.sceneMeta.radar_ext}`;
    }
    
    /**
     * 获取标注文件路径
     * @returns {string} 标注文件路径
     */
    this.get_anno_path = function(){
            if (this.annotation_format=="psr"){
                return 'data/'+this.scene + "/label/" + this.frame + ".json";
            }
            else{
                return 'data/'+this.scene + "/bbox.xyz/" + this.frame + ".bbox.txt";
            }
            
        };
    
    /**
     * 将标注文本转换为边界框对象
     * @param {string} text - 标注文本内容
     * @returns {*} 边界框对象数组
     */
    this.anno_to_boxes = function(text){
            var _self = this;
            if (this.annotation_format == "psr"){
                var boxes = JSON.parse(text);
                return boxes;
            }
            else
                return this.python_xyz_to_psr(text);
        };
    
    /**
     * 坐标变换函数 - 使用变换矩阵对点进行变换
     * @param {*} m - 变换矩阵
     * @param {*} x - X坐标
     * @param {*} y - Y坐标
     * @param {*} z - Z坐标
     * @returns {Array} 变换后的坐标[x, y, z]
     */
    this.transform_point = function(m, x,y, z){
            var rx = x*m[0]+y*m[1]+z*m[2];
            var ry = x*m[3]+y*m[4]+z*m[5];
            var rz = x*m[6]+y*m[7]+z*m[8];
    
            return [rx, ry, rz];
        };
    
    /*
    将Python格式的顶点坐标转换为PSR（位置、尺寸、旋转）格式
    输入是8个顶点的坐标:
    底部左前, 底部右前, 底部右后, 底部左后
    顶部左前, 顶部右前, 顶部右后, 顶部左后

    这种格式是SECOND/PointRcnn等目标检测算法使用的输出格式
    */
    this.python_xyz_to_psr = function(text){
            var _self = this;
    
            // 将文本按行分割并解析为浮点数数组
            var points_array = text.split('\n').filter(function(x){return x;}).map(function(x){return x.split(' ').map(function(x){return parseFloat(x);})})
            
            // 对每个点应用坐标变换
            var boxes = points_array.map(function(ps){
                for (var i=0; i<8; i++){
                    var p = _self.transform_point(_self.transform_matrix, ps[3*i+0],ps[3*i+1],ps[3*i+2]);
                    ps[i*3+0] = p[0];
                    ps[i*3+1] = p[1];
                    ps[i*3+2] = p[2];                
                }
                return ps;
            });
            
            // 将XYZ格式转换为PSR格式
            var boxes_ann = boxes.map(this.xyz_to_psr);
    
            return boxes_ann;
        };

    /**
     * 将XYZ顶点坐标转换为PSR（位置、尺寸、旋转）表示
     * @param {*} ann_input - 输入的顶点坐标数组
     * @returns {*} PSR格式的对象{position, scale, rotation}
     */
    this.xyz_to_psr = function(ann_input){
            var ann = [];
            // 处理输入数据格式
            if (ann_input.length==24)
                ann = ann_input;
            else
                for (var i = 0; i<ann_input.length; i++){
                    if ((i+1) % 4 != 0){
                        ann.push(ann_input[i]);
                    }
                }

            // 计算中心点坐标（8个顶点的平均值）
            var pos={x:0,y:0,z:0};
            for (var i=0; i<8; i++){
                pos.x+=ann[i*3];
                pos.y+=ann[i*3+1];
                pos.z+=ann[i*3+2];
            }
            pos.x /=8;
            pos.y /=8;
            pos.z /=8;

            // 计算尺寸（长宽高）
            var scale={
                x: Math.sqrt((ann[0]-ann[3])*(ann[0]-ann[3])+(ann[1]-ann[4])*(ann[1]-ann[4])), // 长度
                y: Math.sqrt((ann[0]-ann[9])*(ann[0]-ann[9])+(ann[1]-ann[10])*(ann[1]-ann[10])), // 宽度
                z: ann[14]-ann[2], // 高度
            };
            
            /*
            1. atan2(y,x)而不是atan2(x,y)
            2. XY平面上的点顺序
                0   1
                3   2
            */

            // 计算绕Z轴的旋转角度
            var angle = Math.atan2(ann[4]+ann[7]-2*pos.y, ann[3]+ann[6]-2*pos.x);

            // 返回PSR格式的对象
            return {
                position: pos,      // 位置
                scale:scale,        // 尺寸
                rotation:{x:0,y:0,z:angle}, // 旋转
            }
        };
}

/**
 * 图像管理类 - 管理场景中的多摄像头图像数据
 * @param {*} sceneMeta - 场景元数据
 * @param {*} sceneName - 场景名称
 * @param {*} frame - 帧标识
 */
function Images(sceneMeta, sceneName, frame){
    /**
     * 检查所有图像是否已加载完成
     * @returns {boolean} 是否全部加载完成
     */
    this.loaded = function(){
        for (var n in this.names){
            if (!this.loaded_flag[this.names[n]])
                return false;
        }

        return true;
    };

    this.names = sceneMeta.camera; // 摄像头名称列表 ["image","left","right"]
    this.loaded_flag = {}; // 各图像加载状态标记
    this.content = {}; // 图像内容存储
    this.on_all_loaded = null; // 全部加载完成回调函数

    /**
     * 根据名称获取图像
     * @param {string} name - 图像名称
     * @returns {*} 图像对象
     */
    this.getImageByName = function(name){
        return this.content[name];
    };

    /**
     * 加载图像数据
     * @param {*} on_all_loaded - 全部加载完成的回调函数
     * @param {*} active_name - 当前激活的摄像头名称
     */
    this.load = function(on_all_loaded, active_name){
        this.on_all_loaded = on_all_loaded;
        
        var _self = this;

        // 如果存在摄像头定义，则逐个加载图像
        if (this.names){
            this.names.forEach(function(cam){
                _self.content[cam] = new Image();
                // 设置图像加载完成事件处理
                _self.content[cam].onload= function(){ 
                    _self.loaded_flag[cam] = true;
                    _self.on_image_loaded();
                };
                // 设置图像加载错误事件处理
                _self.content[cam].onerror=function(){ 
                    _self.loaded_flag[cam] = true;
                    _self.on_image_loaded();
                };

                // 设置图像源路径并开始加载
                _self.content[cam].src = 'data/'+sceneName+'/camera/' + cam + '/'+ frame + sceneMeta.camera_ext;
                console.log("image set")
            });
        }
    },

    /**
     * 图像加载事件处理函数
     * 当一张图像加载完成后检查是否所有图像都已加载
     */
    this.on_image_loaded = function(){
        if (this.loaded()){
            this.on_all_loaded();
        }
    }
}

/**
 * 世界坐标系统类 - 管理整个3D场景的坐标系统和数据加载
 * @param {*} data - 数据对象
 * @param {*} sceneName - 场景名称
 * @param {*} frame - 帧标识
 * @param {*} coordinatesOffset - 坐标偏移量
 * @param {*} on_preload_finished - 预加载完成回调函数
 */
function World(data, sceneName, frame, coordinatesOffset, on_preload_finished){
    this.data = data; // 数据对象
    this.sceneMeta = this.data.getMetaBySceneName(sceneName); // 获取场景元数据
    this.frameInfo = new FrameInfo(this.data, this.sceneMeta, sceneName, frame); // 帧信息对象

    this.coordinatesOffset = coordinatesOffset; // 坐标偏移量

    /**
     * 转换为字符串表示
     * @returns {string} 场景和帧的组合字符串
     */
    this.toString = function(){
        return this.frameInfo.scene + "," + this.frameInfo.frame;
    }
    
    // 各种数据管理器实例
    this.cameras = new Images(this.sceneMeta, sceneName, frame); // 图像管理器
    this.radars = new RadarManager(this.sceneMeta, this, this.frameInfo); // 雷达管理器
    this.lidar = new Lidar(this.sceneMeta, this, this.frameInfo); // 激光雷达管理器
    this.annotation = new Annotation(this.sceneMeta, this, this.frameInfo); // 标注管理器
    this.aux_lidars = new AuxLidarManager(this.sceneMeta, this, this.frameInfo); // 辅助激光雷达管理器
    this.egoPose = new EgoPose(this.sceneMeta, this, this.FrameInfo); // 自车姿态管理器

    // 世界状态标志
    this.points_loaded = false, // 点云是否加载完成

    /**
     * 检查是否预加载完成
     * @returns {boolean} 是否所有子项都预加载完成
     */
    this.preloaded=function(){
        return this.lidar.preloaded && 
               this.annotation.preloaded && 
               this.aux_lidars.preloaded() && 
               this.radars.preloaded()&&
               this.egoPose.preloaded;
    };

    this.create_time = 0; // 创建时间戳
    this.finish_time = 0; // 完成时间戳
    this.on_preload_finished = null; // 预加载完成回调函数
    
    /**
     * 子项预加载完成处理函数
     * @param {*} on_preload_finished - 预加载完成回调函数
     */
    this.on_subitem_preload_finished = function(on_preload_finished){
        if (this.preloaded()){
            
            logger.log(`finished preloading ${this.frameInfo.scene} ${this.frameInfo.frame}`);

            this.calcTransformMatrix(); // 计算变换矩阵

            // 如果设置了预加载完成回调函数，则执行
            if (this.on_preload_finished){
                this.on_preload_finished(this);                
            }

            // 如果当前世界处于激活状态，则开始渲染
            if (this.active){
                this.go();
            } 
        }
    };

    /**
     * 计算坐标变换矩阵
     * 包括激光雷达到世界坐标系、世界坐标系到场景坐标系等各种变换矩阵
     */
    this.calcTransformMatrix = function()
    {
        // 如果存在自车姿态数据
        if (this.egoPose.egoPose){
                let thisPose = this.egoPose.egoPose; // 当前姿态
                let refPose = this.data.getRefEgoPose(this.frameInfo.scene, thisPose); // 参考姿态
                
                // 计算当前姿态的角度（转为弧度）
                let thisRot = {
                    x: thisPose.pitch * Math.PI/180.0,  // 俯仰角
                    y: thisPose.roll * Math.PI/180.0,   // 翻滚角              
                    z: - thisPose.azimuth * Math.PI/180.0 // 方位角
                };
    
                // 计算相对于参考姿态的位置差值
                let posDelta = {
                    x: thisPose.x - refPose.x,
                    y: thisPose.y - refPose.y,
                    z: thisPose.z - refPose.z,
                };
    
                // 构建激光雷达到自车坐标系的变换矩阵
                let trans_lidar_ego = new THREE.Matrix4().makeRotationFromEuler(new THREE.Euler(0,0,Math.PI, "ZYX"))
                                                         .setPosition(0, 0, 0.4);
    
                // 构建自车坐标系到UTM坐标系的变换矩阵
                let trans_ego_utm = new THREE.Matrix4().makeRotationFromEuler(new THREE.Euler(thisRot.x, thisRot.y, thisRot.z, "ZXY"))
                                                       .setPosition(posDelta.x, posDelta.y, posDelta.z);    
                
                // 构建UTM坐标系到场景坐标系的变换矩阵（基于坐标偏移量）
                let trans_utm_scene = new THREE.Matrix4().identity().setPosition(this.coordinatesOffset[0], this.coordinatesOffset[1], this.coordinatesOffset[2]);
    
                // 计算激光雷达到UTM坐标系的总变换矩阵
                this.trans_lidar_utm = new THREE.Matrix4().multiplyMatrices(trans_ego_utm, trans_lidar_ego);

                // 根据配置决定使用哪种坐标系统
                if (this.data.cfg.coordinateSystem == "utm")
                    this.trans_lidar_scene = new THREE.Matrix4().multiplyMatrices(trans_utm_scene, this.trans_lidar_utm);
                else
                    this.trans_lidar_scene = trans_utm_scene;  //只进行偏移

                // 计算逆变换矩阵
                this.trans_utm_lidar = new THREE.Matrix4().copy(this.trans_lidar_utm).invert();
                this.trans_scene_lidar = new THREE.Matrix4().copy(this.trans_lidar_scene).invert();
            }
            else
            {
                // 如果没有姿态数据，则使用恒等变换和偏移
                let trans_utm_scene = new THREE.Matrix4().identity().setPosition(this.coordinatesOffset[0], this.coordinatesOffset[1], this.coordinatesOffset[2]);
                let id = new THREE.Matrix4().identity();

                this.trans_lidar_utm = id;
                this.trans_lidar_scene = trans_utm_scene;
                
                this.trans_utm_lidar = new THREE.Matrix4().copy(this.trans_lidar_utm).invert();
                this.trans_scene_lidar = new THREE.Matrix4().copy(this.trans_lidar_scene).invert();
            }

            // 设置WebGL组的变换矩阵
            this.webglGroup.matrix.copy(this.trans_lidar_scene);
            this.webglGroup.matrixAutoUpdate = false;
    };

    /**
     * 将场景坐标转换为激光雷达坐标
     * @param {*} pos - 场景坐标位置
     * @returns {*} 激光雷达坐标位置
     */
    this.scenePosToLidar = function(pos)
    {
        let tp = new THREE.Vector4(pos.x, pos.y, pos.z, 1).applyMatrix4(this.trans_scene_lidar);
        return tp;        
    }

    /**
     * 将激光雷达坐标转换为场景坐标
     * @param {*} pos - 激光雷达坐标位置
     * @returns {*} 场景坐标位置
     */
    this.lidarPosToScene = function(pos)
    {
        let tp = new THREE.Vector3(pos.x, pos.y, pos.z).applyMatrix4(this.trans_lidar_scene);
        return tp;        
    }

    /**
     * 将激光雷达坐标转换为UTM坐标
     * @param {*} pos - 激光雷达坐标位置
     * @returns {*} UTM坐标位置
     */
    this.lidarPosToUtm = function(pos)
    {
        let tp = new THREE.Vector3(pos.x, pos.y, pos.z).applyMatrix4(this.trans_lidar_utm);
        return tp;        
    }

    /**
     * 将场景旋转转换为激光雷达旋转
     * @param {*} rotEuler - 场景欧拉角旋转
     * @returns {*} 激光雷达欧拉角旋转
     */
    this.sceneRotToLidar = function(rotEuler)
    {
        if (!rotEuler.isEuler)
        {
            rotEuler = new THREE.Euler(rotEuler.x, rotEuler.y, rotEuler.z, "XYZ");
        }

        let rotG = new THREE.Quaternion().setFromEuler(rotEuler);
        let GlobalToLocalRot = new THREE.Quaternion().setFromRotationMatrix(this.trans_scene_lidar);

        let retQ = rotG.multiply(GlobalToLocalRot);

        let retEuler = new THREE.Euler().setFromQuaternion(retQ, rotEuler.order);

        return retEuler;
    }

    /**
     * 将激光雷达旋转转换为场景旋转
     * @param {*} rotEuler - 激光雷达欧拉角旋转
     * @returns {*} 场景欧拉角旋转
     */
    this.lidarRotToScene = function(rotEuler)
    {
        if (!rotEuler.isEuler)
        {
            rotEuler = new THREE.Euler(rotEuler.x, rotEuler.y, rotEuler.z, "XYZ");
        }

        let rotL = new THREE.Quaternion().setFromEuler(rotEuler);
        let localToGlobalRot = new THREE.Quaternion().setFromRotationMatrix(this.trans_lidar_scene)

        let retQ = rotL.multiply(localToGlobalRot);

        let retEuler = new THREE.Euler().setFromQuaternion(retQ, rotEuler.order);

        return retEuler;
    }

    /**
     * 将激光雷达旋转转换为UTM旋转
     * @param {*} rotEuler - 激光雷达欧拉角旋转
     * @returns {*} UTM欧拉角旋转
     */
    this.lidarRotToUtm = function(rotEuler)
    {
        if (!rotEuler.isEuler)
        {
            rotEuler = new THREE.Euler(rotEuler.x, rotEuler.y, rotEuler.z, "XYZ");
        }

        let rotL = new THREE.Quaternion().setFromEuler(rotEuler);
        let localToGlobalRot = new THREE.Quaternion().setFromRotationMatrix(this.trans_lidar_utm)

        let retQ = rotL.multiply(localToGlobalRot);

        let retEuler = new THREE.Euler().setFromQuaternion(retQ, rotEuler.order);

        return retEuler;
    }

    /**
     * 将UTM旋转转换为激光雷达旋转
     * @param {*} rotEuler - UTM欧拉角旋转
     * @returns {*} 激光雷达欧拉角旋转
     */
    this.utmRotToLidar = function(rotEuler)
    {
        if (!rotEuler.isEuler)
        {
            rotEuler = new THREE.Euler(rotEuler.x, rotEuler.y, rotEuler.z, "XYZ");
        }

        let rot = new THREE.Quaternion().setFromEuler(rotEuler);
        let trans = new THREE.Quaternion().setFromRotationMatrix(this.trans_utm_lidar);

        let retQ = rot.multiply(trans);

        let retEuler = new THREE.Euler().setFromQuaternion(retQ, rotEuler.order);

        return retEuler;
    }

    /**
     * 预加载函数 - 初始化并预加载所有场景数据
     * @param {*} on_preload_finished - 预加载完成回调函数
     */
    this.preload=function(on_preload_finished){
        this.create_time = new Date().getTime();
        console.log(this.create_time, sceneName, frame, "start");

        // 创建WebGL组用于组织场景对象
        this.webglGroup = new THREE.Group();
        this.webglGroup.name = "world";
        
        // 定义预加载完成回调函数
        let _preload_cb = ()=>this.on_subitem_preload_finished(on_preload_finished);

        // 启动各个模块的预加载过程
        this.lidar.preload(_preload_cb);
        this.annotation.preload(_preload_cb)
        this.radars.preload(_preload_cb);
        this.cameras.load(_preload_cb, this.data.active_camera_name);
        this.aux_lidars.preload(_preload_cb);
        this.egoPose.preload(_preload_cb);        
    };

    this.scene = null, // THREE场景对象
    this.destroy_old_world = null, // 销毁旧世界的函数
    this.on_finished = null, // 完成回调函数
    
    /**
     * 激活世界 - 将当前世界添加到场景中并开始渲染
     * @param {*} scene - THREE场景对象
     * @param {*} destroy_old_world - 销毁旧世界的函数
     * @param {*} on_finished - 完成回调函数
     */
    this.activate=function(scene, destroy_old_world, on_finished){
        this.scene = scene;
        this.active = true;
        this.destroy_old_world = destroy_old_world;
        this.on_finished = on_finished;
        if (this.preloaded()){
            this.go();
        }
    };

    this.active = false, // 是否处于激活状态
    this.everythingDone = false; // 是否已完成所有加载和初始化
    
    /**
     * 开始渲染 - 将所有加载的数据添加到场景中
     */
    this.go=function(){

        if (this.everythingDone){
            // 如果已经完成，直接调用完成回调
            if (this.on_finished){
                this.on_finished();
            }
            return;
        }

        // 检查是否预加载完成
        if (this.preloaded()){
            // 如果需要销毁旧世界，则执行
            if (this.destroy_old_world){
                this.destroy_old_world();
            }

            // 检查是否已被销毁
            if (this.destroyed){
                console.log("go after destroyed.");
                this.unload();
                return;
            }

            // 将WebGL组添加到场景中
            this.scene.add(this.webglGroup);
            
            // 启动各个模块的渲染
            this.lidar.go(this.scene);
            this.annotation.go(this.scene);
            this.radars.go(this.scene);            
            this.aux_lidars.go(this.scene);

            // 记录完成时间并输出日志
            this.finish_time = new Date().getTime();
            console.log(this.finish_time, sceneName, frame, "loaded in ", this.finish_time - this.create_time, "ms");
                
            // 渲染在on_finished()回调中调用
            if (this.on_finished){
                this.on_finished();
            }

            this.everythingDone = true;
        }
    };

    /**
     * 添加线段到场景中
     * @param {*} start - 起点坐标
     * @param {*} end - 终点坐标
     * @param {*} color - 线段颜色
     */
    this.add_line=function(start, end, color){
        var line = this.new_line(start, end, color);
        this.scene.add(line);
    };

    /**
     * 创建新的线段对象
     * @param {*} start - 起点坐标
     * @param {*} end - 终点坐标
     * @param {*} color - 线段颜色
     * @returns {*} THREE线段对象
     */
    this.new_line=function(start, end, color){

        var vertex = start.concat(end); // 合并起点和终点坐标
        this.world.data.dbg.alloc();
        var line = new THREE.BufferGeometry();
        line.addAttribute( 'position', new THREE.Float32BufferAttribute(vertex, 3 ) );
        
        // 设置默认颜色
        if (!color){
            color = 0x00ff00;
        }
        var material = new THREE.LineBasicMaterial( { color: color, linewidth: 1, opacity: this.data.cfg.box_opacity, transparent: true } );
        return new THREE.LineSegments( line, material );                
    };

    this.destroyed = false; // 是否已被销毁

    /**
     * 卸载世界 - 从场景中移除所有对象但不销毁
     */
    this.unload = function(){
        if (this.everythingDone){
            // 从场景卸载所有内容
            this.lidar.unload();
            this.radars.unload();
            this.aux_lidars.unload();
            this.annotation.unload();

            this.scene.remove(this.webglGroup);
            
            this.active = false;
            this.everythingDone = false;
        }
    };

    /**
     * 删除所有内容 - 彻底清理和销毁所有资源
     */
    this.deleteAll = function(){
        var _self= this;

        logger.log(`delete world ${this.frameInfo.scene},${this.frameInfo.frame}`);

        if (this.everythingDone){
            this.unload();
        }
        
        // 删除所有管理器的内容
        this.lidar.deleteAll();
        this.radars.deleteAll();
        this.aux_lidars.deleteAll();
        this.annotation.deleteAll();

        this.destroyed = true;
        console.log(this.frameInfo.scene, this.frameInfo.frame, "destroyed");
    };

    // 启动预加载过程
    this.preload(on_preload_finished);  
}

export {World};