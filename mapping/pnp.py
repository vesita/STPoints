import cv2
import numpy as np
import glob
import os
import argparse
import yaml
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 参考https://blog.csdn.net/qq_43508270/article/details/147984900?fromshare=blogdetail&sharetype=blogdetail&sharerId=147984900&sharerefer=PC&sharesource=Rea_one&sharefrom=from_link

# 线程锁用于保护共享资源
lock = threading.Lock()

def process_image(fname, chessboard_size, flags, criteria):
    """
    处理单张图像，检测棋盘格角点
    
    Args:
        fname: 图像文件路径
        chessboard_size: 棋盘格尺寸
        flags: 角点检测标志
        criteria: 亚像素优化参数
    
    Returns:
        tuple: (ret, objp, corners_refined, gray_shape) 或 None
    """
    img = cv2.imread(fname)
    if img is None:
        print(f"警告: 无法读取图像 {fname}")
        return None
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_shape = gray.shape
    
    # 尝试查找棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, flags)
    
    # 如果没找到，尝试其他方法
    if not ret:
        # 应用直方图均衡化再尝试
        gray_eq = cv2.equalizeHist(gray)
        ret, corners = cv2.findChessboardCorners(gray_eq, chessboard_size, flags)
        
        if not ret:
            return None
    
    # 亚像素角点优化
    corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    
    return (ret, fname, corners_refined, gray_shape)

def calibrate_camera(image_path_pattern, chessboard_size, square_size, max_workers=4):
    # 棋盘格参数设置
    CHESSBOARD_SIZE = chessboard_size
    SQUARE_SIZE = square_size         # 棋盘格方块实际尺寸（米）

    # 生成真实世界坐标点
    objp = np.zeros((CHESSBOARD_SIZE[0]*CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2) * SQUARE_SIZE

    # 存储3D点和2D点
    obj_points = []   # 真实世界3D点
    img_points = []   # 图像平面2D点

    # 读取标定图像
    images = glob.glob(image_path_pattern)

    if len(images) == 0:
        print(f"错误: 在 '{image_path_pattern}' 中没有找到图像文件")
        return None

    print(f"找到 {len(images)} 张图像用于标定")

    # 角点检测参数
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # 不同的棋盘格检测标志
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    
    # 用于获取图像尺寸的灰度图
    gray_shape = None

    print("开始处理图像...")
    # 使用多线程处理图像
    successful_images = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_image = {
            executor.submit(process_image, fname, CHESSBOARD_SIZE, flags, criteria): fname 
            for fname in images
        }
        
        # 收集结果
        for future in tqdm(as_completed(future_to_image), total=len(images), desc="处理图像"):
            result = future.result()
            if result is not None:
                ret, fname, corners_refined, img_gray_shape = result
                successful_images.append((fname, corners_refined, img_gray_shape))
                
                # 设置图像尺寸（只需要一次）
                if gray_shape is None:
                    gray_shape = img_gray_shape

    if len(successful_images) == 0:
        print("错误: 没有在任何图像中检测到棋盘格角点")
        print("请检查以下几点:")
        print("1. 图像是否包含指定大小的棋盘格图案 ({0}x{1})".format(CHESSBOARD_SIZE[0], CHESSBOARD_SIZE[1]))
        print("2. 棋盘格尺寸参数是否正确")
        print("3. 图像质量是否足够好")
        print("4. 是否提供了正确的图像路径")
        return None

    print(f"成功使用 {len(successful_images)} 张图像进行标定")
    
    # 准备标定点数据
    for fname, corners_refined, _ in successful_images:
        obj_points.append(objp.copy())
        img_points.append(corners_refined)
    
    # 添加进度提示，因为calibrateCamera可能需要很长时间
    print("开始执行相机标定，请稍候...")
    
    # 执行相机标定
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, 
        img_points, 
        gray_shape[::-1], 
        None, 
        None
    )
    
    print("相机标定完成！")

    # 计算重投影误差
    print("计算重投影误差...")
    mean_error = 0
    # 使用tqdm显示重投影误差计算进度
    for i in tqdm(range(len(obj_points)), desc="计算重投影误差"):
        imgpoints2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(img_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error

    print("\n内参矩阵:")
    print(mtx)
    print("\n畸变系数:")
    print(dist)
    print(f"\n平均重投影误差: {mean_error/len(obj_points):.6f} 像素")

    return {
        'camera_matrix': mtx,
        'distortion_coefficients': dist,
        'rotation_vectors': rvecs,
        'translation_vectors': tvecs,
        'reprojection_error': mean_error/len(obj_points),
        'image_size': gray_shape
    }

def save_calibration_ros_format(calib_result, output_filename, camera_name="camera"):
    """
    将相机标定结果保存为ROS兼容的YAML格式
    
    Args:
        calib_result: 相机标定结果字典
        output_filename: 输出文件名
        camera_name: 相机名称
    """
    mtx = calib_result['camera_matrix']
    dist = calib_result['distortion_coefficients'].flatten()
    image_size = calib_result['image_size']
    
    # 创建ROS兼容的YAML格式
    ros_calib = {
        'image_width': int(image_size[1]),
        'image_height': int(image_size[0]),
        'camera_name': camera_name,
        'camera_matrix': {
            'rows': 3,
            'cols': 3,
            'data': [float(x) for x in mtx.flatten().tolist()]
        },
        'distortion_model': 'plumb_bob',  # ROS中radtan模型的名称
        'distortion_coefficients': {
            'rows': 1,
            'cols': len(dist),
            'data': [float(x) for x in dist.tolist()]
        },
        'rectification_matrix': {
            'rows': 3,
            'cols': 3,
            'data': [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]  # 单位矩阵
        },
        'projection_matrix': {
            'rows': 3,
            'cols': 4,
            'data': [
                float(mtx[0, 0]), 0.0, float(mtx[0, 2]), 0.0,
                0.0, float(mtx[1, 1]), float(mtx[1, 2]), 0.0,
                0.0, 0.0, 1.0, 0.0
            ]
        }
    }
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_filename) if os.path.dirname(output_filename) else '.', exist_ok=True)
    
    # 写入YAML文件
    with open(output_filename, 'w') as f:
        f.write("%YAML:1.0\n")
        yaml.dump(ros_calib, f, default_flow_style=False, sort_keys=False)
    
    print(f"ROS格式的标定文件已保存至: {output_filename}")

if __name__ == "__main__":
    image_path_pattern = 'data/calibration/camera/image/*.jpg'
    chessboard_size = (6, 7)
    square_size = 0.025
    # 将输出文件保存到messages目录
    output_filename = 'messages/camera_calib_ros.yaml'
    camera_name = 'camera'
    
    result = calibrate_camera(
        image_path_pattern,
        chessboard_size,
        square_size,
    )
    
    if result is not None:
        print("\n相机标定完成!")
        # 保存为ROS格式
        save_calibration_ros_format(result, output_filename, camera_name)
    else:
        print("\n相机标定失败!")