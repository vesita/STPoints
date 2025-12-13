#!/usr/bin/env python3
"""
相机内参标定脚本
使用棋盘格图案标定相机内参和畸变系数
"""

import numpy as np
import cv2
import json
import os
from pathlib import Path

try:
    from calibpy.fast_config import get_config
except ImportError:
    from fast_config import get_config


class IntrinsicCalibrator:
    def __init__(self, chessboard_size=(6, 7)):
        """
        初始化内参标定器
        
        Args:
            chessboard_size: 棋盘格内角点数 (宽, 高)
        """
        self.config = get_config()
        self.chessboard_size = chessboard_size
        # 优化角点检测参数，提高收敛速度和精度
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # 生成理论角点坐标（世界坐标系中）
        self.object_points = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        self.object_points[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        
        # 存储检测到的角点
        self.image_points_list = []  # 2D角点在图像中的位置
        self.object_points_list = []  # 3D角点在世界坐标系中的位置
        
    def detect_chessboard_corners(self, image_path):
        """
        检测图像中的棋盘格角点
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            bool: 是否成功检测到角点
            array: 检测到的角点坐标
        """
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            print(f"无法读取图像: {image_path}")
            return False, None
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 查找棋盘格角点，使用优化的参数
        ret, corners = cv2.findChessboardCorners(
            gray, self.chessboard_size, 
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
        )
        
        if ret:
            # 精细化角点位置
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
            return True, corners
        else:
            return False, None
    
    def add_calibration_image(self, image_path):
        """
        添加一张用于标定的图像
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            bool: 是否成功添加
        """
        success, corners = self.detect_chessboard_corners(image_path)
        if success:
            self.image_points_list.append(corners)
            self.object_points_list.append(self.object_points.copy())  # 使用copy避免引用问题
            return True
        else:
            return False
    
    def calibrate(self, image_size):
        """
        执行相机内参标定
        
        Args:
            image_size: 图像尺寸 (width, height)
            
        Returns:
            bool: 标定是否成功
            dict: 标定结果（内参矩阵、畸变系数等）
        """
        if len(self.image_points_list) < 3:
            print("至少需要3张图像才能进行标定")
            return False, None
            
        # 执行标定，使用优化的标志位
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            self.object_points_list, 
            self.image_points_list, 
            image_size, 
            None, 
            None,
            flags=cv2.CALIB_FIX_PRINCIPAL_POINT  # 固定主点以提高稳定性
        )
        
        if ret:
            # 计算重投影误差 - 向量化计算以提高效率
            total_error = 0
            total_points = 0
            
            for i in range(len(self.object_points_list)):
                image_points2, _ = cv2.projectPoints(
                    self.object_points_list[i], 
                    rvecs[i], 
                    tvecs[i], 
                    camera_matrix, 
                    dist_coeffs
                )
                error = cv2.norm(
                    self.image_points_list[i], 
                    image_points2, 
                    cv2.NORM_L2
                )
                total_error += error
                total_points += len(image_points2)
                
            reprojection_error = total_error / total_points
            
            result = {
                'camera_matrix': camera_matrix.tolist(),
                'distortion_coefficients': dist_coeffs.tolist(),
                'reprojection_error': reprojection_error
            }
            
            print(f"标定成功！")
            print(f"重投影误差: {reprojection_error:.4f} 像素")
            print(f"相机内参矩阵:")
            print(camera_matrix)
            print(f"畸变系数:")
            print(dist_coeffs)
            
            return True, result
        else:
            print("标定失败")
            return False, None
    
    def save_calibration(self, result, output_path):
        """
        保存标定结果到配置文件
        
        Args:
            result: 标定结果
            output_path: 输出文件路径
        """
        # 读取现有配置
        config_path = "config/config.json"
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_data = json.load(f)
        else:
            config_data = {}
            
        # 更新相机内参和畸变系数
        config_data['camera_intrinsic'] = result['camera_matrix']
        config_data['camera_distortion_coefficients'] = result['distortion_coefficients']
        
        # 保存到输出文件
        with open(output_path, 'w') as f:
            json.dump(config_data, f, indent=2)
            
        print(f"标定结果已保存到 {output_path}")


def main():
    """
    主函数 - 执行相机内参标定
    """
    print("开始相机内参标定...")
    
    # 初始化标定器
    calibrator = IntrinsicCalibrator(chessboard_size=(6, 7))
    
    # 获取配置中的图像路径
    config = get_config()
    
    # 缓存样本图像，避免重复读取
    sample_image = None
    sample_image_path = None
    
    # 如果配置中有单个图像文件
    if config.image and os.path.exists(config.image):
        success = calibrator.add_calibration_image(config.image)
        if not success:
            print("无法从配置中获取有效的图像")
            return
        sample_image_path = config.image
    
    # 如果配置指向目录，查找其中的图像文件
    elif config.target_path and os.path.isdir(config.target_path):
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_count = 0
        
        # 遍历目录中的图像文件
        for filename in os.listdir(config.target_path):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_path = os.path.join(config.target_path, filename)
                if calibrator.add_calibration_image(image_path):
                    image_count += 1
                    # 保存第一个成功的图像作为样本
                    if sample_image is None:
                        sample_image_path = image_path
                else:
                    print(f"跳过无法检测到棋盘格的图像: {filename}")
                    
        if image_count == 0:
            print("未找到可用于标定的图像")
            return
        else:
            print(f"成功加载 {image_count} 张图像用于标定")
    
    else:
        print("配置中未提供有效的图像路径")
        return
    
    # 获取图像尺寸（使用缓存的样本图像）
    if sample_image_path:
        sample_image = cv2.imread(sample_image_path)
    if sample_image is None:
        print("无法读取样本图像以获取尺寸")
        return
        
    image_size = (sample_image.shape[1], sample_image.shape[0])  # (width, height)
    print(f"图像尺寸: {image_size[0]}x{image_size[1]}")
    
    # 执行标定
    success, result = calibrator.calibrate(image_size)
    if success:
        # 保存结果
        output_path = "config/calibrated_config.json"
        calibrator.save_calibration(result, output_path)
        print("相机内参标定完成!")
    else:
        print("相机内参标定失败!")


if __name__ == "__main__":
    main()