#!/usr/bin/env python3
"""
相机内参标定脚本
使用棋盘格图案标定相机内参和畸变系数
"""

import cv2
import numpy as np
import json
import os
from glob import glob
from tqdm import tqdm

try:
    from calibpy.fast_config import get_config
except ImportError:
    from fast_config import get_config


class IntrinsicCalibrator:
    def __init__(self, inside_corner_size=(6, 7), square_size=100):
        """
        初始化内参标定器
        
        Args:
            inside_corner_size: 内部角点数量 (行数, 列数)
            square_size: 棋盘格方格大小 (m)
        """
        self.chessboard_size = inside_corner_size
        self.square_size = square_size
        
        # 创建对象点（世界坐标系中的点）
        self.object_points = np.zeros((inside_corner_size[0] * inside_corner_size[1], 3), np.float32)
        self.object_points[:, :2] = np.mgrid[0:inside_corner_size[0], 0:inside_corner_size[1]].T.reshape(-1, 2)
        self.object_points *= square_size  # 乘以方格实际大小
        
        # 存储检测到的角点和对应的对象点
        self.image_points_list = []  # 2D角点在图像中的位置
        self.object_points_list = []  # 3D角点在世界坐标系中的位置
        self.valid_images = []  # 有效图像路径列表

    def detect_chessboard_corners(self, image_path):
        """
        检测图像中的棋盘格角点
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            tuple: (是否成功检测, 角点坐标)
        """
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            print(f"无法读取图像: {image_path}")
            return False, None
            
        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 查找棋盘格角点
        ret, corners = cv2.findChessboardCorners(
            gray, self.chessboard_size, 
            None
        )
        
        if ret:
            # 精细化角点位置
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
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
            self.object_points_list.append(self.object_points)
            self.valid_images.append(image_path)
            return True
        else:
            return False

    def calibrate(self, image_size, fix_principal_point=True, use_rational_model=True):
        """
        执行相机内参标定
        
        Args:
            image_size: 图像尺寸 (width, height)
            fix_principal_point: 是否固定主点
            use_rational_model: 是否使用有理模型
            
        Returns:
            bool: 标定是否成功
            dict: 标定结果（内参矩阵、畸变系数等）
        """
        image_count = len(self.image_points_list)
        if image_count < 3:  # 至少需要3张图像
            print("至少需要3张图像才能进行标定，当前有效图像数量:", image_count)
            return False, None
            
        print(f"开始标定，使用 {image_count} 张图像")
        
        # 执行标定
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            self.object_points_list, 
            self.image_points_list, 
            image_size, 
            None,
            None
        )
        
        if ret:
            # 计算每张图像的重投影误差
            total_error = 0.0
            # 使用tqdm显示重投影误差计算进度
            for i in tqdm(range(len(self.object_points_list)), desc="计算重投影误差"):
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
                ) / len(image_points2)
                total_error += error
            
            mean_reprojection_error = total_error / len(self.object_points_list)
            
            # 处理畸变系数
            dist_coeffs_processed = dist_coeffs.flatten()
            
            result = {
                'camera_matrix': camera_matrix.flatten().tolist(),
                'distortion_coefficients': dist_coeffs_processed.tolist(),
                'reprojection_error': float(mean_reprojection_error),
                'image_count': image_count,
                'fix_principal_point': fix_principal_point,
                'use_rational_model': use_rational_model
            }
            
            print(f"标定成功！")
            print(f"使用的图像数量: {image_count}")
            print(f"平均重投影误差: {mean_reprojection_error:.4f} 像素")
            if fix_principal_point:
                print("使用固定主点模式")
            if use_rational_model:
                print("使用有理畸变模型")
            print(f"相机内参矩阵:")
            print(camera_matrix)
            print(f"畸变系数:")
            print(dist_coeffs_processed)
            
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
        # 确保畸变系数为一维数组格式
        config_data['camera_distortion_coefficients'] = result['distortion_coefficients']
        
        # 保存到输出文件
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(config_data, f, indent=2)
            
        print(f"标定结果已保存到 {output_path}")

    def evaluate_intrinsic_effect(self, image_path=None, camera_matrix=None, dist_coeffs=None):
        """
        评估内参标定效果
        
        Args:
            image_path: 图像路径，如果为None则随机选择一张用于标定的图像
            camera_matrix: 相机内参矩阵，如果为None则使用类中存储的结果
            dist_coeffs: 畸变系数，如果为None则使用类中存储的结果
            
        Returns:
            tuple: (原始图像, 矫正后的图像)
        """
        import random
        
        # 如果没有提供图像路径，则随机选择一张用于标定的图像
        if image_path is None:
            if not self.valid_images:
                raise ValueError("没有可用的图像用于评估，请先添加标定图像")
            image_path = random.choice(self.valid_images)
            print(f"随机选择图像: {image_path}")
        else:
            print(f"使用指定图像: {image_path}")
            
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")
            
        # 获取图像尺寸
        h, w = img.shape[:2]
        image_size = (w, h)
        
        # 如果提供了相机矩阵和畸变系数，则使用它们；否则抛出错误
        if camera_matrix is None or dist_coeffs is None:
            raise ValueError("必须提供相机内参矩阵和畸变系数用于评估")
            
        # 将相机矩阵和畸变系数转换为正确形状
        camera_matrix = np.array(camera_matrix).reshape(3, 3)
        dist_coeffs = np.array(dist_coeffs)
        
        # 计算优化的相机内参矩阵和ROI区域
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, image_size, 1, image_size
        )
        
        # 应用去畸变
        undistorted_img = cv2.undistort(
            img, camera_matrix, dist_coeffs, None, new_camera_matrix
        )
        
        # 裁剪去畸变图像
        x, y, w, h = roi
        undistorted_img = undistorted_img[y:y+h, x:x+w]
        
        # 调整图像大小使其一致便于对比
        resized_original = cv2.resize(img, (w, h))
        
        print(f"原图尺寸: {img.shape[1]}x{img.shape[0]}")
        print(f"矫正后图像尺寸: {undistorted_img.shape[1]}x{undistorted_img.shape[0]}")
        print("图像矫正完成，可以通过比较原图和矫正图来评估标定效果")
        
        return resized_original, undistorted_img

    def save_evaluation_result(self, original_img, undistorted_img, output_dir="temp/eval_result"):
        """
        保存评估结果
        
        Args:
            original_img: 原始图像
            undistorted_img: 矫正后的图像
            output_dir: 输出目录
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存原始图像
        original_path = os.path.join(output_dir, "original.jpg")
        cv2.imwrite(original_path, original_img)
        print(f"原始图像已保存到: {original_path}")
        
        # 保存矫正后的图像
        undistorted_path = os.path.join(output_dir, "undistorted.jpg")
        cv2.imwrite(undistorted_path, undistorted_img)
        print(f"矫正图像已保存到: {undistorted_path}")
        
        # 拼接两张图像进行对比
        comparison = np.hstack((original_img, undistorted_img))
        comparison_path = os.path.join(output_dir, "comparison.jpg")
        cv2.imwrite(comparison_path, comparison)
        print(f"对比图像已保存到: {comparison_path}")


def main():
    """
    主函数 - 执行相机内参标定
    """
    print("开始相机内参标定...")
    
    # 初始化标定器
    calibrator = IntrinsicCalibrator()
    
    # 查找标定图像
    calibration_images_path = "data/calib/camera/image"
    
    # 查找目录中的图像文件
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob(os.path.join(calibration_images_path, ext)))
    

    print(f"找到 {len(image_files)} 个图像文件")
    
    # 添加图像到标定器
    valid_count = 0
    # 使用tqdm显示进度条
    for image_path in tqdm(image_files, desc="处理图像"):
        if calibrator.add_calibration_image(image_path):
            valid_count += 1
    
    if valid_count < 3:  # 至少需要3张图像
        print("有效图像数量不足3张，无法进行标定")
        return
    
    print(f"成功加载 {valid_count} 张图像用于标定")
    
    # 获取图像尺寸（使用第一张图像）
    sample_image = cv2.imread(image_files[0])
        
    image_size = (sample_image.shape[1], sample_image.shape[0])  # (width, height)
    print(f"图像尺寸: {image_size[0]}x{image_size[1]}")
    
    # 执行标定
    success, result = calibrator.calibrate(image_size, fix_principal_point=False, use_rational_model=False)
    if success:
        # 保存结果
        output_path = "config/calibrated_config.json"
        calibrator.save_calibration(result, output_path)
        
        # 评估标定效果
        print("\n开始评估内参标定效果...")
        camera_matrix = np.array(result['camera_matrix']).reshape(3, 3)
        dist_coeffs = np.array(result['distortion_coefficients'])
        
        try:
            original_img, undistorted_img = calibrator.evaluate_intrinsic_effect(
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs
            )
            
            # 保存评估结果
            calibrator.save_evaluation_result(original_img, undistorted_img)
            print("内参标定效果评估完成!")
        except Exception as e:
            print(f"评估过程中出现错误: {e}")
        
        print("相机内参标定完成!")
    else:
        print("相机内参标定失败!")


if __name__ == "__main__":
    main()