#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
绘制脚本，用于验证相机-LiDAR外参计算的正确性
"""

import os
import time
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

try:
    import fast_config
    import camera
    import lidar
except:
    import calibpy.camera as camera
    import calibpy.lidar as lidar
    import calibpy.fast_config as fast_config


class CalibrationVisualizer:
    """
    相机-LiDAR标定结果可视化工具类
    用于验证外参计算的正确性
    """
    
    def __init__(self, cam=None, lid=None):
        """
        初始化可视化工具
        
        Args:
            cam: 相机对象，如果为None则自动加载
            lid: LiDAR对象，如果为None则自动加载
        """
        self.cam = cam if cam else camera.Camera()
        self.lid = lid if lid else lidar.Lidar()
    
    def project_points_to_image(self, points):
        """
        使用OpenCV的projectPoints函数将3D点云投影到图像上
        
        Args:
            points: 3D点云数据 (Nx3)，在LiDAR坐标系中
        
        Returns:
            projected_points: 投影到图像上的点 (Nx2)
        """
        if len(points) == 0:
            return np.array([])
        
        # 使用与calib.py中相同的投影方法
        camera_matrix = self.cam.intrinsic
        dist_coeffs = self.cam.distortion_coefficients.reshape(1, -1)
        
        # 从外参矩阵提取旋转向量和平移向量（与calib.py保持一致）
        rvec, _ = cv2.Rodrigues(self.cam.extrinsic[:3, :3])
        tvec = self.cam.extrinsic[:3, 3].reshape(-1, 1)
        
        # 使用cv2.projectPoints进行投影
        projected_points, _ = cv2.projectPoints(
            np.array(points, dtype=np.float32), 
            rvec, tvec, 
            camera_matrix.astype(np.float32), 
            dist_coeffs.astype(np.float32)
        )
        
        return projected_points.reshape(-1, 2)
    
    def img_cloud_mapping(self, save_image=True):
        """
        将LiDAR投影的标定板角点与图像中检测到的角点进行对比
        
        Args:
            save_image: 是否保存带标注的图像
            
        Returns:
            dict: 包含对比结果的字典
                - image: 带标注的图像 (OpenCV格式)
                - reprojection_error: 重投影误差 (像素)
                - lidar_projected_corners: LiDAR投影角点
                - image_corners: 图像检测角点
        """
        # 将PIL图像转换为OpenCV格式
        image_cv = cv2.cvtColor(np.array(self.cam.image), cv2.COLOR_RGB2BGR)
        
        # 获取LiDAR坐标系中的标定板角点
        lidar_board_corners = self.lid.target_corners()
        lidar_board_corners = [(self.lid.extrinsic @ np.append(point, 1))[:3] for point in lidar_board_corners]
        print(f"LiDAR棋盘格角点数量: {len(lidar_board_corners)}")
        
        # 获取相机内外参
        camera_matrix = self.cam.intrinsic
        dist_coeffs = self.cam.distortion_coefficients.reshape(1, -1)  # 确保形状正确
        cam_extrinsic = self.cam.extrinsic
        
        # 使用与calib.py完全一致的方式进行投影
        rvec, _ = cv2.Rodrigues(cam_extrinsic[:3, :3])
        tvec = cam_extrinsic[:3, 3].reshape(-1, 1)
        print(f"旋转矩阵 (rvec):\n{rvec}")
        print(f"平移向量 (tvec):\n{tvec}")
        
        lidar_projected_corners, _ = cv2.projectPoints(
            np.array(lidar_board_corners, dtype=np.float32), 
            rvec, tvec, 
            camera_matrix.astype(np.float32), 
            dist_coeffs.astype(np.float32)
        )
        lidar_projected_corners = lidar_projected_corners.reshape(-1, 2)
        
        # 获取图像中检测到的棋盘格角点
        image_corners = self.cam.get_caliboard()
        print(f"图像检测角点数量: {len(image_corners)}")
        
        print(f"LiDAR投影角点数量: {len(lidar_projected_corners)}")
        print(f"图像检测角点数量: {len(image_corners)}")
        
        # 确保点的数量相同
        if len(lidar_projected_corners) != len(image_corners):
            print(f"警告: LiDAR投影角点数量({len(lidar_projected_corners)})与图像检测角点数量({len(image_corners)})不匹配")
            # 选择较小的数量进行比较
            min_len = min(len(lidar_projected_corners), len(image_corners))
            lidar_projected_corners = lidar_projected_corners[:min_len]
            image_corners = image_corners[:min_len]
        
        # 绘制LiDAR投影的角点（红色）
        for i, point in enumerate(lidar_projected_corners):
            if not np.isnan(point[0]) and not np.isnan(point[1]):
                x, y = point
                if 0 <= x < image_cv.shape[1] and 0 <= y < image_cv.shape[0]:
                    cv2.circle(image_cv, (int(x), int(y)), 4, (0, 0, 255), -1)  # 红色圆点
                    cv2.putText(image_cv, f'L{i}', (int(x)+5, int(y)-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # 绘制图像检测的角点（蓝色）
        for i, point in enumerate(image_corners):
            x, y = point
            if 0 <= x < image_cv.shape[1] and 0 <= y < image_cv.shape[0]:
                cv2.circle(image_cv, (int(x), int(y)), 4, (255, 0, 0), -1)  # 蓝色圆点
                cv2.putText(image_cv, f'I{i}', (int(x)+5, int(y)+15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        
        # 计算重投影误差，确保只有在有数据时才计算
        reprojection_error = float('inf')
        if len(image_corners) > 0 and len(lidar_projected_corners) > 0:
            # 使用欧氏距离的均方根作为重投影误差
            reprojection_error = np.sqrt(np.mean((image_corners - lidar_projected_corners) ** 2))
        
        print(f"重投影误差统计:")
        print(f"  平均误差: {reprojection_error:.2f} 像素")
        print(f"  使用点数: {len(image_corners)}")
        
        # 在图像上显示误差信息
        cv2.putText(image_cv, f'重投影误差: {reprojection_error:.2f} 像素', 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        if save_image:
            # 确保目录存在
            os.makedirs('temp/image', exist_ok=True)
            
            # 生成时间戳文件名
            timestamp = int(time.time())
            save_path = f'temp/image/calibration_result_{timestamp}.png'
            
            cv2.imwrite(save_path, image_cv)
            print(f"已保存标注图像到 {save_path}")

        return {
            'image': image_cv,
            'reprojection_error': reprojection_error,
            'lidar_projected_corners': lidar_projected_corners,
            'image_corners': image_corners
        }

    def opencv_project_principle(self, points_3d):
        """
        演示OpenCV projectPoints的正向投影原理
        手动实现3D点到2D图像坐标的完整投影过程，包括：
        1. 外参变换（世界坐标系 -> 相机坐标系）
        2. 透视投影（相机坐标系 -> 归一化图像坐标系）
        3. 应用畸变模型
        4. 内参变换（归一化坐标系 -> 像素坐标系）
        
        Args:
            points_3d: 3D点云数据 (Nx3)，在世界坐标系中
            
        Returns:
            tuple: (projected_cv, projected_manual)
                - projected_cv: 使用cv2.projectPoints得到的结果
                - projected_manual: 手动计算得到的结果
        """
        # 获取相机参数
        camera_matrix = self.cam.intrinsic
        dist_coeffs = self.cam.distortion_coefficients
        rvec, _ = cv2.Rodrigues(self.cam.extrinsic[:3, :3])  # 旋转向量
        tvec = self.cam.extrinsic[:3, 3].reshape(-1, 1)     # 平移向量
        
        points_3d = np.array(points_3d, dtype=np.float32)
        
        # 步骤1: 从世界坐标系变换到相机坐标系
        # 将3D点转换为齐次坐标并应用变换
        ones = np.ones((points_3d.shape[0], 1))
        points_3d_homo = np.hstack([points_3d, ones])  # 齐次坐标
        
        # 应用外参变换 (旋转和平移)
        # R_world_to_cam * P_world + t
        rmat, _ = cv2.Rodrigues(rvec)
        points_cam = (rmat @ points_3d.T).T + tvec.flatten()  # 应用旋转和平移
        
        # 步骤2: 相机坐标系到归一化图像坐标系 (透视投影)
        # 转换到归一化平面 (除以Z得到归一化坐标)
        valid_z = points_cam[:, 2] > 1e-6  # 避免除以接近0的值
        points_normalized = np.zeros_like(points_cam)
        points_normalized[valid_z, :2] = points_cam[valid_z, :2] / points_cam[valid_z, 2:3]
        
        # 步骤3: 应用畸变模型
        x, y = points_normalized[:, 0], points_normalized[:, 1]
        
        # 使用径向和切向畸变系数
        k1, k2, p1, p2, k3 = dist_coeffs[:5]  # 取前5个畸变系数
        
        # 计算径向畸变
        r2 = x*x + y*y
        r4 = r2*r2
        r6 = r4*r2
        radial_distortion = 1 + k1*r2 + k2*r4 + k3*r6
        
        # 计算切向畸变
        x_distorted = x * radial_distortion + (2*p1*x*y + p2*(r2 + 2*x*x))
        y_distorted = y * radial_distortion + (p1*(r2 + 2*y*y) + 2*p2*x*y)
        
        # 步骤4: 应用内参矩阵转换到像素坐标系
        u = camera_matrix[0, 0] * x_distorted + camera_matrix[0, 2]  # fx*x + cx
        v = camera_matrix[1, 1] * y_distorted + camera_matrix[1, 2]  # fy*y + cy
        
        projected_manual = np.column_stack([u, v])
        
        return projected_manual
    
    def manually_eval(self, save_image=True):
        # 将PIL图像转换为OpenCV格式
        image_cv = cv2.cvtColor(np.array(self.cam.image), cv2.COLOR_RGB2BGR)
        
        # 获取LiDAR坐标系中的标定板角点
        lidar_board_corners = self.lid.target_corners()
        # lidar_board_corners = [(self.lid.extrinsic @ np.append(point, 1))[:3] for point in lidar_board_corners]
        
        # targets = [
        #     (self.cam.intrinsic @ (self.cam.extrinsic @ np.append(point, 1))[:3])[:2] for point in lidar_board_corners
        # ]
        
        targets = self.opencv_project_principle(lidar_board_corners)
        
        '''# 获取相机内外参
        camera_matrix = self.cam.intrinsic
        dist_coeffs = self.cam.distortion_coefficients.reshape(1, -1)  # 确保形状正确
        cam_extrinsic = self.cam.extrinsic
        
        # 使用与calib.py完全一致的方式进行投影（使用OpenCV的projectPoints）
        rvec, _ = cv2.Rodrigues(cam_extrinsic[:3, :3])
        tvec = cam_extrinsic[:3, 3].reshape(-1, 1)
        
        targets, _ = cv2.projectPoints(
            np.array(lidar_board_corners, dtype=np.float32), 
            rvec, tvec, 
            camera_matrix.astype(np.float32), 
            dist_coeffs.astype(np.float32)
        )
        targets = targets.reshape(-1, 2)'''

        # 绘制LiDAR投影的角点（红色）
        for i, point in enumerate(targets):
            if not np.isnan(point[0]) and not np.isnan(point[1]):
                x, y = point
                if 0 <= x < image_cv.shape[1] and 0 <= y < image_cv.shape[0]:
                    cv2.circle(image_cv, (int(x), int(y)), 4, (0, 0, 255), -1)  # 红色圆点
                    cv2.putText(image_cv, f'L{i}', (int(x)+5, int(y)-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        if save_image:
            # 确保目录存在
            os.makedirs('temp/image', exist_ok=True)
            
            # 生成时间戳文件名
            timestamp = int(time.time())
            save_path = f'temp/image/calibration_result_{timestamp}.png'
            
            cv2.imwrite(save_path, image_cv)
            print(f"已保存标注图像到 {save_path}")
            
        return image_cv

    def visualize_calibration(self, show_plot=True, save_image=True):
        """
        可视化标定结果
        
        Args:
            show_plot: 是否显示matplotlib图表
            save_image: 是否保存标注图像
            
        Returns:
            dict: 包含所有结果的字典
        """
        print("开始验证相机-LiDAR外参计算...")
        
        # 对比LiDAR投影角点和图像检测角点
        print("对比LiDAR投影角点和图像检测角点...")
        comparison_result = self.img_cloud_mapping(save_image=save_image)
        return comparison_result


def quick_visualize_calibration(show_plot=True, save_image=True):
    """
    快速可视化标定结果的便捷函数
    
    Args:
        show_plot: 是否显示matplotlib图表
        save_image: 是否保存标注图像
        
    Returns:
        dict: 包含所有结果的字典
    """
    visualizer = CalibrationVisualizer()
    return visualizer.visualize_calibration(show_plot=show_plot, save_image=save_image)


def main():
    """主函数"""
    # quick_visualize_calibration()
    visualizer = CalibrationVisualizer()
    visualizer.manually_eval(save_image=True)


if __name__ == "__main__":
    main()