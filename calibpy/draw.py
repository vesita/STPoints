#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
绘制脚本，用于验证相机-LiDAR外参计算的正确性
"""

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
        将3D点云投影到图像上
        
        Args:
            points: 3D点云数据 (Nx3)
        
        Returns:
            projected_points: 投影到图像上的点 (Nx2)
        """
        if len(points) == 0:
            return np.array([])
        
        # 构造变换矩阵：世界坐标系 -> 相机坐标系
        world_to_camera = self.cam.extrinsic
        
        # 构造齐次坐标
        points_world_homo = np.hstack([points, np.ones((points.shape[0], 1))])
        
        # 变换到相机坐标系
        points_camera_homo = (world_to_camera @ points_world_homo.T).T
        
        # 检查是否在相机前方（Z坐标为正）
        valid_indices = points_camera_homo[:, 2] > 0
        points_camera = points_camera_homo[valid_indices, :3]
        
        print(f"在相机前方的点数量: {np.sum(valid_indices)} / {len(points)}")
        
        if len(points_camera) == 0:
            return np.array([])
        
        # 应用相机内参进行投影
        camera_points_homo = (self.cam.intrinsic @ points_camera.T).T
        
        # 透视除法得到图像坐标
        projected_points = camera_points_homo[:, :2] / camera_points_homo[:, 2:3]
        
        # 检查投影点是否在图像范围内 (从图像获取实际尺寸)
        img_width, img_height = self.cam.image.size
        valid_x = (projected_points[:, 0] >= 0) & (projected_points[:, 0] < img_width)
        valid_y = (projected_points[:, 1] >= 0) & (projected_points[:, 1] < img_height)
        valid_indices_img = valid_x & valid_y
        
        print(f"在图像范围内的点数量: {np.sum(valid_indices_img)} / {len(projected_points)}")
        if np.sum(valid_indices_img) > 0:
            print("前5个在图像范围内的点坐标:")
            print(projected_points[valid_indices_img][:5])
        
        return projected_points
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
        
        # 获取LiDAR投影的标定板角点
        lidar_board_corners = self.lid.target_corners()
        lidar_projected_corners = self.project_points_to_image(lidar_board_corners)
        
        # 获取图像中检测到的棋盘格角点
        image_corners = self.cam.get_caliboard()
        
        print(f"LiDAR投影角点数量: {len(lidar_projected_corners)}")
        print(f"图像检测角点数量: {len(image_corners)}")
        
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
        
        camera_matrix = self.cam.intrinsic
        dist_coeffs = np.zeros((4, 1), dtype=np.float32)
        
        rvec, _ = cv2.Rodrigues(self.cam.extrinsic[:3, :3])
        tvec = self.cam.extrinsic[:3, 3].reshape(-1, 1)
        print("从外参矩阵提取的rvec:")
        print(rvec)
        print("从外参矩阵提取的tvec:")
        print(tvec)
        
        # 直接使用LiDAR坐标进行重投影
        projected_points, _ = cv2.projectPoints(
            lidar_board_corners, rvec, tvec, camera_matrix, dist_coeffs)
        
        # 计算重投影误差 (均方根误差)，此计算方式必须与calib.py中的评估标准一致
        projected_points = projected_points.reshape(-1, 2)
        print("前10个重投影点:")
        print(projected_points[:10])
        print("前10个图像检测点:")
        print(image_corners[:10])
        # 使用欧氏距离的均方根作为重投影误差
        reprojection_error = np.sqrt(np.mean((image_corners - projected_points) ** 2))
        
        print(f"重投影误差统计:")
        print(f"  平均误差: {reprojection_error:.2f} 像素")
        
        # 在图像上显示误差信息
        cv2.putText(image_cv, f'重投影误差: {reprojection_error:.2f} 像素', 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # 保存带标注的图像
        if save_image:
            cv2.imwrite('calibration_result.png', image_cv)
            print("已保存标注图像到 calibration_result.png")
        
        return {
            'image': image_cv,
            'reprojection_error': reprojection_error,
            'lidar_projected_corners': lidar_projected_corners,
            'image_corners': image_corners
        }

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
    quick_visualize_calibration()


if __name__ == "__main__":
    main()