#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试脚本，用于验证坐标系和投影问题
"""

import cv2
import numpy as np

try:
    import camera
    import lidar
except:
    import calibpy.camera as camera
    import calibpy.lidar as lidar

def test_coordinate_system():
    """测试坐标系转换"""
    print("开始测试坐标系...")
    
    # 加载相机和LiDAR数据
    cam = camera.Camera()
    lid = lidar.Lidar()
    
    # 获取LiDAR点云
    lidar_points = lid.target_corners()
    print(f"LiDAR点云数量: {len(lidar_points)}")
    
    if len(lidar_points) > 0:
        print("LiDAR点云坐标范围:")
        points_array = np.array(lidar_points)
        print(f"  X: [{np.min(points_array[:, 0]):.3f}, {np.max(points_array[:, 0]):.3f}]")
        print(f"  Y: [{np.min(points_array[:, 1]):.3f}, {np.max(points_array[:, 1]):.3f}]")
        print(f"  Z: [{np.min(points_array[:, 2]):.3f}, {np.max(points_array[:, 2]):.3f}]")
        
        # 显示前几个点
        print("\n前5个LiDAR点:")
        for i, point in enumerate(lidar_points[:5]):
            print(f"  点{i}: ({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f})")
        
        # 获取相机参数
        camera_matrix = cam.intrinsic
        dist_coeffs = cam.distortion_coefficients.reshape(1, -1)
        extrinsic = cam.extrinsic
        
        print(f"\n相机内参矩阵:")
        print(camera_matrix)
        print(f"\n相机外参矩阵:")
        print(extrinsic)
        
        # 提取旋转和平移
        rvec, _ = cv2.Rodrigues(extrinsic[:3, :3])
        tvec = extrinsic[:3, 3].reshape(-1, 1)
        
        print(f"\n旋转向量:")
        print(rvec)
        print(f"\n平移向量:")
        print(tvec)
        
        # 尝试不同的坐标系转换
        print("\n尝试不同的坐标系转换...")
        
        # 原始投影
        projected_points, _ = cv2.projectPoints(
            points_array.astype(np.float32), 
            rvec, tvec, 
            camera_matrix.astype(np.float32), 
            dist_coeffs.astype(np.float32)
        )
        projected_points = projected_points.reshape(-1, 2)
        
        print("原始投影结果 (前5个点):")
        for i, point in enumerate(projected_points[:5]):
            print(f"  点{i}: ({point[0]:.2f}, {point[1]:.2f})")
        
        # 尝试坐标轴翻转
        # 尝试翻转Y轴
        flipped_points = points_array.copy()
        flipped_points[:, 1] = -flipped_points[:, 1]  # 翻转Y轴
        
        projected_flipped_y, _ = cv2.projectPoints(
            flipped_points.astype(np.float32), 
            rvec, tvec, 
            camera_matrix.astype(np.float32), 
            dist_coeffs.astype(np.float32)
        )
        projected_flipped_y = projected_flipped_y.reshape(-1, 2)
        
        print("\n翻转Y轴后的投影结果 (前5个点):")
        for i, point in enumerate(projected_flipped_y[:5]):
            print(f"  点{i}: ({point[0]:.2f}, {point[1]:.2f})")
        
        # 尝试翻转Z轴
        flipped_points_z = points_array.copy()
        flipped_points_z[:, 2] = -flipped_points_z[:, 2]  # 翻转Z轴
        
        projected_flipped_z, _ = cv2.projectPoints(
            flipped_points_z.astype(np.float32), 
            rvec, tvec, 
            camera_matrix.astype(np.float32), 
            dist_coeffs.astype(np.float32)
        )
        projected_flipped_z = projected_flipped_z.reshape(-1, 2)
        
        print("\n翻转Z轴后的投影结果 (前5个点):")
        for i, point in enumerate(projected_flipped_z[:5]):
            print(f"  点{i}: ({point[0]:.2f}, {point[1]:.2f})")
        
        # 尝试同时翻转Y和Z轴
        flipped_points_yz = points_array.copy()
        flipped_points_yz[:, 1] = -flipped_points_yz[:, 1]  # 翻转Y轴
        flipped_points_yz[:, 2] = -flipped_points_yz[:, 2]  # 翻转Z轴
        
        projected_flipped_yz, _ = cv2.projectPoints(
            flipped_points_yz.astype(np.float32), 
            rvec, tvec, 
            camera_matrix.astype(np.float32), 
            dist_coeffs.astype(np.float32)
        )
        projected_flipped_yz = projected_flipped_yz.reshape(-1, 2)
        
        print("\n同时翻转Y和Z轴后的投影结果 (前5个点):")
        for i, point in enumerate(projected_flipped_yz[:5]):
            print(f"  点{i}: ({point[0]:.2f}, {point[1]:.2f})")

if __name__ == "__main__":
    test_coordinate_system()