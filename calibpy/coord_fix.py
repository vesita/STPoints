#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
坐标系修复脚本，用于解决LiDAR和相机坐标系不匹配问题
"""

import cv2
import numpy as np

try:
    import camera
    import lidar
except:
    import calibpy.camera as camera
    import calibpy.lidar as lidar

def fix_coordinate_system():
    """修复坐标系问题"""
    print("开始修复坐标系问题...")
    
    # 加载相机和LiDAR数据
    cam = camera.Camera()
    lid = lidar.Lidar()
    
    # 获取LiDAR点云
    lidar_points = lid.target_corners()
    print(f"LiDAR点云数量: {len(lidar_points)}")
    
    if len(lidar_points) > 0:
        points_array = np.array(lidar_points)
        print("LiDAR点云坐标范围:")
        print(f"  X: [{np.min(points_array[:, 0]):.3f}, {np.max(points_array[:, 0]):.3f}]")
        print(f"  Y: [{np.min(points_array[:, 1]):.3f}, {np.max(points_array[:, 1]):.3f}]")
        print(f"  Z: [{np.min(points_array[:, 2]):.3f}, {np.max(points_array[:, 2]):.3f}]")
        
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
        
        # 尝试不同的坐标变换
        transforms = [
            ("原始坐标", lambda p: p),
            ("LiDAR标准坐标变换", lambda p: np.column_stack([p[:, 1], -p[:, 0], p[:, 2]])),  # (y, -x, z)
            ("Open3D到OpenCV", lambda p: np.column_stack([p[:, 0], -p[:, 1], -p[:, 2]])),   # (x, -y, -z)
            ("ROS到OpenCV", lambda p: np.column_stack([p[:, 2], -p[:, 0], -p[:, 1]])),       # (z, -x, -y)
        ]
        
        image_corners = cam.get_caliboard()
        print(f"\n图像角点数量: {len(image_corners)}")
        if len(image_corners) > 0:
            print("前5个图像角点:")
            for i, point in enumerate(image_corners[:5]):
                print(f"  点{i}: ({point[0]:.2f}, {point[1]:.2f})")
        
        best_transform_name = ""
        best_error = float('inf')
        best_projected = None
        
        for transform_name, transform_func in transforms:
            print(f"\n测试变换: {transform_name}")
            
            # 应用坐标变换
            transformed_points = transform_func(points_array)
            
            # 投影到图像
            projected_points, _ = cv2.projectPoints(
                transformed_points.astype(np.float32), 
                rvec, tvec, 
                camera_matrix.astype(np.float32), 
                dist_coeffs.astype(np.float32)
            )
            projected_points = projected_points.reshape(-1, 2)
            
            print(f"  投影结果前5个点:")
            for i, point in enumerate(projected_points[:5]):
                print(f"    点{i}: ({point[0]:.2f}, {point[1]:.2f})")
            
            # 计算与图像角点的误差（如果有图像角点的话）
            if len(image_corners) > 0 and len(image_corners) == len(projected_points):
                error = np.sqrt(np.mean((image_corners - projected_points) ** 2))
                print(f"  重投影误差: {error:.2f} 像素")
                
                if error < best_error:
                    best_error = error
                    best_transform_name = transform_name
                    best_projected = projected_points
                    
        print(f"\n最佳变换: {best_transform_name} (误差: {best_error:.2f} 像素)")
        
        if best_projected is not None:
            print("最佳投影结果前5个点:")
            for i, point in enumerate(best_projected[:5]):
                print(f"  点{i}: ({point[0]:.2f}, {point[1]:.2f})")

if __name__ == "__main__":
    fix_coordinate_system()