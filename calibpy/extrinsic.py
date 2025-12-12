import cv2
import numpy as np
import json
import os
import shutil
from datetime import datetime

try:
    import camera
    import lidar
except:
    import calibpy.camera as camera
    import calibpy.lidar as lidar

from rdsend.client import ClientSender

import utils.draw_box as draw

def backup_config(config_file_path="./config/config.json", backup_dir="./config/backups"):
    """
    备份配置文件
    
    Args:
        config_file_path: 配置文件路径
        backup_dir: 备份目录路径
        
    Returns:
        bool: 备份是否成功
    """
    # 检查配置文件是否存在
    if not os.path.exists(config_file_path):
        print(f"配置文件 {config_file_path} 不存在")
        return False
    
    # 创建备份目录
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
    
    # 生成备份文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_filename = f"config_backup_{timestamp}.json"
    backup_path = os.path.join(backup_dir, backup_filename)
    
    try:
        # 复制配置文件到备份目录
        shutil.copy2(config_file_path, backup_path)
        print(f"配置文件已备份到: {backup_path}")
        return True
    except Exception as e:
        print(f"备份配置文件时出错: {e}")
        return False


def save_camera_extrinsic(extrinsic_matrix, config_file_path="./config/config.json"):
    """
    将标定得到的相机外参矩阵保存到配置文件中
    
    Args:
        extrinsic_matrix: 4x4的相机外参矩阵
        config_file_path: 配置文件路径
    """
    # 检查配置文件是否存在
    if not os.path.exists(config_file_path):
        print(f"配置文件 {config_file_path} 不存在")
        return False
    
    try:
        # 读取现有配置
        with open(config_file_path, 'r') as f:
            config_data = json.load(f)
        
        # 更新相机外参
        # 将4x4矩阵转换为一维列表
        extrinsic_list = extrinsic_matrix.flatten().tolist()
        config_data["camera_extrinsic"] = extrinsic_list
        
        # 写回配置文件
        with open(config_file_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"相机外参已保存到 {config_file_path}")
        return True
    except Exception as e:
        print(f"保存相机外参时出错: {e}")
        return False


def main():
    client = ClientSender()
    client.connect()
    
    print("开始相机标定过程...")
    cam = camera.Camera()
    lid = lidar.Lidar()
    
    cloud = lid.cloud
    # cloud = [(lid.extrinsic @ np.append(point, 1))[:3] for point in cloud]
    for point in cloud:
        client.send_point(point[0], point[1], point[2])
    
    corners = draw.corners_of_psr(lid.bounding_box.position, lid.bounding_box.scale, lid.bounding_box.rotation)
    index = draw.edges()
    
    
    for idx in index:
        start = [corners[idx[0]][0], corners[idx[0]][1], corners[idx[0]][2]]
        end = [corners[idx[1]][0], corners[idx[1]][1], corners[idx[1]][2]]
        client.send_segment(start, end)
    
    print("获取LiDAR生成的棋盘格3D点...")
    caliboard3D = lid.target_corners()
    
    # caliboard3D = [(lid.extrinsic @ np.append(point, 1))[:3] for point in caliboard3D]
    # for point in caliboard3D:
    #     client.send_point(point[0], point[2], point[1])
    
    print("获取图像中的棋盘格2D点...")
    caliboard2D = cam.get_caliboard()
    
    print(f"3D点数量: {len(caliboard3D) if caliboard3D is not None else 'None'}")
    print(f"2D点数量: {len(caliboard2D) if caliboard2D is not None else 'None'}")
    
    # 确保点的数量相同
    if len(caliboard3D) == len(caliboard2D):

        # 验证坐标范围
        print("\n坐标范围验证:")
        print(f"3D点坐标范围 - X:{np.min(caliboard3D[:,0]):.3f}~{np.max(caliboard3D[:,0]):.3f}, "
                f"Y:{np.min(caliboard3D[:,1]):.3f}~{np.max(caliboard3D[:,1]):.3f}, "
                f"Z:{np.min(caliboard3D[:,2]):.3f}~{np.max(caliboard3D[:,2]):.3f}")
        print(f"2D点坐标范围 - U:{np.min(caliboard2D[:,0]):.3f}~{np.max(caliboard2D[:,0]):.3f}, "
                f"V:{np.min(caliboard2D[:,1]):.3f}~{np.max(caliboard2D[:,1]):.3f}")
        
            
        # 确保内参矩阵已定义
        camera_matrix = np.array(cam.intrinsic, dtype=np.float32)
        print(f"相机内参矩阵: \n{camera_matrix}")
        
        # 使用从配置文件加载的畸变系数，确保正确的形状
        dist_coeffs = np.array(cam.distortion_coefficients, dtype=np.float32).reshape(1, -1)
        
        success, rvec, tvec = cv2.solvePnP(
            caliboard3D, 
            caliboard2D, 
            camera_matrix, 
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if success:
            # 计算重投影误差
            projected_points, _ = cv2.projectPoints(
                caliboard3D, rvec, tvec, camera_matrix, dist_coeffs)
            
            projected_points = projected_points.reshape(-1, 2)
            reprojection_error = np.sqrt(np.mean((caliboard2D - projected_points) ** 2))
            
            print(f"  方法 pnp 重投影误差: {reprojection_error:.2f} 像素")
            
            # 检查旋转矩阵的正交性
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            
            # 构造4x4的变换矩阵（从LiDAR坐标系到相机坐标系）
            extrinsic_matrix = np.eye(4, dtype=np.float32)
            extrinsic_matrix[:3, :3] = rotation_matrix
            extrinsic_matrix[:3, 3] = tvec.reshape(-1)
            
            print("\n外参矩阵详细信息 (LiDAR到相机):")
            print("旋转矩阵 R:")
            print(rotation_matrix)
            print("平移向量 T:", tvec.flatten())
            print("\n完整4x4外参矩阵:")
            print(extrinsic_matrix)
            
            # 验证变换矩阵
            print(f"\n变换矩阵验证:")
            print(f"  行列式: {np.linalg.det(rotation_matrix):.6f} " 
                    f"(理想值接近1.0)")
            print(f"  平移距离: {np.linalg.norm(tvec):.4f} 米")
            
            # 计算重投影误差
            projected_points, _ = cv2.projectPoints(
                caliboard3D, rvec, tvec, camera_matrix, dist_coeffs)
            
            # 计算均方根误差
            projected_points = projected_points.reshape(-1, 2)
            reprojection_error = np.sqrt(np.mean((caliboard2D - projected_points) ** 2))
            
            print(f"最终重投影误差: {reprojection_error} 像素")
            
            # 先备份配置文件
            print("正在备份当前配置文件...")
            if backup_config():
                # 保存外参矩阵到配置文件
                save_camera_extrinsic(extrinsic_matrix)
            else:
                print("备份失败，取消保存操作")


if __name__ == "__main__":
    main()