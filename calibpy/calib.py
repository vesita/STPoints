import cv2
import numpy as np
import json
import os
import shutil
from datetime import datetime

try:
    import fast_config
    import camera
    import lidar
except:
    import calibpy.fast_config as fast_config
    import calibpy.camera as camera
    import calibpy.lidar as lidar


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


def save_js_compatible_camera_extrinsic(extrinsic_matrix, config_file_path="./config/config.json"):
    """
    将标定得到的相机外参矩阵转换为JS兼容的坐标系统并保存到配置文件中
    
    在JavaScript中使用的坐标变换矩阵:
    [0, -1,  0,  0]
    [0,  0, -1,  0]
    [1,  0,  0,  0]
    [0,  0,  0,  1]
    
    Args:
        extrinsic_matrix: 4x4的相机外参矩阵
        config_file_path: 配置文件路径
    """
    # 检查配置文件是否存在
    if not os.path.exists(config_file_path):
        print(f"配置文件 {config_file_path} 不存在")
        return False
    
    try:
        # 定义从Python到JavaScript坐标系统的变换矩阵
        view_matrix = np.array([
            [0,  0,  1,  0],
            [-1,  0, 0,  0],
            [0,  -1,  0,  0],
            [0,  0,  0,  1]
        ], dtype=np.float32)
        
        # 转换外参矩阵以适应JavaScript坐标系统
        # JS使用: viewMatrix * extrinsic
        js_compatible_extrinsic = view_matrix @ extrinsic_matrix
        
        # 读取现有配置
        with open(config_file_path, 'r') as f:
            config_data = json.load(f)
        
        # 添加JS兼容的相机外参到配置中
        js_extrinsic_list = js_compatible_extrinsic.flatten().tolist()
        config_data["camera_extrinsic_js"] = js_extrinsic_list
        
        # 写回配置文件
        with open(config_file_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"JS兼容的相机外参已保存到 {config_file_path} 的 camera_extrinsic_js 字段")
        return True
    except Exception as e:
        print(f"保存JS兼容的相机外参时出错: {e}")
        return False


def main():
    print("开始相机标定过程...")
    cam = camera.Camera()
    lid = lidar.Lidar()
    
    print("获取LiDAR生成的棋盘格3D点...")
    caliboard3D = lid.target_corners()
    print("获取图像中的棋盘格2D点...")
    caliboard2D = cam.get_caliboard()
    
    print(f"3D点数量: {len(caliboard3D) if caliboard3D is not None else 'None'}")
    print(f"2D点数量: {len(caliboard2D) if caliboard2D is not None else 'None'}")
    
    # 使用PnP计算相机外参
    # 增加对 None 和空数组的检查
    if caliboard3D is not None and len(caliboard3D) > 0 and caliboard2D is not None and len(caliboard2D) > 0:
        # 确保点的数量相同
        if len(caliboard3D) == len(caliboard2D):
            # 转换为numpy数组
            object_points = np.array(caliboard3D, dtype=np.float32)
            image_points = np.array(caliboard2D, dtype=np.float32)
            
            print(f"object_points shape: {object_points.shape}")
            print(f"image_points shape: {image_points.shape}")
            
            # 输出一些样本点用于调试
            print("前5个3D点:")
            print(object_points[:5])
            print("前5个2D点:")
            print(image_points[:5])
            
            # 检查是否有无效值
            if np.isnan(object_points).any() or np.isinf(object_points).any():
                print("错误：3D点中包含NaN或Inf值")
                return
                
            if np.isnan(image_points).any() or np.isinf(image_points).any():
                print("错误：2D点中包含NaN或Inf值")
                return
            
            # 确保3D点是N×3的形状，2D点是N×2的形状
            if len(object_points.shape) != 2 or object_points.shape[1] != 3:
                print(f"错误：3D点形状不正确，期望(N,3)，实际{object_points.shape}")
                return
                
            if len(image_points.shape) != 2 or image_points.shape[1] != 2:
                print(f"错误：2D点形状不正确，期望(N,2)，实际{image_points.shape}")
                return
            
            print(f"处理后 object_points shape: {object_points.shape}")
            print(f"处理后 image_points shape: {image_points.shape}")
            
            # 确保有足够的点进行solvePnP (至少4个点)
            if len(object_points) < 4 or len(image_points) < 4:
                print(f"点数不足，需要至少4个点，当前3D点数: {len(object_points)}, 2D点数: {len(image_points)}")
                return
            # object_points = lid.to_world(object_points)
                
            # 确保内参矩阵已定义
            camera_matrix = np.array(cam.intrinsic, dtype=np.float32)
            print(f"相机内参矩阵: \n{camera_matrix}")
            
            # 默认的畸变系数（如果没有提供的话）
            dist_coeffs = np.zeros((4, 1), dtype=np.float32)
            
            print("开始调用solvePnP...")
            print(f"solvePnP参数类型检查:")
            print(f"  object_points类型: {type(object_points)}, 形状: {object_points.shape if hasattr(object_points, 'shape') else '无形状属性'}")
            print(f"  image_points类型: {type(image_points)}, 形状: {image_points.shape if hasattr(image_points, 'shape') else '无形状属性'}")
            print(f"  camera_matrix类型: {type(camera_matrix)}, 形状: {camera_matrix.shape if hasattr(camera_matrix, 'shape') else '无形状属性'}")
            print(f"  dist_coeffs类型: {type(dist_coeffs)}, 形状: {dist_coeffs.shape if hasattr(dist_coeffs, 'shape') else '无形状属性'}")
            
            # 使用solvePnP计算外参
            success, rvec, tvec = cv2.solvePnP(
                object_points, 
                image_points, 
                camera_matrix, 
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if success:
                # 将旋转向量转换为旋转矩阵
                rotation_matrix, _ = cv2.Rodrigues(rvec)
                
                # 构造4x4的变换矩阵
                extrinsic_matrix = np.eye(4, dtype=np.float32)
                extrinsic_matrix[:3, :3] = rotation_matrix
                extrinsic_matrix[:3, 3] = tvec.flatten()
                
                print("外参矩阵 (世界到相机):")
                print(extrinsic_matrix)
                
                # 计算重投影误差
                projected_points, _ = cv2.projectPoints(
                    object_points, rvec, tvec, camera_matrix, dist_coeffs)
                
                # 计算均方根误差
                projected_points = projected_points.reshape(-1, 2)
                reprojection_error = np.sqrt(np.mean((image_points - projected_points) ** 2))
                
                print(f"重投影误差: {reprojection_error} 像素")
                
                # 询问用户是否保存标定结果
                save_choice = input("是否将标定结果保存并覆盖相机外参？(y/N): ")
                if save_choice.lower() in ['y', 'yes']:
                    # 先备份配置文件
                    print("正在备份当前配置文件...")
                    if backup_config():
                        # 保存外参矩阵到配置文件
                        if save_camera_extrinsic(extrinsic_matrix):
                            '''# 同时保存JS兼容的外参矩阵
                            if save_js_compatible_camera_extrinsic(extrinsic_matrix):
                                print("相机外参及JS兼容外参已成功更新")
                            else:
                                print("保存JS兼容外参失败")'''
                        else:
                            print("保存相机外参失败")
                    else:
                        print("备份失败，取消保存操作")
                else:
                    print("未保存相机外参")
            else:
                print("PnP问题求解失败")
        else:
            print(f"点数量不匹配: 3D点 ({len(caliboard3D)}) vs 2D点 ({len(caliboard2D)})")
    else:
        print("未找到标定点")
        if caliboard3D is None:
            print("3D点为 None")
        elif len(caliboard3D) == 0:
            print("3D点为空")
            
        if caliboard2D is None:
            print("2D点为 None")
        elif len(caliboard2D) == 0:
            print("2D点为空")


if __name__ == "__main__":
    main()