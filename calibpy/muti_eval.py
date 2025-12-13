import numpy as np
import cv2
import os
import sys
import copy



# goal:
'''
    需要从config.py中读取相机内参、畸变矩阵还有雷达外参矩阵，
    然后需要通过标定，分别计算data/extr和data/ann对应的外参矩阵
    （这里的外参矩阵不需要保存）
    然后分别交叉验证(使用从一帧计算出的外参矩阵验证到另一帧上)
    data/d1 -> data/d2 && data/d3
    data/d2 -> data/d1 && data/d3
    验证误差应该是小于10才是对的，哪怕误差突然下降，也不能说明最终结果是正确的，
    误差数值小于10是硬标准。

    如果误差过大，可以尝试修改矩阵求逆尝试以及坐标映射切换尝试
    （这里有两个方向，一个是对于矩阵的使用方法可能理解有误，
    另一个是可能我们对于坐标系的定义与opencv的定义不同）

    一切路径可以尝试硬编码，毕竟目前这个脚本是用来试错的
'''


# 添加上级目录到sys.path以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from calibpy.camera import Camera
    from calibpy.lidar import Lidar
    from calibpy.fast_config import Config, get_config, Target
except ImportError:
    from camera import Camera
    from lidar import Lidar
    from fast_config import Config, get_config, Target


def compute_extrinsic_matrix(dataset_path):
    """
    计算指定数据集的外参矩阵
    """
    print(f"计算数据集 {dataset_path} 的外参矩阵...")
    
    # 保存原始配置
    original_config = None
    try:
        config = get_config()
        original_config = {
            'target_path': config.target_path,
            'cloud': config.cloud,
            'image': config.image,
            'label': config.label
        }
        # 更新配置以使用指定数据集
        config.target_path = dataset_path
        target = Target(dataset_path)
        config.cloud = target.cloud
        config.image = target.image
        config.label = target.label
        
        print(f"更新配置: cloud={config.cloud}, image={config.image}, label={config.label}")
    except Exception as e:
        print(f"更新配置时出错: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    try:
        # 创建相机和激光雷达对象
        cam = Camera()
        lid = Lidar()
        
        # 获取棋盘格3D点和2D点
        caliboard3D = lid.target_corners()
        caliboard2D = cam.get_caliboard()
        
        print(f"3D点数量: {len(caliboard3D) if caliboard3D is not None else 0}")
        print(f"2D点数量: {len(caliboard2D) if caliboard2D is not None else 0}")
        
        if caliboard3D is None or len(caliboard3D) == 0:
            print(f"数据集 {dataset_path} 中未找到有效的3D点")
            return None
            
        if caliboard2D is None or len(caliboard2D) == 0:
            print(f"数据集 {dataset_path} 中未找到有效的2D点")
            return None
            
        if len(caliboard3D) != len(caliboard2D):
            print(f"数据集 {dataset_path} 中3D点和2D点数量不匹配: {len(caliboard3D)} vs {len(caliboard2D)}")
            return None
            
        # 转换为numpy数组
        object_points = np.array(caliboard3D, dtype=np.float32)
        image_points = np.array(caliboard2D, dtype=np.float32)
        
        print(f"对象点坐标范围: X[{np.min(object_points[:, 0]):.3f}, {np.max(object_points[:, 0]):.3f}], "
              f"Y[{np.min(object_points[:, 1]):.3f}, {np.max(object_points[:, 1]):.3f}], "
              f"Z[{np.min(object_points[:, 2]):.3f}, {np.max(object_points[:, 2]):.3f}]")
        print(f"图像点坐标范围: U[{np.min(image_points[:, 0]):.3f}, {np.max(image_points[:, 0]):.3f}], "
              f"V[{np.min(image_points[:, 1]):.3f}, {np.max(image_points[:, 1]):.3f}]")
        
        # 应用坐标系变换 (ROS到OpenCV): (x, y, z) -> (z, -x, -y)
        # 这里我们手动应用坐标变换以确保一致性
        transformed_object_points = np.column_stack([
            object_points[:, 2],    # z -> x
            -object_points[:, 0],   # -x -> y
            -object_points[:, 1]    # -y -> z
        ])
        
        # 相机内参和畸变系数
        camera_matrix = cam.intrinsic
        dist_coeffs = cam.distortion_coefficients.reshape(1, -1)  # 确保形状正确
        
        print(f"相机内参矩阵:\n{camera_matrix}")
        print(f"畸变系数: {dist_coeffs}")
        
        # 使用多种方法尝试solvePnP
        methods = [
            (cv2.SOLVEPNP_ITERATIVE, "ITERATIVE"),
            (cv2.SOLVEPNP_P3P, "P3P"),
            (cv2.SOLVEPNP_AP3P, "AP3P"),
            (cv2.SOLVEPNP_EPNP, "EPNP"),
            (cv2.SOLVEPNP_IPPE, "IPPE"),
            (cv2.SOLVEPNP_SQPNP, "SQPNP")
        ]
        
        best_result = None
        best_error = float('inf')
        
        for method, method_name in methods:
            try:
                if method not in [cv2.SOLVEPNP_ITERATIVE, cv2.SOLVEPNP_EPNP] and len(object_points) != 4:
                    continue
                    
                print(f"尝试方法 {method_name}")
                success, rvec, tvec = cv2.solvePnP(
                    transformed_object_points, 
                    image_points, 
                    camera_matrix, 
                    dist_coeffs,
                    flags=method
                )
                
                if success:
                    # 计算重投影误差
                    projected_points, _ = cv2.projectPoints(
                        transformed_object_points, rvec, tvec, camera_matrix, dist_coeffs)
                    
                    projected_points = projected_points.reshape(-1, 2)
                    reprojection_error = np.sqrt(np.mean((image_points - projected_points) ** 2))
                    
                    print(f"  方法 {method_name}: 误差 {reprojection_error:.2f} 像素")
                    
                    if reprojection_error < best_error:
                        best_error = reprojection_error
                        best_result = (success, rvec, tvec, method_name)
                else:
                    print(f"  方法 {method_name} 求解失败")
            except Exception as e:
                print(f"  方法 {method_name} 出错: {e}")
        
        if best_result:
            success, rvec, tvec, method_name = best_result
            print(f"选择最佳方法: {method_name}，误差: {best_error:.2f} 像素")
            
            # 构造4x4的变换矩阵
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            extrinsic_matrix = np.eye(4, dtype=np.float32)
            extrinsic_matrix[:3, :3] = rotation_matrix
            extrinsic_matrix[:3, 3] = tvec.flatten()
            
            print(f"外参矩阵:")
            print(extrinsic_matrix)
            
            return extrinsic_matrix
        else:
            print(f"所有PnP方法在数据集 {dataset_path} 上都失败了")
            return None
            
    except Exception as e:
        print(f"计算数据集 {dataset_path} 的外参矩阵时出错: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # 恢复原始配置
        if original_config:
            try:
                config = get_config()
                config.target_path = original_config['target_path']
                config.cloud = original_config['cloud']
                config.image = original_config['image']
                config.label = original_config['label']
                print("已恢复原始配置")
            except Exception as e:
                print(f"恢复原始配置时出错: {e}")


def validate_extrinsic_matrix(source_dataset, target_dataset, extrinsic_matrix):
    """
    使用源数据集计算的外参矩阵验证目标数据集上的表现
    """
    print(f"使用 {source_dataset} 的外参矩阵验证 {target_dataset} ...")
    
    # 保存原始配置
    original_config = None
    try:
        config = get_config()
        original_config = {
            'target_path': config.target_path,
            'cloud': config.cloud,
            'image': config.image,
            'label': config.label
        }
        # 更新配置以使用目标数据集
        config.target_path = target_dataset
        target = Target(target_dataset)
        config.cloud = target.cloud
        config.image = target.image
        config.label = target.label
        
        print(f"更新配置: cloud={config.cloud}, image={config.image}, label={config.label}")
    except Exception as e:
        print(f"更新配置时出错: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    try:
        # 创建相机和激光雷达对象
        cam = Camera()
        lid = Lidar()
        
        # 获取目标数据集的点
        caliboard3D = lid.target_corners()
        caliboard2D = cam.get_caliboard()
        
        print(f"3D点数量: {len(caliboard3D) if caliboard3D is not None else 0}")
        print(f"2D点数量: {len(caliboard2D) if caliboard2D is not None else 0}")
        
        if caliboard3D is None or len(caliboard3D) == 0:
            print(f"目标数据集 {target_dataset} 中未找到有效的3D点")
            return None
            
        if caliboard2D is None or len(caliboard2D) == 0:
            print(f"目标数据集 {target_dataset} 中未找到有效的2D点")
            return None
            
        if len(caliboard3D) != len(caliboard2D):
            print(f"目标数据集 {target_dataset} 中3D点和2D点数量不匹配: {len(caliboard3D)} vs {len(caliboard2D)}")
            return None
            
        # 转换为numpy数组
        object_points = np.array(caliboard3D, dtype=np.float32)
        image_points = np.array(caliboard2D, dtype=np.float32)
        
        # 相机内参和畸变系数
        camera_matrix = cam.intrinsic
        dist_coeffs = cam.distortion_coefficients.reshape(1, -1)  # 确保形状正确
        
        print(f"对象点坐标范围: X[{np.min(object_points[:, 0]):.3f}, {np.max(object_points[:, 0]):.3f}], "
              f"Y[{np.min(object_points[:, 1]):.3f}, {np.max(object_points[:, 1]):.3f}], "
              f"Z[{np.min(object_points[:, 2]):.3f}, {np.max(object_points[:, 2]):.3f}]")
        print(f"图像点坐标范围: U[{np.min(image_points[:, 0]):.3f}, {np.max(image_points[:, 0]):.3f}], "
              f"V[{np.min(image_points[:, 1]):.3f}, {np.max(image_points[:, 1]):.3f}]")
        
        # 应用坐标系变换 (ROS到OpenCV): (x, y, z) -> (z, -x, -y)
        # 这里我们手动应用坐标变换以确保一致性
        transformed_object_points = np.column_stack([
            object_points[:, 2],    # z -> x
            -object_points[:, 0],   # -x -> y
            -object_points[:, 1]    # -y -> z
        ])
        
        # 从外参矩阵提取旋转和平移向量
        rvec = cv2.Rodrigues(extrinsic_matrix[:3, :3])[0]
        tvec = extrinsic_matrix[:3, 3]
        
        print(f"使用的RVEC: {rvec.flatten()}")
        print(f"使用的TVEC: {tvec}")
        
        # 投影3D点到图像
        projected_points, _ = cv2.projectPoints(
            transformed_object_points, rvec, tvec, camera_matrix, dist_coeffs)
        
        # 计算重投影误差
        projected_points = projected_points.reshape(-1, 2)
        reprojection_errors = np.sqrt(np.sum((image_points - projected_points) ** 2, axis=1))
        mean_error = np.mean(reprojection_errors)
        
        print(f"  重投影误差: 平均 {mean_error:.2f} 像素, 最小 {np.min(reprojection_errors):.2f} 像素, 最大 {np.max(reprojection_errors):.2f} 像素")
        print(f"  使用点数: {len(reprojection_errors)}")
        
        # 生成验证图片
        img = cv2.cvtColor(np.array(cam.image), cv2.COLOR_RGB2BGR)
        height, width = img.shape[:2]
        
        # 绘制原始点（绿色）
        for point in image_points:
            cv2.circle(img, tuple(point.astype(int)), 5, (0, 255, 0), -1)
        
        # 绘制重投影点（红色）
        for point in projected_points:
            cv2.circle(img, tuple(point.astype(int)), 5, (0, 0, 255), -1)
        
        # 连接对应点对
        for pt1, pt2 in zip(image_points, projected_points):
            cv2.line(img, tuple(pt1.astype(int)), tuple(pt2.astype(int)), (255, 0, 0), 1)
        
        # 添加文字信息
        text = f'Mean Error: {mean_error:.2f}px'
        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        text2 = f'{os.path.basename(source_dataset)} -> {os.path.basename(target_dataset)}'
        cv2.putText(img, text2, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # 确保eval_images目录存在
        os.makedirs('eval_images', exist_ok=True)
        
        # 保存图片
        image_filename = f'eval_images/{os.path.basename(source_dataset)}_to_{os.path.basename(target_dataset)}.png'
        cv2.imwrite(image_filename, img)
        print(f"  验证图像已保存至: {image_filename}")
        
        return mean_error
        
    except Exception as e:
        print(f"验证 {source_dataset} -> {target_dataset} 时出错: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # 恢复原始配置
        if original_config:
            try:
                config = get_config()
                config.target_path = original_config['target_path']
                config.cloud = original_config['cloud']
                config.image = original_config['image']
                config.label = original_config['label']
                print("已恢复原始配置")
            except Exception as e:
                print(f"恢复原始配置时出错: {e}")


def main():
    """
    实现多数据集评估：
    1. 从config.json中读取相机内参、畸变系数和激光雷达外参矩阵
    2. 分别计算data/d1和data/d2对应的外参矩阵
    3. 交叉验证(使用从一帧计算出的外参矩阵验证到另一帧上)
    """
    print("开始多数据集标定评估...")
    
    # 定义数据集路径
    datasets = {
        'd1': 'data/d1',
        'd2': 'data/d2',
        'd3': 'data/d3'
    }
    
    # 检查数据集是否存在
    for name, path in datasets.items():
        if not os.path.exists(path):
            print(f"警告: 数据集路径 {path} 不存在")
            return
    
    # 存储每个数据集计算出的外参矩阵
    extrinsic_matrices = {}
    
    # 为每个数据集计算外参矩阵
    for name, path in datasets.items():
        print(f"\n=== 计算 {name} 数据集的外参矩阵 ===")
        extrinsic_matrix = compute_extrinsic_matrix(path)
        if extrinsic_matrix is not None:
            extrinsic_matrices[name] = extrinsic_matrix
        else:
            print(f"无法计算 {name} 数据集的外参矩阵")
    
    # 交叉验证
    print("\n=== 交叉验证 ===")
    validation_results = {}
    
    for source_name, source_matrix in extrinsic_matrices.items():
        validation_results[source_name] = {}
        for target_name in datasets.keys():
            if source_name != target_name:
                print(f"\n--- 使用 {source_name} 的外参矩阵验证 {target_name} ---")
                error = validate_extrinsic_matrix(datasets[source_name], datasets[target_name], source_matrix)
                validation_results[source_name][target_name] = error
                
                if error is not None:
                    if error < 10:
                        print(f"✓ 验证通过: 误差 {error:.2f} < 10 像素")
                    else:
                        print(f"✗ 验证失败: 误差 {error:.2f} >= 10 像素")
                else:
                    print(f"? 验证无法完成")
    
    # 输出总结
    print("\n=== 验证结果总结 ===")
    for source_name, targets in validation_results.items():
        print(f"\n使用 {source_name} 数据集标定的外参矩阵:")
        for target_name, error in targets.items():
            if error is not None:
                status = "通过" if error < 10 else "失败"
                print(f"  验证 {target_name}: {error:.2f} 像素 [{status}]")
            else:
                print(f"  验证 {target_name}: 无法计算")


if __name__ == "__main__":
    main()