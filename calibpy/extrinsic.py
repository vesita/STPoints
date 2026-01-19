import cv2
import numpy as np
import json
import os
import shutil
from datetime import datetime

try:
    import camera
    import lidar
    import coordinate
except:
    import calibpy.camera as camera
    import calibpy.lidar as lidar
    import calibpy.coordinate as coordinate

# from rdsend.client import ClientSender

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


def compute_extrinsics_with_coordinate_method(caliboard3D, caliboard2D, camera_matrix, dist_coeffs):
    """
    使用 coordinate.py 中的函数计算外参矩阵，不使用OpenCV的solvePnP
    
    Args:
        caliboard3D: 3D标定板点坐标 (LiDAR坐标系下的真实3D点)
        caliboard2D: 2D图像中标定板点坐标 (图像平面坐标)
        camera_matrix: 相机内参矩阵
        dist_coeffs: 畸变系数
        
    Returns:
        extrinsic_matrix: 4x4外参矩阵
    """
    if len(caliboard3D) >= 3:
        print("使用LiDAR 3D点和图像2D点计算外参矩阵...")
        
        # 只取前三对点用于计算
        src_points = np.array(caliboard3D[:3])  # LiDAR坐标系下的3D点
        target_2d_points = np.array(caliboard2D[:3])  # 图像中的2D点
        
        # 将2D点去畸变并转换为相机坐标系下的归一化坐标
        undistorted_2d = cv2.undistortPoints(
            target_2d_points.astype(np.float32),
            camera_matrix.astype(np.float32),
            dist_coeffs.astype(np.float32)
        ).reshape(-1, 2)
        
        # 计算LiDAR坐标系中点之间的距离（这些距离在坐标变换下保持不变）
        ld_dists = []
        for i in range(len(src_points)):
            for j in range(i+1, len(src_points)):
                dist = np.linalg.norm(src_points[i] - src_points[j])
                ld_dists.append(dist)
        
        import scipy.optimize as opt
        
        def cost_function(params):
            # params包含3个相机坐标系中的3D点的坐标 (x1, y1, z1, x2, y2, z2, x3, y3, z3)
            camera_points = params.reshape(-1, 3)
            
            # 检查是否有任何点位于相机后方（Z <= 0），如果有则施加极大惩罚
            z_coords = camera_points[:, 2]
            invalid_z_mask = z_coords <= 0
            if np.any(invalid_z_mask):
                penalty = 10000 * np.sum(np.abs(z_coords[invalid_z_mask]))
            else:
                penalty = 0.0
            
            # 计算相机坐标系中点之间的距离
            cam_dists = []
            for i in range(len(camera_points)):
                for j in range(i+1, len(camera_points)):
                    dist = np.linalg.norm(camera_points[i] - camera_points[j])
                    cam_dists.append(dist)
            
            # 距离一致性误差
            dist_error = np.sum((np.array(cam_dists) - np.array(ld_dists)) ** 2)
            
            # 投影误差
            projected_2d, _ = cv2.projectPoints(
                camera_points.reshape(-1, 1, 3).astype(np.float32),
                np.zeros(3), np.zeros(3),
                camera_matrix.astype(np.float32),
                dist_coeffs.astype(np.float32)
            )
            projected_2d = projected_2d.reshape(-1, 2)
            
            proj_error = np.sum((projected_2d - target_2d_points) ** 2)
            
            return dist_error + 0.1 * proj_error + penalty  # 给投影误差较小权重，加上Z方向约束的惩罚
        
        # 使用多个初始猜测值来提高优化成功率
        best_result = None
        best_error = float('inf')
        
        # 不同深度的初始猜测
        depths = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
        
        # 将2D点转换为相机坐标系中的方向向量
        rays = []
        for pt2d_norm in undistorted_2d:
            ray = np.array([pt2d_norm[0], pt2d_norm[1], 1.0])
            ray /= np.linalg.norm(ray)  # 单位化
            rays.append(ray)
        
        rays = np.array(rays)
        
        # 尝试不同的初始猜测
        for depth in depths:
            # 使用射线方向和当前深度作为初始估计
            initial_guess = []
            for ray in rays:
                pt = ray * depth
                initial_guess.extend(pt)
            
            initial_guess = np.array(initial_guess)
            
            try:
                # 尝试不同的优化算法
                result_bfgs = opt.minimize(cost_function, initial_guess, method='BFGS')
                result_l_bfgs_b = opt.minimize(cost_function, initial_guess, method='L-BFGS-B')
                
                # 检查哪个结果更好
                for result in [result_bfgs, result_l_bfgs_b]:
                    if result.success:
                        total_error = result.fun
                        
                        if total_error < best_error:
                            best_error = total_error
                            best_result = result
            except:
                continue
        
        # 如果BFGS系列算法失败，尝试使用更稳定的算法
        if best_result is None or not best_result.success:
            for depth in depths:
                initial_guess = []
                for ray in rays:
                    pt = ray * depth
                    initial_guess.extend(pt)
                
                initial_guess = np.array(initial_guess)
                
                try:
                    result_dogleg = opt.minimize(cost_function, initial_guess, method='dogleg')
                    result_trust_ncg = opt.minimize(cost_function, initial_guess, method='trust-ncg')
                    
                    for result in [result_dogleg, result_trust_ncg]:
                        if result.success:
                            total_error = result.fun
                            
                            if total_error < best_error:
                                best_error = total_error
                                best_result = result
                except:
                    continue
        
        if best_result is not None and best_result.success:
            optimal_points = best_result.x.reshape(-1, 3)
            
            # 确保相机坐标系中的点在前方（Z > 0）
            for i in range(len(optimal_points)):
                if optimal_points[i][2] <= 0:
                    optimal_points[i][2] = abs(optimal_points[i][2]) + 0.1  # 确保在前方
            
            # 计算从LiDAR坐标系到相机坐标系的变换
            extrinsic_matrix = coordinate.calculate_transform_from_three_points(
                src_points,      # LiDAR坐标系中的3D点
                optimal_points   # 相机坐标系中的对应3D点
            )
            
            print("使用优化方法计算的变换矩阵 (LiDAR -> Camera):")
            print(extrinsic_matrix)
            
            # 验证变换矩阵
            R, t = coordinate.decompose_transformation_matrix(extrinsic_matrix)
            print(f"\n变换矩阵验证:")
            print(f"  旋转矩阵行列式: {np.linalg.det(R):.6f} (理想值接近1.0)")
            print(f"  平移向量范数: {np.linalg.norm(t):.4f} 米")
            
            # 检查旋转矩阵的正交性
            is_orthogonal = np.allclose(np.dot(R, R.T), np.eye(3), atol=1e-6)
            print(f"  旋转矩阵正交性: {'是' if is_orthogonal else '否'}")
            
            # 验证投影误差
            transformed_3d = coordinate.apply_transform(src_points, extrinsic_matrix)
            projected_2d, _ = cv2.projectPoints(
                transformed_3d.reshape(-1, 1, 3).astype(np.float32),
                np.zeros(3), np.zeros(3),
                camera_matrix.astype(np.float32),
                dist_coeffs.astype(np.float32)
            )
            projected_2d = projected_2d.reshape(-1, 2)
            
            error = np.mean(np.sqrt(np.sum((projected_2d - target_2d_points)**2, axis=1)))
            print(f"  平均投影误差: {error:.4f} 像素")
            
            # 如果投影误差仍然很大，尝试使用OpenCV的PnP算法作为备选方案
            if error > 50:  # 设置一个阈值
                print("优化方法投影误差过大，尝试使用OpenCV的PnP算法...")
                
                # 使用OpenCV的solvePnP
                success, rvec, tvec = cv2.solvePnP(
                    src_points.astype(np.float32),
                    target_2d_points.astype(np.float32),
                    camera_matrix.astype(np.float32),
                    dist_coeffs.astype(np.float32),
                    flags=cv2.SOLVEPNP_P3P
                )
                
                if not success:
                    # 如果P3P失败，尝试EPNP
                    success, rvec, tvec = cv2.solvePnP(
                        src_points.astype(np.float32),
                        target_2d_points.astype(np.float32),
                        camera_matrix.astype(np.float32),
                        dist_coeffs.astype(np.float32),
                        flags=cv2.SOLVEPNP_EPNP
                    )
                
                if success:
                    # 构建变换矩阵
                    rotation_matrix, _ = cv2.Rodrigues(rvec)
                    opencv_extrinsic = np.eye(4)
                    opencv_extrinsic[:3, :3] = rotation_matrix
                    opencv_extrinsic[:3, 3] = tvec.flatten()
                    
                    # 验证OpenCV的结果
                    transformed_3d_opencv = coordinate.apply_transform(src_points, opencv_extrinsic)
                    projected_2d_opencv, _ = cv2.projectPoints(
                        transformed_3d_opencv.reshape(-1, 1, 3).astype(np.float32),
                        np.zeros(3), np.zeros(3),
                        camera_matrix.astype(np.float32),
                        dist_coeffs.astype(np.float32)
                    )
                    projected_2d_opencv = projected_2d_opencv.reshape(-1, 2)
                    
                    opencv_error = np.mean(np.sqrt(np.sum((projected_2d_opencv - target_2d_points)**2, axis=1)))
                    print(f"  OpenCV PnP平均投影误差: {opencv_error:.4f} 像素")
                    
                    if opencv_error < error:
                        print("使用OpenCV PnP结果，因为误差更小")
                        return opencv_extrinsic
                    else:
                        print("使用优化方法结果，尽管误差较大")
                        return extrinsic_matrix
                else:
                    print("OpenCV PnP也失败了，返回当前结果")
                    return extrinsic_matrix
            else:
                return extrinsic_matrix
        else:
            print("优化失败，尝试其他方法...")
            
            # 如果优化失败，尝试使用更简单的几何方法
            # 根据LiDAR点的平均深度来估算相机坐标系中的点
            avg_lidar_depth = np.mean([abs(p[2]) for p in src_points])  # 使用Z坐标的绝对值
            approx_camera_points = [ray * max(avg_lidar_depth, 0.5) for ray in rays]  # 确保最小深度为0.5米
            approx_camera_points = np.array(approx_camera_points)
            
            # 确保所有点都在相机前方
            for i in range(len(approx_camera_points)):
                if approx_camera_points[i][2] <= 0:
                    approx_camera_points[i][2] = abs(approx_camera_points[i][2]) + 0.1
            
            # 计算变换
            extrinsic_matrix = coordinate.calculate_transform_from_three_points(
                src_points,          # LiDAR坐标系中的3D点
                approx_camera_points # 相机坐标系中的近似3D点
            )
            
            print("使用几何方法计算的变换矩阵 (LiDAR -> Camera):")
            print(extrinsic_matrix)
            
            # 验证投影误差
            transformed_3d = coordinate.apply_transform(src_points, extrinsic_matrix)
            projected_2d, _ = cv2.projectPoints(
                transformed_3d.reshape(-1, 1, 3).astype(np.float32),
                np.zeros(3), np.zeros(3),
                camera_matrix.astype(np.float32),
                dist_coeffs.astype(np.float32)
            )
            projected_2d = projected_2d.reshape(-1, 2)
            
            error = np.mean(np.sqrt(np.sum((projected_2d - target_2d_points)**2, axis=1)))
            print(f"  平均投影误差: {error:.4f} 像素")
            
            if error > 100:  # 提高阈值，允许更大的误差
                print("投影误差过大，结果不可靠")
                return None
            
            return extrinsic_matrix
    else:
        print("不足3个点，无法计算外参矩阵")
        return None


def evaluate_extrinsic_for_test_dirs(extrinsic_matrix, test_base_path="test"):
    """
    遍历test目录下的所有子目录，使用给定的外参矩阵将标定板投影到图像上并评估投影效果
    """
    print(f"开始评估 {test_base_path} 目录下所有子目录的外参投影效果...")
    
    print("使用的外参矩阵:")
    print(extrinsic_matrix)
    
    # 获取test目录下的所有子目录
    subdirs = [d for d in os.listdir(test_base_path) 
               if os.path.isdir(os.path.join(test_base_path, d))]
    
    all_errors = []
    
    for subdir in sorted(subdirs):
        subdir_path = os.path.join(test_base_path, subdir)
        
        # 检查必要的文件是否存在
        cloud_file = os.path.join(subdir_path, "cloud.pcd")
        image_file = os.path.join(subdir_path, "image.jpg")
        label_file = os.path.join(subdir_path, "label.json")
        
        if not all(os.path.exists(f) for f in [cloud_file, image_file, label_file]):
            print(f"跳过 {subdir_path}, 缺少必要文件")
            continue
        
        print(f"\n处理 {subdir_path}...")
        
        try:
            # 加载标签文件
            with open(label_file, 'r') as f:
                label_data = json.load(f)
            
            # 假设第一个对象是标定板
            if len(label_data) == 0:
                print(f"标签文件 {label_file} 为空")
                continue
                
            # 获取标定板的PSR参数
            bbox_data = label_data[0]["psr"]  
            pos = bbox_data["position"]
            scale = bbox_data["scale"]
            rot = bbox_data["rotation"]
            
            # 从config.json加载外参矩阵
            with open("./config/config.json", 'r') as f:
                config_data = json.load(f)
            lidar_extrinsic = np.array(config_data["lidar_extrinsic"]).reshape(4, 4)
            
            # 使用corner_caliboard函数生成42个棋盘格角点（类似lidar.py中的逻辑）
            caliboard3D = lidar.corner_caliboard(pos, rot, scale)
            
            # 应用lidar外参变换到世界坐标系
            caliboard3D_world = [(lidar_extrinsic @ np.append(point, 1))[:3] for point in caliboard3D]
            caliboard3D_world = np.array(caliboard3D_world, dtype=np.float32)
            
            # 加载相机参数
            camera_intrinsic = np.array(config_data["camera_intrinsic"]).reshape(3, 3)
            camera_distortion = np.array(config_data["camera_distortion_coefficients"])
            
            # 加载图像并获取2D标定板点
            from PIL import Image
            img = Image.open(image_file)
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            
            # 查找棋盘格角点
            chessboard_size = (6, 7)  # 默认棋盘格大小
            ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
            
            if not ret:
                # 如果没找到角点，尝试其他尺寸
                for size in [(5, 6), (7, 8), (8, 9), (9, 6)]:
                    ret, corners = cv2.findChessboardCorners(gray, size, None)
                    if ret:
                        chessboard_size = size
                        break
            
            if not ret:
                print(f"在 {image_file} 中未找到棋盘格角点")
                continue
                
            # 精细化角点位置
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # 将角点坐标从图像坐标转换为(x, y)坐标形式
            caliboard2D = []
            for corner in corners:
                u, v = corner.ravel()
                caliboard2D.append([u, v])
                
            caliboard2D = np.array(caliboard2D, dtype=np.float32)

            print(f"3D点数量: {len(caliboard3D_world)}, 2D点数量: {len(caliboard2D)}")

            # 确保点的数量相同
            if len(caliboard3D_world) != len(caliboard2D):
                print(f"3D和2D点数量不匹配: {len(caliboard3D_world)} vs {len(caliboard2D)}")
                # 尝试使用最小数量的点
                min_len = min(len(caliboard3D_world), len(caliboard2D))
                caliboard3D_world = caliboard3D_world[:min_len]
                caliboard2D = caliboard2D[:min_len]
                
                if min_len < 4:
                    print("点数太少，无法进行有效的投影评估")
                    continue

            # 使用给定外参矩阵将3D点投影到2D图像
            dist_coeffs = camera_distortion.reshape(1, -1)

            # 应用外参变换 (从LiDAR坐标系到相机坐标系)
            transformed_3d = coordinate.apply_transform(caliboard3D_world, extrinsic_matrix)

            # 将3D点投影到2D图像
            projected_2d, _ = cv2.projectPoints(
                transformed_3d.reshape(-1, 1, 3).astype(np.float32),
                np.zeros(3), np.zeros(3),
                camera_intrinsic.astype(np.float32),
                dist_coeffs.astype(np.float32)
            )
            projected_2d = projected_2d.reshape(-1, 2)

            # 计算投影误差
            error = np.mean(np.sqrt(np.sum((projected_2d - caliboard2D)**2, axis=1)))
            all_errors.append(error)

            print(f"{subdir}: 平均投影误差 = {error:.4f} 像素")

            # 输出前几个点的对比
            print(f"  前3个点的对比 (投影点 vs 实际点):")
            for i in range(min(3, len(projected_2d))):
                print(f"    {i}: ({projected_2d[i][0]:.2f}, {projected_2d[i][1]:.2f}) vs "
                      f"({caliboard2D[i][0]:.2f}, {caliboard2D[i][1]:.2f})")

        except Exception as e:
            print(f"处理 {subdir_path} 时出现错误: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # 输出总体统计
    if all_errors:
        print(f"\n总体评估结果:")
        print(f"  总共处理了 {len(all_errors)} 个目录")
        print(f"  平均投影误差: {np.mean(all_errors):.4f} 像素")
        print(f"  最小投影误差: {np.min(all_errors):.4f} 像素")
        print(f"  最大投影误差: {np.max(all_errors):.4f} 像素")
        print(f"  投影误差标准差: {np.std(all_errors):.4f} 像素")
    else:
        print("没有成功处理任何目录")


def main():
    # client = ClientSender()
    # client.connect()
    
    print("开始相机标定过程...")
    cam = camera.Camera()
    lid = lidar.Lidar()
    
    # cloud = lid.cloud
    # cloud = [(lid.extrinsic @ np.append(point, 1))[:3] for point in cloud]
    # for point in cloud:
    #     client.send_point(point[0], point[2], point[1])
    
    corners = draw.corners_of_psr(lid.bounding_box.position, lid.bounding_box.scale, lid.bounding_box.rotation)
    index = draw.edges()
    
    
    corners = [(lid.extrinsic @ np.append(point, 1))[:3] for point in corners]
    
    for idx in index:
        start = [corners[idx[0]][0], corners[idx[0]][2], corners[idx[0]][1]]
        end = [corners[idx[1]][0], corners[idx[1]][2], corners[idx[1]][1]]
        # client.send_segment(start, end)
    
    print("获取LiDAR生成的棋盘格3D点...")
    caliboard3D = lid.target_corners()
    
    caliboard3D = [(lid.extrinsic @ np.append(point, 1))[:3] for point in caliboard3D]
    
    print("获取图像中的棋盘格2D点...")
    caliboard2D = cam.get_caliboard()
    
    print(f"3D点数量: {len(caliboard3D) if caliboard3D is not None else 'None'}")
    print(f"2D点数量: {len(caliboard2D) if caliboard2D is not None else 'None'}")
    
    # 确保点的数量相同
    if len(caliboard3D) == len(caliboard2D):
        # 将列表转换为 numpy 数组以供后续处理
        caliboard3D = np.array(caliboard3D, dtype=np.float32)
        caliboard2D = np.array(caliboard2D, dtype=np.float32)        
            
        # 确保内参矩阵已定义
        camera_matrix = np.array(cam.intrinsic, dtype=np.float32)
        print(f"相机内参矩阵: \n{camera_matrix}")
        
        # 使用从配置文件加载的畸变系数，确保正确的形状
        dist_coeffs = np.array(cam.distortion_coefficients, dtype=np.float32).reshape(1, -1)
        
        # for point in caliboard3D:
        #     client.send_point(point[0], point[2], point[1])
        
        # for point in caliboard2D:
        #     client.send_point(1, -point[1]/100, -point[0]/100)
        
        # caliboard2D.sort()
        # # 使用 numpy 的随机排列功能替代 shuffle 方法
        # indices = np.random.permutation(caliboardD.shape[0])
        # caliboard3D = caliboard3D[indices]
        
        # 对于NumPy数组，使用argsort来实现类似key的排序
        sort_indices_2d = np.argsort(1024 * caliboard2D[:, 0] + caliboard2D[:, 1])
        caliboard2D = caliboard2D[sort_indices_2d]
        
        # for point in caliboard2D:
        #     print(point)
        
        sort_indices_3d = np.argsort(-1024 * caliboard3D[:, 1] - caliboard3D[:, 2])
        caliboard3D = caliboard3D[sort_indices_3d]
        
        # for point in caliboard3D:
        #     print(point)
            
        caliboard3D = np.array([
            [3.5889852, -0.38507494, -0.34156674],
            [3.5668185, 0.21403857, -0.31765765],
            [3.477577, -0.36975443, -0.82875615],
            # [3.4554105, 0.22935908, -0.80484706]
        ])
        
        # caliboard3D = np.array([
        #     [3.4554105, 0.22935908, -0.80484706],
        #     [3.477577, -0.36975443, -0.82875615],
        #     [3.5668185, 0.21403857, -0.31765765],
        #     [3.5889852, -0.38507494, -0.34156674]
        # ])
        
        caliboard2D = np.array([
            [317.62103, 277.33878],
            [411.76627, 275.93457],
            [317.63156, 199.48656],
            # [410.19217, 199.42244]
        ])
        
        # 尝试使用coordinate.py中的函数计算外参
        print("\n尝试使用coordinate.py中的函数计算外参...")
        extrinsic_matrix_coord = compute_extrinsics_with_coordinate_method(
            caliboard3D, caliboard2D, camera_matrix, dist_coeffs
        )
        
        if extrinsic_matrix_coord is not None:
            print("\n使用coordinate方法计算的外参矩阵:")
            print(extrinsic_matrix_coord)
            
            # 验证变换矩阵
            R, t = coordinate.decompose_transformation_matrix(extrinsic_matrix_coord)
            print(f"\n变换矩阵验证:")
            print(f"  旋转矩阵行列式: {np.linalg.det(R):.6f} (理想值接近1.0)")
            print(f"  平移向量范数: {np.linalg.norm(t):.4f} 米")
            
            # 检查旋转矩阵的正交性
            is_orthogonal = np.allclose(np.dot(R, R.T), np.eye(3), atol=1e-6)
            print(f"  旋转矩阵正交性: {'是' if is_orthogonal else '否'}")
            
            # 验证coordinate.py函数的准确性
            transformed_test_points = coordinate.apply_transform(caliboard3D[:3], extrinsic_matrix_coord)
            print(f"  变换后测试点: {transformed_test_points}")

        # success, rvec, tvec = cv2.solvePnP(
        #     caliboard3D, 
        #     caliboard2D, 
        #     camera_matrix, 
        #     dist_coeffs,
        #     flags=cv2.SOLVEPNP_ITERATIVE
        # )
        
        # if success:
        #     # 计算重投影误差
        #     projected_points, _ = cv2.projectPoints(
        #         caliboard3D, rvec, tvec, camera_matrix, dist_coeffs)
            
        #     projected_points = projected_points.reshape(-1, 2)
        #     # for point in projected_points:
        #     #     client.send_point(1, -point[0]/100, -point[1]/100)
        #     reprojection_error = np.sqrt(np.mean((caliboard2D - projected_points) ** 2))
            
        #     print(f"  方法 pnp 重投影误差: {reprojection_error:.2f} 像素")
            
        #     # 检查旋转矩阵的正交性
        #     rotation_matrix, _ = cv2.Rodrigues(rvec)
            
        #     # 构造4x4的变换矩阵（从LiDAR坐标系到相机坐标系）
        #     extrinsic_matrix = np.eye(4, dtype=np.float32)
        #     extrinsic_matrix[:3, :3] = rotation_matrix
        #     extrinsic_matrix[:3, 3] = tvec.reshape(-1)
            
        #     print("\n外参矩阵详细信息 (LiDAR到相机):")
        #     print("旋转矩阵 R:")
        #     print(rotation_matrix)
        #     print("平移向量 T:", tvec.flatten())
        #     print("\n完整4x4外参矩阵:")
        #     print(extrinsic_matrix)
            
        #     # 验证变换矩阵
        #     print(f"\n变换矩阵验证:")
        #     print(f"  行列式: {np.linalg.det(rotation_matrix):.6f} " 
        #             f"(理想值接近1.0)")
        #     print(f"  平移距离: {np.linalg.norm(tvec):.4f} 米")
            
        #     # 计算重投影误差
        #     projected_points, _ = cv2.projectPoints(
        #         caliboard3D, rvec, tvec, camera_matrix, dist_coeffs)
            
        #     # 计算均方根误差
        #     projected_points = projected_points.reshape(-1, 2)
        #     reprojection_error = np.sqrt(np.mean((caliboard2D - projected_points) ** 2))
            
        #     print(f"最终重投影误差: {reprojection_error} 像素")
            
        #     # 先备份配置文件
        #     print("正在备份当前配置文件...")
        #     if backup_config():
        #         # 保存外参矩阵到配置文件
        #         save_camera_extrinsic(extrinsic_matrix)
        #     else:
        #         print("备份失败，取消保存操作")

        # 将2D点转换为齐次坐标后再进行矩阵运算
        # img_caliboard3D = [np.linalg.inv(cam.intrinsic) @ np.append(point, 1) for point in caliboard2D]
        # img_caliboard3D = [[1, point[0], point[1]] for point in img_caliboard3D]
        
        
        
    # 添加对test目录下所有数据的评估，使用计算出的外参
    if 'extrinsic_matrix_coord' in locals() and extrinsic_matrix_coord is not None:
        print("\n开始评估test目录下所有数据的外参投影效果...")
        evaluate_extrinsic_for_test_dirs(extrinsic_matrix_coord)
    else:
        print("\n没有有效的外参矩阵用于评估")


if __name__ == "__main__":
    main()