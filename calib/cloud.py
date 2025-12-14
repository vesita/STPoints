import os
import sys
import open3d as o3d
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def inside(point, position, rotation, scale):
    """
    检查点是否在由位置、旋转和尺度定义的盒子内
    点云和标签使用相同坐标系，但仍需要正确的坐标变换
    """
    # 将点转换到盒子坐标系
    dx = point[0] - position["x"]
    dy = point[1] - position["y"]
    dz = point[2] - position["z"]
    
    # 根据欧拉角创建旋转矩阵 (ZYX约定)
    cos_z, sin_z = np.cos(rotation["z"]), np.sin(rotation["z"])
    cos_y, sin_y = np.cos(rotation["y"]), np.sin(rotation["y"])
    cos_x, sin_x = np.cos(rotation["x"]), np.sin(rotation["x"])
    
    # 旋转矩阵
    rot_z = np.array([[cos_z, -sin_z, 0], [sin_z, cos_z, 0], [0, 0, 1]])
    rot_y = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]])
    rot_x = np.array([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]])
    
    # 组合旋转矩阵 (R = Rz * Ry * Rx)
    rotation_matrix = np.dot(rot_x, np.dot(rot_y, rot_z))
    
    # 对点应用逆变换(旋转矩阵的转置)
    local_point = np.dot(rotation_matrix.T, [dx, dy, dz])
    
    # 检查点是否在盒子内
    return (
        abs(local_point[0]) <= scale["x"] / 2 and
        abs(local_point[1]) <= scale["y"] / 2 and
        abs(local_point[2]) <= scale["z"] / 2
    )

def box_corners(position, rotation, scale):
    original = [[-scale["x"] / 2, -scale["y"] / 2, -scale["z"] / 2],
                [scale["x"] / 2, -scale["y"] / 2, -scale["z"] / 2],
                [scale["x"] / 2, -scale["y"] / 2, scale["z"] / 2],
                [-scale["x"] / 2, -scale["y"] / 2, scale["z"] / 2],
                [-scale["x"] / 2, scale["y"] / 2, -scale["z"] / 2],
                [scale["x"] / 2, scale["y"] / 2, -scale["z"] / 2],
                [scale["x"] / 2, scale["y"] / 2, scale["z"] / 2],
                [-scale["x"] / 2, scale["y"] / 2, scale["z"] / 2]]
    
    # 根据欧拉角创建旋转矩阵 (ZYX约定)
    cos_z, sin_z = np.cos(rotation["z"]), np.sin(rotation["z"])
    cos_y, sin_y = np.cos(rotation["y"]), np.sin(rotation["y"])
    cos_x, sin_x = np.cos(rotation["x"]), np.sin(rotation["x"])
    
    # 旋转矩阵
    rot_z = np.array([[cos_z, -sin_z, 0], [sin_z, cos_z, 0], [0, 0, 1]])
    rot_y = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]])
    rot_x = np.array([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]])
    
    # 组合旋转矩阵 (R = Rz * Ry * Rx)
    rotation_matrix = np.dot(rot_x, np.dot(rot_y, rot_z))
    
    points = []
    for point in original:
        rotated_point = np.dot(rotation_matrix, point)
        rotated_point[0] += position["x"]
        rotated_point[1] += position["y"]
        rotated_point[2] += position["z"]
        points.append(rotated_point)
    return np.array(points)

def draw_box_lines(ax, corners):
    """
    在3D图中绘制包围盒的边线
    """
    # 定义包围盒的12条边
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # 底面
        [4, 5], [5, 6], [6, 7], [7, 4],  # 顶面
        [0, 4], [1, 5], [2, 6], [3, 7]   # 垂直边
    ]
    
    for edge in edges:
        points = corners[edge, :]
        ax.plot3D(points[:, 0], points[:, 1], points[:, 2], 'red')

def main(cloud, position, rotation, scale):
    """
    筛选在标记区域内的点
    """
    points_inside = []
    for point in cloud:
        if inside(point, position, rotation, scale):
            points_inside.append(point)
    return points_inside

def corner_caliboard(position, rotation, scale):
    
    # 根据欧拉角创建旋转矩阵 (ZYX约定)
    cos_z, sin_z = np.cos(rotation["z"]), np.sin(rotation["z"])
    cos_y, sin_y = np.cos(rotation["y"]), np.sin(rotation["y"])
    cos_x, sin_x = np.cos(rotation["x"]), np.sin(rotation["x"])

    # 旋转矩阵
    rot_z = np.array([[cos_z, -sin_z, 0], [sin_z, cos_z, 0], [0, 0, 1]])
    rot_y = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]])
    rot_x = np.array([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]])

    # 组合旋转矩阵 (R = Rz * Ry * Rx)
    rotation_matrix = np.dot(rot_x, np.dot(rot_y, rot_z))
    
    # 根据实际的x、y、z轴尺寸确定棋盘格方向，而不是简单的排序
    # 找到最长和次长的轴
    scale_items = [("x", scale["x"]), ("y", scale["y"]), ("z", scale["z"])]
    scale_items.sort(key=lambda x: x[1], reverse=True)
    
    # 最长轴作为length方向，次长轴作为width方向
    length_axis = scale_items[0][0]  # 最长轴名称 ("x", "y", 或 "z")
    width_axis = scale_items[1][0]   # 次长轴名称
    
    # 获取对应的尺寸值
    length = scale[length_axis]
    width = scale[width_axis]
    
    length = 0.1 * (7 + 2)
    width = 0.1 * (6 + 2)
    
    # 计算棋盘格步长
    width_step = width / (6 + 2)
    length_step = length / (7 + 2)
    
    width_base = -width / 2
    length_base = -length / 2
    
    points = []
    
    # 生成棋盘格点 (6行 x 7列)
    for i in range(1, 7):
        for j in range(1, 8):
            # 创建局部坐标的初始点 (在xy平面)
            local_point = [0, 0, 0]
            local_point[["x", "y", "z"].index(width_axis)] = width_base + i * width_step
            local_point[["x", "y", "z"].index(length_axis)] = length_base + j * length_step
            
            points.append(local_point)
            
    result = []
    
    # 应用旋转和平移到每个点
    for point in points:
        rotated_point = np.dot(rotation_matrix, point)
        rotated_point[0] += position[0]  # 修改这里：使用整数索引而不是字符串索引
        rotated_point[1] += position[1]  # 修改这里：使用整数索引而不是字符串索引
        rotated_point[2] += position[2]  # 修改这里：使用整数索引而不是字符串索引
        result.append(rotated_point)
    
    # 返回numpy数组而不是普通列表
    return np.array(result)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        target = "data/d3"
    cloud_path = os.path.join(target, "cloud.pcd")
    label_path = os.path.join(target, "label.json")
    output = os.path.join(target, "caliboard.csv")
    
    clouds = o3d.io.read_point_cloud(cloud_path)
    cloud = clouds.points
    label_file = json.load(open(label_path))
    psr = label_file[0]["psr"]
    position = psr["position"]
    rotation = psr["rotation"]
    scale = psr["scale"]
    points = main(cloud, position, rotation, scale)
    with open(output, "w") as f:
        f.write("x,y,z\n")
        for elem in points:
            f.write("{},{},{}\n".format(*elem))
            
    # 计算包围盒角点
    corners = box_corners(position, rotation, scale)
    
    points_array = np.array(points)
    weighted_center = points_array.mean(axis=0)
    
    # 计算校准板点（42个点 = 6行 × 7列）
    caliboard_points = corner_caliboard(weighted_center, rotation, scale)
    
    # 使用 matplotlib 可视化过滤后的点云和包围盒
    if points:
        points_array = np.array(points)
        fig = plt.figure(figsize=(18, 5))
        
        # 原始角点方法
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(points_array[:, 0], points_array[:, 1], points_array[:, 2], s=1)
        ax1.scatter(corners[:, 0], corners[:, 1], corners[:, 2], c='red', s=50)
        draw_box_lines(ax1, corners)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('Original Corner Calculation')
        
        # 显示校准板点（42个点）
        ax3 = fig.add_subplot(122, projection='3d')
        ax3.scatter(points_array[:, 0], points_array[:, 1], points_array[:, 2], s=1)
        ax3.scatter(caliboard_points[:, 0], caliboard_points[:, 1], caliboard_points[:, 2], c='green', s=30)
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        ax3.set_title('Calibration Board Points (42 points)')
        
        plt.tight_layout()
        plt.show()

        # 显示角点坐标对比
        print("\n原始角点:")
        for i, corner in enumerate(corners):
            print(f"  点 {i}: ({corner[0]:.3f}, {corner[1]:.3f}, {corner[2]:.3f})")
            
            
        print(f"\n校准板点数量: {len(caliboard_points)}")
        print("前几个校准板点:")
        for i in range(min(5, len(caliboard_points))):
            point = caliboard_points[i]
            print(f"  点 {i}: ({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f})")
    else:
        print("在框内未找到点")