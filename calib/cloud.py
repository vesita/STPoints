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


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        target = "data/cam_lid"
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
    
    # 使用 matplotlib 可视化过滤后的点云和包围盒
    if points:
        points_array = np.array(points)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points_array[:, 0], points_array[:, 1], points_array[:, 2], s=1)
        
        # 绘制包围盒角点和边线
        ax.scatter(corners[:, 0], corners[:, 1], corners[:, 2], c='red', s=20)
        draw_box_lines(ax, corners)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Filtered Point Cloud with Bounding Box')
        plt.show()
    else:
        print("No points found inside the box")