import json
import open3d as o3d
import numpy as np
import sklearn as sk

import utils.draw_box as draw

try:
    from calibpy.fast_config import get_config
    from calibpy.utils.rotation import rotation_matrix
except:
    from fast_config import get_config
    from utils.rotation import rotation_matrix

from rdsend.client import ClientSender
class BoundingBox:
    def __init__(self):
        self.position = {"x": 0, "y": 0, "z": 0}
        self.rotation = {"x": 0, "y": 0, "z": 0}
        self.scale = {"x": 0, "y": 0, "z": 0}

    def from_file(self, file_path):
        with open(file_path, 'r') as f:
            file = json.load(f)
        psr = file[0]["psr"]
        self.position = psr["position"]
        self.rotation = psr["rotation"]
        self.scale = psr["scale"]

    def corners(self):
        return box_corners(self.position, self.scale, self.rotation)

    def inside(self, point):
        """
        检查点是否在由位置、旋转和尺度定义的盒子内
        点云和标签使用相同坐标系，但仍需要正确的坐标变换
        """
        # # 将点转换到盒子坐标系
        # dx = point[0] - self.position["x"]
        # dy = point[1] - self.position["y"]
        # dz = point[2] - self.position["z"]
        
        # # 根据欧拉角创建旋转矩阵 (ZYX约定)
        # cos_z, sin_z = np.cos(self.rotation["z"]), np.sin(self.rotation["z"])
        # cos_y, sin_y = np.cos(self.rotation["y"]), np.sin(self.rotation["y"])
        # cos_x, sin_x = np.cos(self.rotation["x"]), np.sin(self.rotation["x"])
        
        # # 旋转矩阵
        # rot_z = np.array([[cos_z, -sin_z, 0], [sin_z, cos_z, 0], [0, 0, 1]])
        # rot_y = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]])
        # rot_x = np.array([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]])
        
        # # 组合旋转矩阵 (R = Rz * Ry * Rx)
        # rotation_matrix = np.dot(rot_x, np.dot(rot_y, rot_z))
        
        # # 对点应用逆变换(旋转矩阵的转置)，将列表转换为numpy数组后再进行转置操作
        # local_point = np.dot(rotation_matrix.T, np.array([dx, dy, dz]).T)
        
        # # 检查点是否在盒子内
        # return (
        #     abs(local_point[0]) <= self.scale["x"] / 2 and
        #     abs(local_point[1]) <= self.scale["y"] / 2 and
        #     abs(local_point[2]) <= self.scale["z"] / 2
        # )
        
        rotation = rotation_matrix(self.rotation["x"], self.rotation["y"], self.rotation["z"])
        matrix = np.array([
            [rotation[0,0], rotation[0,1], rotation[0,2], self.position["x"]],
            [rotation[1,0], rotation[1,1], rotation[1,2], self.position["y"]],
            [rotation[2,0], rotation[2,1], rotation[2,2], self.position["z"]],
            [0, 0, 0, 1]
        ])
        
        point = np.linalg.inv(matrix) @ np.array([point[0], point[1], point[2], 1]).T
        
        return (
            abs(point[0]) <= self.scale["x"] / 2 and
            abs(point[1]) <= self.scale["y"] / 2 and
            abs(point[2]) <= self.scale["z"] / 2
        )

class Lidar:
    def __init__(self):
        config = get_config()
        self.extrinsic = np.array(config.get_lidar_extrinsic()).reshape(4, 4)
        self.cloud = o3d.io.read_point_cloud(config.cloud).points
        bx = BoundingBox()
        bx.from_file(config.label)
        self.bounding_box = bx
        
        # 添加调试信息
        print(f"LiDAR点云文件: {config.cloud}")
        print(f"LiDAR标签文件: {config.label}")
        print(f"LiDAR点云点数: {len(self.cloud)}")
        
    def target_cloud_self(self):
        sender = ClientSender()
        sender.connect()
        points = [point for point in self.cloud if self.bounding_box.inside(point)]
        for point in points:
            sender.send_point(point[0], -point[2], -point[1])
        return points
    
    def target_cloud(self):
        """
        获取LiDAR坐标系中的目标点云
        注意：这里返回的是LiDAR坐标系中的点，不是世界坐标系中的点
        """
        points = self.target_cloud_self()
        return self.to_world(points)
    
    def target_corners(self):
        """
        获取LiDAR坐标系中的棋盘格角点，不进行额外的坐标变换
        注意：此方法直接返回LiDAR坐标系中的点，坐标变换将在后续处理中进行
        """
        
        sender = ClientSender()
        sender.connect()
        
        # cloud = self.target_cloud_self()
        # print(f"LiDAR目标点云数量: {len(cloud)}")
        
        # # 创建一个Open3D点云对象并计算定向包围盒
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(cloud)
        # obb = pcd.get_oriented_bounding_box()
        
        # # 提取包围盒参数
        # center = obb.get_center()
        # extent = obb.extent  # XYZ三个方向的尺寸
        # rotation_matrix = obb.R  # 3x3旋转矩阵
        
        
        # corners = np.array([
        #     [extent[0] / 2, extent[1] / 2, extent[2] / 2],
        #     [extent[0] / 2, extent[1] / 2, -extent[2] / 2],
        #     [extent[0] / 2, -extent[1] / 2, extent[2] / 2],
        #     [extent[0] / 2, -extent[1] / 2, -extent[2] / 2],
        #     [-extent[0] / 2, extent[1] / 2, extent[2] / 2],
        #     [-extent[0] / 2, extent[1] / 2, -extent[2] / 2],
        #     [-extent[0] / 2, -extent[1] / 2, extent[2] / 2],
        #     [-extent[0] / 2, -extent[1] / 2, -extent[2] / 2]
        # ])
        
        # corners = [(rotation_matrix @ point.T).T for point in corners]
        # corners = [point + center for point in corners]
        
        # index = draw.edges()
        
        # for idx in index:
        #     start = [corners[idx[0]][0], -corners[idx[0]][1], -corners[idx[0]][2]]
        #     end = [corners[idx[1]][0], -corners[idx[1]][1], -corners[idx[1]][2]]
        #     sender.send_segment(start, end)
        
        # # 将旋转矩阵转换为欧拉角
        # rotation = rotation_matrix_to_euler_angles(rotation_matrix)
        
        # # 构建包围盒参数字典
        # box = {
        #     'position': {"x": center[0], "y": center[1], "z": center[2]},
        #     'scale': {"x": extent[0], "y": extent[1], "z": extent[2]},
        #     'rotation': rotation
        # }
        
        # # 使用原来的min_box函数（基于PCA）
        # box = min_box(cloud)
        
        # 使用cloud.py中的逻辑计算包围盒角点
        # corners = box_corners(box["position"], box["scale"], box["rotation"])
        
        
        # print(f"包围盒8个角点: {corners}")
        # sender = ClientSender()
        # sender.connect()
        
        # index = draw.edges()
        
        # for idx in index:
        #     start = [corners[idx[0]][0], -corners[idx[0]][1], -corners[idx[0]][2]]
        #     end = [corners[idx[1]][0], -corners[idx[1]][1], -corners[idx[1]][2]]
        #     sender.send_segment(start, end)
        
        # corner_board = corner_caliboard(box["position"], box["rotation"], box["scale"])
        # print(f"棋盘格角点数量: {len(corner_board)}")
        
        corners = self.bounding_box.corners()
        corner_board = corner_caliboard(self.bounding_box.position, self.bounding_box.rotation, self.bounding_box.scale)
        
        # for point in corner_board:
        #     sender.send_point(point[0], -point[2], -point[1])
        
        return corner_board
        # return self.to_world(corner_board)
    
    def to_world(self, points):
        # rot = self.extrinsic[:3, :3]
        # result = []
        # for point in points:
        #     rotated_point = np.dot(rot.T, point)
        #     # point = [-point[1], -point[2], point[0]]
        #     # result.append(point)
        #     result.append(rotated_point)
        # result = [[point[0], -point[1], -point[2]] for point in points]
        # return np.array(result)

        return [(self.extrinsic @ np.append(point, 1))[:3] for point in points]

    
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
    
    # 使用固定步长而不是按比例分配，确保棋盘格点均匀分布
    # 这样可以提高OpenCV solvePnP算法的准确性
    # fixed_step = min(width, length) / 10  # 使用较小尺寸的1/10作为固定步长
    fixed_step = 0.1
    
    # 计算棋盘格的边界范围
    width_range = fixed_step * 6  # 6个间隔
    length_range = fixed_step * 7  # 7个间隔
    
    # 计算基准点（左上角）
    width_base = -width_range / 2
    length_base = -length_range / 2
    
    points = []
    
    # 生成棋盘格点 (6行 x 7列)，使用均匀分布的固定步长
    for i in range(1, 7):
        for j in range(1, 8):
            # 创建局部坐标的初始点 (在xy平面)
            local_point = [0, 0, 0]
            local_point[["x", "y", "z"].index(width_axis)] = width_base + i * fixed_step
            local_point[["x", "y", "z"].index(length_axis)] = length_base + j * fixed_step
            
            points.append(local_point)
            
    result = []
    
    # 应用旋转和平移到每个点
    for point in points:
        rotated_point = np.dot(rotation_matrix, point)
        rotated_point[0] += position["x"]
        rotated_point[1] += position["y"]
        rotated_point[2] += position["z"]
        result.append(rotated_point)
    
    # 返回numpy数组而不是普通列表
    return np.array(result)

def box_corners(position, scale, rotation):
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


def min_box(cloud):
    """
    计算点云的最小包围盒（Oriented Bounding Box）
    :param cloud: 点云数据列表
    :return: 包围盒参数 (中心点, 尺寸, 旋转)
    """
    if len(cloud) == 0:
        return None
        
    # 转换为numpy数组
    points = np.array(cloud, dtype=np.float32)
    
    # 计算协方差矩阵
    cov_matrix = np.cov(points, rowvar=False)
    
    # 计算特征向量和特征值
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # 按特征值大小排序特征向量
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    
    # 确保构成右手坐标系
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, -1] *= -1
    
    # 将点云变换到主成分坐标系
    transformed_points = points @ eigenvectors
    
    # 在主成分坐标系中计算AABB
    min_coords = np.min(transformed_points, axis=0)
    max_coords = np.max(transformed_points, axis=0)
    
    scale = {
        'x': max_coords[0] - min_coords[0],
        'y': max_coords[1] - min_coords[1],
        'z': max_coords[2] - min_coords[2]
    }
    
    center_local = (min_coords + max_coords) / 2
    
    # 将中心点变换回世界坐标系
    center_world = eigenvectors @ center_local
    
    # 从特征向量计算欧拉角
    rotation = rotation_matrix_to_euler_angles(eigenvectors)
    
    return {
        'position': {"x": center_world[0], "y": center_world[1], "z": center_world[2]},
        'scale': scale,
        'rotation': rotation
    }


def rotation_matrix_to_euler_angles(R):
    """
    将旋转矩阵转换为欧拉角 (ZYX约定)
    :param R: 3x3 旋转矩阵
    :return: 欧拉角字典
    """
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    
    singular = sy < 1e-6
    
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
        
    return {"x": x, "y": y, "z": z}



