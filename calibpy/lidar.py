import json
import open3d as o3d
import numpy as np

try:
    from calibpy.fast_config import get_config
except:
    from fast_config import get_config


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

    def inside(self, point):
        """
        检查点是否在由位置、旋转和尺度定义的盒子内
        点云和标签使用相同坐标系，但仍需要正确的坐标变换
        """
        # 将点转换到盒子坐标系
        dx = point[0] - self.position["x"]
        dy = point[1] - self.position["y"]
        dz = point[2] - self.position["z"]
        
        # 根据欧拉角创建旋转矩阵 (ZYX约定)
        cos_z, sin_z = np.cos(self.rotation["z"]), np.sin(self.rotation["z"])
        cos_y, sin_y = np.cos(self.rotation["y"]), np.sin(self.rotation["y"])
        cos_x, sin_x = np.cos(self.rotation["x"]), np.sin(self.rotation["x"])
        
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
            abs(local_point[0]) <= self.scale["x"] / 2 and
            abs(local_point[1]) <= self.scale["y"] / 2 and
            abs(local_point[2]) <= self.scale["z"] / 2
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


class Lidar:
    def __init__(self):
        config = get_config()
        self.extrinsic = np.array(config.get_lidar_extrinsic()).reshape(4, 4)
        self.cloud = o3d.io.read_point_cloud(config.cloud)
        bx = BoundingBox()
        bx.from_file(config.label)
        self.bounding_box = bx
        
        # 添加调试信息
        print(f"LiDAR点云文件: {config.cloud}")
        print(f"LiDAR标签文件: {config.label}")
        print(f"LiDAR点云点数: {len(self.cloud.points)}")
        
    def target_cloud_self(self):
        return [point for point in self.cloud.points if self.bounding_box.inside(point)]
    
    def target_cloud(self):
        """
        获取LiDAR坐标系中的目标点云
        注意：这里返回的是LiDAR坐标系中的点，不是世界坐标系中的点
        """
        points = self.target_cloud_self()
        # 如果没有点在包围盒内，返回空列表而不是 None
        if len(points) == 0:
            return []
        return points
    
    def target_corners(self):
        """
        获取LiDAR坐标系中的棋盘格角点，不进行额外的坐标变换
        注意：此方法直接返回LiDAR坐标系中的点，坐标变换将在后续处理中进行
        """
        cloud = self.target_cloud()
        print(f"LiDAR目标点云数量: {len(cloud)}")
        
        # 添加更多调试信息
        if len(cloud) > 0:
            points_array = np.array(cloud)
            print(f"点云坐标范围: X[{np.min(points_array[:, 0]):.3f}, {np.max(points_array[:, 0]):.3f}], "
                  f"Y[{np.min(points_array[:, 1]):.3f}, {np.max(points_array[:, 1]):.3f}], "
                  f"Z[{np.min(points_array[:, 2]):.3f}, {np.max(points_array[:, 2]):.3f}]")
        
        # 使用cloud.py中的逻辑计算包围盒角点
        corners = box_corners(
            self.bounding_box.position,
            self.bounding_box.rotation,
            self.bounding_box.scale
        )
        print(f"包围盒8个角点: {corners}")
        
        corner_board = self.corner_caliboard(corners)
        print(f"棋盘格角点数量: {len(corner_board)}")
        
        # 不在此处应用坐标变换，直接返回LiDAR坐标系中的点
        # 坐标变换将在后续的投影计算中进行
        return corner_board
    
    def cloud2box(self, cloud):
        """
        根据输入的点云，计算最小包围盒
        :param cloud: 点云数据列表
        :return: 包围盒参数 (中心点, 尺寸)
        """
        if len(cloud) == 0:
            return None
            
        # 转换为numpy数组
        points = np.array(cloud, dtype=np.float32)
        
        # 计算各维度的最小值和最大值
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        
        # 计算中心点和尺寸
        center = (min_coords + max_coords) / 2
        size = max_coords - min_coords
        
        return {
            'center': center.astype(np.float32),
            'size': size.astype(np.float32)
        }
    
    def box_corners(self, box):
        """
        根据最小包围盒计算包围盒的8个顶点
        :param box: 包围盒参数
        :return: 8个顶点坐标
        """
        if box is None:
            return np.array([])
            
        center = box['center']
        size = box['size']
        
        # 计算半尺寸
        half_size = size / 2
        
        # 计算8个顶点
        corners = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    dx = half_size[0] if i == 1 else -half_size[0]
                    dy = half_size[1] if j == 1 else -half_size[1]
                    dz = half_size[2] if k == 1 else -half_size[2]
                    corner = [
                        center[0] + dx,
                        center[1] + dy,
                        center[2] + dz
                    ]
                    corners.append(corner)
        
        return np.array(corners, dtype=np.float32)
    
    def corner_caliboard(self, corners):
        """
        根据包围盒的8个顶点，计算出6*7的内角点数量
        :param corners: 包围盒的8个顶点
        :return: 棋盘格角点坐标 (6*7=42个点)
        """
        if len(corners) == 0:
            return []
            
        # 检查corners是否有效
        if np.isnan(corners).any() or np.isinf(corners).any():
            print("警告：包围盒角点包含无效值")
            return []
        
        # 棋盘格是6列7行的内角点，总共42个点
        rows, cols = 7, 6  # 7行6列
        
        # 找到包围盒的不同表面
        # 按照x, y, z坐标分别聚类
        x_coords = corners[:, 0]
        y_coords = corners[:, 1]
        z_coords = corners[:, 2]
        
        # 找到x, y, z的极值
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        z_min, z_max = np.min(z_coords), np.max(z_coords)
        
        print(f"包围盒X范围: [{x_min:.3f}, {x_max:.3f}]")
        print(f"包围盒Y范围: [{y_min:.3f}, {y_max:.3f}]")
        print(f"包围盒Z范围: [{z_min:.3f}, {z_max:.3f}]")
        
        # 总是选择YZ平面（固定X坐标）以确保一致性
        # 这样可以确保不同数据集之间的坐标处理一致性
        fixed_val = x_max if np.sum(corners[:, 0] > (x_min + x_max) / 2) > 2 else x_min
        surface_points = corners[np.abs(corners[:, 0] - fixed_val) < 1e-3]
        plane_type = "YZ"
        fixed_coord = fixed_val
            
        print(f"选择的表面: {plane_type} 平面, 固定坐标值: {fixed_coord:.3f}")
        print(f"表面点数量: {len(surface_points)}")
        print(f"表面点坐标:")
        for i, point in enumerate(surface_points):
            print(f"  点{i}: ({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f})")
        
        if len(surface_points) < 4:
            print(f"错误：表面点数不足，只有{len(surface_points)}个点")
            return []
        
        # 在选定的表面上找到四个角点
        # 使用更稳定的方法找到角点
        if plane_type == "YZ":
            # YZ平面，使用y和z坐标
            coords = [(p[1], p[2]) for p in surface_points]  # (y, z)
        elif plane_type == "XZ":
            # XZ平面，使用x和z坐标
            coords = [(p[0], p[2]) for p in surface_points]  # (x, z)
        else:
            # XY平面，使用x和y坐标
            coords = [(p[0], p[1]) for p in surface_points]  # (x, y)
        
        # 找到四个角点：最小/最大y和z（或x和z，或x和y）组合
        coord_array = np.array(coords)
        min_y_idx = np.argmin(coord_array[:, 0])
        max_y_idx = np.argmax(coord_array[:, 0])
        min_z_idx = np.argmin(coord_array[:, 1])
        max_z_idx = np.argmax(coord_array[:, 1])
        
        # 构造四个角点
        top_left_idx = min_y_idx if coord_array[min_y_idx, 1] <= coord_array[min_z_idx, 1] else min_z_idx
        top_right_idx = min_y_idx if coord_array[min_y_idx, 1] >= coord_array[max_z_idx, 1] else max_z_idx
        bottom_left_idx = max_y_idx if coord_array[max_y_idx, 1] <= coord_array[min_z_idx, 1] else min_z_idx
        bottom_right_idx = max_y_idx if coord_array[max_y_idx, 1] >= coord_array[max_z_idx, 1] else max_z_idx
        
        # 确保索引不重复
        indices = list(set([top_left_idx, top_right_idx, bottom_left_idx, bottom_right_idx]))
        if len(indices) < 4:
            # 如果索引重复，则使用简单的极值方法
            y_vals = coord_array[:, 0]
            z_vals = coord_array[:, 1]
            
            # 找到四个象限的点
            y_median = np.median(y_vals)
            z_median = np.median(z_vals)
            
            quadrants = [[] for _ in range(4)]  # topLeft, topRight, bottomLeft, bottomRight
            for i, (y, z) in enumerate(coord_array):
                if y <= y_median and z <= z_median:
                    quadrants[0].append((i, y, z))
                elif y <= y_median and z > z_median:
                    quadrants[1].append((i, y, z))
                elif y > y_median and z <= z_median:
                    quadrants[2].append((i, y, z))
                else:
                    quadrants[3].append((i, y, z))
            
            # 从每个象限中选择一个代表点（距离质心最近的点）
            representative_points = []
            for quadrant in quadrants:
                if quadrant:
                    # 计算象限质心
                    quad_y = np.mean([p[1] for p in quadrant])
                    quad_z = np.mean([p[2] for p in quadrant])
                    # 找到距离质心最近的点
                    distances = [np.sqrt((p[1]-quad_y)**2 + (p[2]-quad_z)**2) for p in quadrant]
                    min_idx = np.argmin(distances)
                    representative_points.append(quadrant[min_idx][0])
                else:
                    # 如果象限为空，使用最近的点
                    representative_points.append(0)
            
            top_left_idx, top_right_idx, bottom_left_idx, bottom_right_idx = representative_points[:4]
        
        top_left = surface_points[top_left_idx]
        top_right = surface_points[top_right_idx]
        bottom_left = surface_points[bottom_left_idx]
        bottom_right = surface_points[bottom_right_idx]
        
        print(f"确定的四个角点:")
        print(f"  左上角: ({top_left[0]:.3f}, {top_left[1]:.3f}, {top_left[2]:.3f})")
        print(f"  右上角: ({top_right[0]:.3f}, {top_right[1]:.3f}, {top_right[2]:.3f})")
        print(f"  左下角: ({bottom_left[0]:.3f}, {bottom_left[1]:.3f}, {bottom_left[2]:.3f})")
        print(f"  右下角: ({bottom_right[0]:.3f}, {bottom_right[1]:.3f}, {bottom_right[2]:.3f})")
        
        # 生成棋盘格点 (6列7行)
        board_points = []
        for i in range(rows):  # 7行
            for j in range(cols):  # 6列
                # 双线性插值计算点坐标
                u = j / (cols - 1) if cols > 1 else 0  # 列比例
                v = i / (rows - 1) if rows > 1 else 0  # 行比例
                
                # 双线性插值
                point = (1 - u) * (1 - v) * top_left + \
                        u * (1 - v) * top_right + \
                        (1 - u) * v * bottom_left + \
                        u * v * bottom_right
                        
                board_points.append(point)
        
        result = np.array(board_points, dtype=np.float32)
        print(f"生成棋盘格点数量: {len(result)}")
        return result
