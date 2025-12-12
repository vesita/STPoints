import json
import open3d as o3d
import numpy as np

try:
    from calibpy.fast_config import get_config
except:
    from fast_config import get_config


class BoundingBox:
    def __init__(self):
        self.width = 0
        self.height = 0
        self.depth = 0
        self.pos = np.eye(4, 4)  # 修复：正确初始化4x4单位矩阵

    def from_file(self, file_path):
        with open(file_path, 'r') as f:
            file = json.load(f)
        self.width = file["width"]
        self.height = file["height"]
        self.depth = file["depth"]
        self.pos = file["pos"]

    def inside(self, point):
        # 将点转换为包围盒局部坐标系
        # 创建齐次坐标
        point_homo = np.array([point[0], point[1], point[2], 1.0])
        
        # 获取包围盒的逆变换矩阵，将点转换到包围盒局部坐标系
        inv_transform = np.linalg.inv(self.pos)
        local_point = inv_transform @ point_homo
        
        # 在局部坐标系中检查点是否在包围盒内
        half_width = self.width / 2
        half_height = self.height / 2
        half_depth = self.depth / 2
        
        return (
            abs(local_point[0]) <= half_width and
            abs(local_point[1]) <= half_height and
            abs(local_point[2]) <= half_depth
        )


class Lidar:
    def __init__(self):
        config = get_config()
        self.extrinsic = np.array(config.get_lidar_extrinsic()).reshape(4, 4)
        self.cloud = o3d.io.read_point_cloud(config.cloud)
        bx = BoundingBox()
        bx.from_file(config.label)
        self.bounding_box = bx
        
    def target_cloud_self(self):
        return [point for point in self.cloud.points if self.bounding_box.inside(point)]
    
    def target_cloud(self):
        points = self.target_cloud_self()
        # 如果没有点在包围盒内，返回空列表而不是 None
        if len(points) == 0:
            return []
            
        trans2world = np.linalg.inv(self.extrinsic)
        # 将点转换为齐次坐标进行变换
        transformed_points = []
        for point in points:
            homogenous_point = np.append(np.array(point), 1)
            transformed_point = trans2world @ homogenous_point
            transformed_points.append(transformed_point[:3])  # 只取x,y,z坐标
        return transformed_points
    
    def target_corners(self):
        cloud = self.target_cloud()
        print(f"LiDAR目标点云数量: {len(cloud)}")
        
        box = self.cloud2box(cloud)
        print(f"包围盒信息: {box}")
        
        corner8 = self.box_corners(box)
        print(f"包围盒8个角点: {corner8}")
        
        corner_board = self.corner_caliboard(corner8)
        print(f"棋盘格角点数量: {len(corner_board)}")
        
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
        
        # 确定哪个表面朝向相机（Z坐标最大的表面）
        # 但在LiDAR坐标系中，我们不知道哪个表面朝向相机
        # 需要根据包围盒的几何特征选择合适的表面
        
        # 选择面积最大的表面作为棋盘格所在表面
        # 计算三个表面的面积
        area_x = (y_max - y_min) * (z_max - z_min)  # YZ平面
        area_y = (x_max - x_min) * (z_max - z_min)  # XZ平面
        area_z = (x_max - x_min) * (y_max - y_min)  # XY平面
        
        print(f"表面面积 - X方向: {area_x:.3f}, Y方向: {area_y:.3f}, Z方向: {area_z:.3f}")
        
        # 选择面积最大的表面
        max_area = max(area_x, area_y, area_z)
        if max_area == area_x:
            # YZ平面，固定X坐标
            fixed_val = x_max if np.sum(corners[:, 0] > (x_min + x_max) / 2) > 2 else x_min
            surface_points = corners[np.abs(corners[:, 0] - fixed_val) < 1e-3]
            plane_type = "YZ"
            fixed_coord = fixed_val
        elif max_area == area_y:
            # XZ平面，固定Y坐标
            fixed_val = y_max if np.sum(corners[:, 1] > (y_min + y_max) / 2) > 2 else y_min
            surface_points = corners[np.abs(corners[:, 1] - fixed_val) < 1e-3]
            plane_type = "XZ"
            fixed_coord = fixed_val
        else:
            # XY平面，固定Z坐标
            fixed_val = z_max if np.sum(corners[:, 2] > (z_min + z_max) / 2) > 2 else z_min
            surface_points = corners[np.abs(corners[:, 2] - fixed_val) < 1e-3]
            plane_type = "XY"
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
        # 按照到质心的距离排序，找到最外围的四个点
        centroid = np.mean(surface_points, axis=0)
        distances = np.linalg.norm(surface_points - centroid, axis=1)
        
        # 找到距离质心最远的点作为参考
        farthest_idx = np.argmax(distances)
        farthest_point = surface_points[farthest_idx]
        
        # 找到与最远点距离最远的点
        distances_to_farthest = np.linalg.norm(surface_points - farthest_point, axis=1)
        opposite_idx = np.argmax(distances_to_farthest)
        opposite_point = surface_points[opposite_idx]
        
        # 在剩下的点中找到另外两个角点
        remaining_indices = [i for i in range(len(surface_points)) if i not in [farthest_idx, opposite_idx]]
        if len(remaining_indices) >= 2:
            # 计算剩余点到已选两点连线的距离
            remaining_points = surface_points[remaining_indices]
            
            # 计算点到线段的距离
            def point_to_line_distance(p, a, b):
                # 点p到线段ab的距离
                ap = p - a
                ab = b - a
                ab_sq = np.dot(ab, ab)
                if ab_sq == 0:
                    return np.linalg.norm(ap)
                t = max(0, min(1, np.dot(ap, ab) / ab_sq))
                projection = a + t * ab
                return np.linalg.norm(p - projection)
            
            distances1 = [point_to_line_distance(p, farthest_point, opposite_point) for p in remaining_points]
            max_dist_idx1 = np.argmax(distances1)
            third_point = remaining_points[max_dist_idx1]
            
            # 移除第三个点，找第四个点
            remaining_indices2 = [i for i in remaining_indices if i != remaining_indices[max_dist_idx1]]
            if len(remaining_indices2) >= 1:
                remaining_points2 = surface_points[remaining_indices2]
                distances2 = [point_to_line_distance(p, farthest_point, opposite_point) for p in remaining_points2]
                max_dist_idx2 = np.argmax(distances2)
                fourth_point = remaining_points2[max_dist_idx2]
            else:
                # 如果找不到第四个点，使用质心附近的一个点
                closest_to_centroid_idx = np.argmin(distances)
                fourth_point = surface_points[closest_to_centroid_idx]
        else:
            # 如果点不够，使用简单方法
            sorted_indices = np.argsort(distances_to_farthest)
            third_point = surface_points[sorted_indices[-2]] if len(sorted_indices) >= 2 else surface_points[0]
            fourth_point = surface_points[sorted_indices[-3]] if len(sorted_indices) >= 3 else surface_points[1]
        
        # 确定四个角点的相对位置
        # 按照某种规则排序四个点
        four_corners = [farthest_point, opposite_point, third_point, fourth_point]
        
        # 按照x+y的和排序，找到左上角和右下角
        sum_coords = [p[0] + p[1] for p in four_corners]
        diff_coords = [p[0] - p[1] for p in four_corners]
        
        top_left_idx = np.argmin(sum_coords)
        bottom_right_idx = np.argmax(sum_coords)
        bottom_left_idx = np.argmin(diff_coords)
        top_right_idx = np.argmax(diff_coords)
        
        top_left = four_corners[top_left_idx]
        top_right = four_corners[top_right_idx]
        bottom_left = four_corners[bottom_left_idx]
        bottom_right = four_corners[bottom_right_idx]
        
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