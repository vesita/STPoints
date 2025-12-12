import numpy as np

def generate_scale(s):
    # 根据参数类型选择访问方式：如果是字典则使用键访问，否则使用属性访问
    if isinstance(s, dict):
        x = s['x'] / 2
        y = s['y'] / 2
        z = s['z'] / 2
    else:
        x = s.x / 2
        y = s.y / 2
        z = s.z / 2
    
    return [
        [x, y, z],
        [x, y, -z],
        [x, -y, z],
        [x, -y, -z],
        [-x, y, z],
        [-x, y, -z],
        [-x, -y, z],
        [-x, -y, -z]
    ]

def generate_rotation(r, points):
    # 根据参数类型选择访问方式
    if isinstance(r, dict):
        rx = np.array([
            [1, 0, 0],
            [0, np.cos(r['x']), -np.sin(r['x'])],
            [0, np.sin(r['x']), np.cos(r['x'])]
        ])
        ry = np.array([
            [np.cos(r['y']), 0, np.sin(r['y'])],
            [0, 1, 0],
            [-np.sin(r['y']), 0, np.cos(r['y'])]
        ])
        rz = np.array([
            [np.cos(r['z']), -np.sin(r['z']), 0],
            [np.sin(r['z']), np.cos(r['z']), 0],
            [0, 0, 1]
        ])
    else:
        rx = np.array([
            [1, 0, 0],
            [0, np.cos(r.x), -np.sin(r.x)],
            [0, np.sin(r.x), np.cos(r.x)]
        ])
        ry = np.array([
            [np.cos(r.y), 0, np.sin(r.y)],
            [0, 1, 0],
            [-np.sin(r.y), 0, np.cos(r.y)]
        ])
        rz = np.array([
            [np.cos(r.z), -np.sin(r.z), 0],
            [np.sin(r.z), np.cos(r.z), 0],
            [0, 0, 1]
        ])
    
    # 计算旋转矩阵
    rotation_matrix = rx @ ry @ rz
    
    # 对每个点应用旋转
    rotated_points = []
    for point in points:
        # 直接对3维点应用3x3旋转矩阵
        rotated_point = rotation_matrix @ np.array(point)
        rotated_points.append(rotated_point)
    
    return rotated_points

def generate_position(p, points):
    # 根据参数类型选择访问方式
    if isinstance(p, dict):
        return [(point + np.array([p['x'], p['y'], p['z']])) for point in points]
    else:
        return [(point + np.array([p.x, p.y, p.z])) for point in points]

def corners_of_psr(p, s, r):
    return generate_position(p, generate_rotation(r, generate_scale(s)))
    # return generate_rotation(r, generate_position(p, generate_scale(s)))

def edges():
    return [
        [0, 1], [1, 2], [2, 3], [3, 0],  # 底面
        [4, 5], [5, 6], [6, 7], [7, 4],  # 顶面
        [0, 4], [1, 5], [2, 6], [3, 7]   # 垂直边
    ]