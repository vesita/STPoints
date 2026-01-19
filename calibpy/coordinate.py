import numpy as np


def calculate_centroid(points):
    """
    计算点集的质心
    
    参数:
        points: 形状为(3, 3)的numpy数组，表示3个3D点
    
    返回:
        centroid: 3x1的质心向量
    """
    return np.mean(points, axis=0)


def center_points(points, centroid):
    """
    将点集中心化（减去质心）
    
    参数:
        points: 形状为(3, 3)的numpy数组，表示3个3D点
        centroid: 3x1的质心向量
    
    返回:
        centered_points: 中心化后的点集
    """
    return points - centroid


def calculate_rigid_transform(points_A, points_B, tolerance=1e-6):
    """
    使用SVD计算将points_A映射到points_B的刚性变换矩阵。
    
    参数:
        points_A: 形状为(3, 3)的numpy数组，表示3D空间中的3个点
        points_B: 形状为(3, 3)的numpy数组，表示3D空间中对应的3个点
        tolerance: 判断浮点误差的阈值
    
    返回:
        transformation_matrix: 4x4变换矩阵，将points_A映射到points_B
    """
    # 确保输入形状正确
    assert points_A.shape == (3, 3), "points_A必须具有形状(3, 3)"
    assert points_B.shape == (3, 3), "points_B必须具有形状(3, 3)"
    
    # 1. 计算质心
    centroid_A = calculate_centroid(points_A)
    centroid_B = calculate_centroid(points_B)
    
    # 2. 中心化点集
    centered_A = center_points(points_A, centroid_A)
    centered_B = center_points(points_B, centroid_B)
    
    # 3. 计算协方差矩阵
    H = np.dot(centered_A.T, centered_B)
    
    # 4. SVD分解计算旋转
    U, _, Vt = np.linalg.svd(H)
    
    # 计算旋转矩阵
    R = np.dot(Vt.T, U.T)
    
    # 处理反射情况（确保是纯旋转）
    if np.linalg.det(R) < 0:
        Vt_temp = Vt.copy()
        Vt_temp[-1, :] *= -1  # Vt[2] = -Vt[2]
        R = np.dot(Vt_temp.T, U.T)
    
    # 5. 计算平移向量
    t = centroid_B - np.dot(R, centroid_A)
    
    # 6. 构建变换矩阵
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = R
    transformation_matrix[:3, 3] = t
    
    return transformation_matrix


def apply_transform(points, transformation_matrix):
    """
    将变换矩阵应用于点集。
    
    参数:
        points: 形状为(N, 3)的numpy数组，表示3D空间中的N个点
        transformation_matrix: 4x4变换矩阵
    
    返回:
        transformed_points: 形状为(N, 3)的numpy数组，表示变换后的点
    """
    # 将点转换为齐次坐标
    points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
    
    # 应用变换
    transformed_points_homogeneous = np.dot(transformation_matrix, points_homogeneous.T).T
    
    # 转换回3D坐标
    transformed_points = transformed_points_homogeneous[:, :3]
    
    return transformed_points


def calculate_rotation_matrix_from_vectors(vec_a, vec_b, tolerance=1e-6):
    """
    使用SVD计算将向量a旋转到向量b的旋转矩阵。
    
    参数:
        vec_a: 形状为(3,)的numpy数组，源向量
        vec_b: 形状为(3,)的numpy数组，目标向量
        tolerance: 判断浮点误差的阈值
    
    返回:
        R: 3x3旋转矩阵
    """
    # 标准化向量
    vec_a = vec_a / np.linalg.norm(vec_a)
    vec_b = vec_b / np.linalg.norm(vec_b)
    
    # 计算叉积以找到旋转轴
    v = np.cross(vec_a, vec_b)
    s = np.linalg.norm(v)
    c = np.dot(vec_a, vec_b)
    
    # 处理特殊情况
    if s < tolerance:  # 平行向量
        if c > 1.0 - tolerance:  # 相同方向，考虑浮点误差
            return np.eye(3)
        elif c < -1.0 + tolerance:  # 相反方向，考虑浮点误差
            # 找到vec_a的最小分量以构造垂直向量
            idx = np.argmin(np.abs(vec_a))
            tmp = np.eye(3)
            u = np.cross(vec_a, tmp[idx])
            # 如果u是零向量，则尝试另一个方向
            if np.allclose(u, np.zeros(3), atol=tolerance):
                idx = (idx + 1) % 3
                u = np.cross(vec_a, tmp[idx])
            u = u / np.linalg.norm(u)
            return np.eye(3) - 2 * np.outer(u, u)
        else:
            # 在容差范围内平行但非完全相同或相反
            return np.eye(3)
    
    # 反对称叉积矩阵
    I = np.eye(3)
    vX = np.array([[0, -v[2], v[1]], 
                   [v[2], 0, -v[0]], 
                   [-v[1], v[0], 0]])
    
    # 计算旋转矩阵使用Rodrigues公式
    R = I + vX + np.dot(vX, vX) * ((1 - c) / (s * s))
    
    # 重新正交化旋转矩阵以修正数值误差
    U_r, _, Vt_r = np.linalg.svd(R)
    R = np.dot(U_r, Vt_r)
    
    return R


def calculate_transform_from_three_points(src_points, dst_points, tolerance=1e-6):
    """
    计算从3个源点到3个目标点的变换矩阵。
    
    参数:
        src_points: 形状为(3, 3)的numpy数组，3个源点[p1, p2, p3]
        dst_points: 形状为(3, 3)的numpy数组，3个目标点[p1', p2', p3']
        tolerance: 判断浮点误差的阈值
    
    返回:
        T: 4x4变换矩阵
    """
    # 验证输入
    if src_points.shape != (3, 3) or dst_points.shape != (3, 3):
        raise ValueError("src_points和dst_points必须具有形状(3, 3)")
    
    # 检查三点是否共线，如果是则不能形成唯一的变换
    def check_collinearity(points, tol=tolerance):
        # 计算由三点形成的两个向量
        v1 = points[1] - points[0]
        v2 = points[2] - points[0]
        # 计算叉积的模长，如果接近0则三点共线
        cross_product = np.linalg.norm(np.cross(v1, v2))
        return cross_product < tol
    
    if check_collinearity(src_points, tolerance) or check_collinearity(dst_points, tolerance):
        raise ValueError("三点不能共线，否则无法唯一确定变换矩阵")
    
    # 计算刚性变换
    T = calculate_rigid_transform(src_points, dst_points, tolerance)
    
    return T


def decompose_transformation_matrix(T):
    """
    将4x4变换矩阵分解为旋转矩阵和平移向量。
    
    参数:
        T: 4x4变换矩阵
    
    返回:
        R: 3x3旋转矩阵
        t: 3x1平移向量
    """
    R = T[:3, :3]
    t = T[:3, 3]
    
    # 验证R是有效的旋转矩阵（正交且行列式为1）
    is_orthogonal = np.allclose(np.dot(R, R.T), np.eye(3), atol=1e-6)
    det_is_positive = np.linalg.det(R) > 0
    
    if not is_orthogonal or not det_is_positive:
        # 尝试修复旋转矩阵使其成为正交矩阵
        U, _, Vt = np.linalg.svd(R)
        R_fixed = np.dot(U, Vt)
        if np.linalg.det(R_fixed) < 0:
            Vt[-1, :] *= -1
            R_fixed = np.dot(U, Vt)
        R = R_fixed
    
    return R, t


def validate_rotation_matrix(R, tolerance=1e-6):
    """
    验证一个矩阵是否为有效的旋转矩阵。
    
    参数:
        R: 待验证的3x3矩阵
        tolerance: 浮点误差容忍度
    
    返回:
        True: 如果R是有效的旋转矩阵
        False: 否则
    """
    if R.shape != (3, 3):
        return False
    
    # 检查是否正交
    is_orthogonal = np.allclose(np.dot(R, R.T), np.eye(3), atol=tolerance)
    
    # 检查行列式是否为1
    det_is_one = abs(np.linalg.det(R) - 1.0) < tolerance
    
    return is_orthogonal and det_is_one