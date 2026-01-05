import numpy as np

def compute_transformation(A1, B1, A2, B2):
    # 将点转换为numpy数组
    A1 = np.array(A1)
    B1 = np.array(B1)
    A2 = np.array(A2)
    B2 = np.array(B2)

    # 计算方向向量
    d1 = B1 - A1
    d2 = B2 - A2

    # 计算平移分量
    T = A2 - A1

    # 标准化方向向量
    norm_d1 = np.linalg.norm(d1)
    norm_d2 = np.linalg.norm(d2)
    if norm_d1 == 0 or norm_d2 == 0:
        raise ValueError("方向向量必须为非零向量")
    
    u = d1 / norm_d1
    v = d2 / norm_d2

    # 计算旋转轴和角度
    rotation_axis = np.cross(u, v)
    angle = np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))

    # 如果旋转轴为零向量，则u和v平行
    if np.linalg.norm(rotation_axis) == 0:
        R = np.eye(3)  # 无需旋转
    else:
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        K = np.array([[0, -rotation_axis[2], rotation_axis[1]],
                      [rotation_axis[2], 0, -rotation_axis[0]],
                      [-rotation_axis[1], rotation_axis[0], 0]])
        R = (np.eye(3) + 
             np.sin(angle) * K + 
             (1 - np.cos(angle)) * np.dot(K, K))

    return R, T


if __name__ == "__main__":
    A1 = (1, 2, 3)
    B1 = (4, 5, 6)
    A2 = (7, 8, 9)
    B2 = (10, 11, 12)

    R, T = compute_transformation(A1, B1, A2, B2)

    print("旋转矩阵 R:")
    print(R)
    print("\n平移向量 T:")
    print(T)