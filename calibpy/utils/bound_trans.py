import numpy as np
import json
import os
from scipy.spatial.transform import Rotation as R


def psr_to_pose_matrix(psr):
    """
    将PSR格式(position, scale, rotation)转换为4x4的姿态矩阵
    
    Args:
        psr (dict): 包含position, scale, rotation的字典
        
    Returns:
        numpy.ndarray: 4x4的姿态矩阵
    """
    # 提取位置信息
    position = psr['position']
    translation = np.array([position['x'], position['y'], position['z']], dtype=np.float32)
    
    # 提取旋转信息 (欧拉角转旋转矩阵)
    rotation = psr['rotation']
    euler_angles = [rotation['x'], rotation['y'], rotation['z']]
    rotation_matrix = R.from_euler('xyz', euler_angles).as_matrix()
    
    # 构建4x4姿态矩阵
    pose_matrix = np.eye(4, dtype=np.float32)
    pose_matrix[:3, :3] = rotation_matrix
    pose_matrix[:3, 3] = translation
    
    return pose_matrix


def extract_dimensions_and_pose(json_file_path):
    """
    从JSON文件中提取物体的尺寸和姿态矩阵
    
    Args:
        json_file_path (str): JSON文件路径
        
    Returns:
        tuple: (dimensions, pose_matrices)
            - dimensions: [width, height, length]
            - pose_matrices: list of 4x4姿态矩阵
    """
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    dimensions = []
    pose_matrices = []
    
    # 处理每个物体
    for obj in data:
        psr = obj.get('psr', {})
        
        # 提取尺寸 (width, height, length)
        scale = psr.get('scale', {})
        width = scale.get('x', 0)
        height = scale.get('y', 0)
        length = scale.get('z', 0)
        dimensions.append([width, height, length])
        
        # 提取姿态矩阵
        pose_matrix = psr_to_pose_matrix(psr)
        pose_matrices.append(pose_matrix)
    
    return dimensions, pose_matrices


def convert_to_config_format(json_file_path):
    """
    将JSON文件中的数据转换为config文件所需的格式
    
    Args:
        json_file_path (str): JSON文件路径
        
    Returns:
        dict: 包含dimensions和pose_matrices的字典
    """
    dimensions, pose_matrices = extract_dimensions_and_pose(json_file_path)
    
    # 将姿态矩阵转换为一维数组格式，以便写入config文件
    flattened_poses = []
    for pose in pose_matrices:
        flattened_poses.append(pose.flatten().tolist())
    
    return {
        'dimensions': dimensions,
        'pose_matrices': flattened_poses
    }

def write_bounding_box(input_json_path, output_json_path):
    """
    将输入JSON文件中的数据转换为包含位姿矩阵的BoundingBox格式并写入输出文件
    
    Args:
        input_json_path (str): 输入JSON文件路径 (如 dif.json)
        output_json_path (str): 输出JSON文件路径 (如 0.json)
    """
    # 读取输入文件
    with open(input_json_path, 'r') as f:
        data = json.load(f)
    
    # 创建符合完整BoundingBox格式的数据（包括位姿矩阵）
    bounding_boxes = []
    
    for obj in data:
        psr = obj.get('psr', {})
        
        # 提取尺寸信息
        scale = psr.get('scale', {})
        width = scale.get('x', 0)
        height = scale.get('y', 0)
        depth = scale.get('z', 0)
        
        # 提取位置信息
        position = psr.get('position', {})
        
        # 计算位姿矩阵
        pose_matrix = psr_to_pose_matrix(psr)
        
        # 构造完整的BoundingBox格式数据
        bbox_data = {
            "width": width,
            "height": height,
            "depth": depth,
            "pos": pose_matrix.tolist()  # 添加4x4位姿矩阵
        }
        
        bounding_boxes.append(bbox_data)
    
    # 写入输出文件
    # 如果只有一个对象，就直接写入该对象，否则写入对象数组
    if len(bounding_boxes) == 1:
        output_data = bounding_boxes[0]
    else:
        output_data = bounding_boxes
    
    with open(output_json_path, 'w') as f:
        json.dump(output_data, f, indent=2)


# 示例用法
if __name__ == "__main__":
    # 示例：处理dif.json文件并将结果写入0.json
    input_file = "data/va/dif.json"
    output_file = "data/va/0.json"
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 写入包含位姿矩阵的BoundingBox格式数据
    write_bounding_box(input_file, output_file)
    
    print(f"已将 {input_file} 中的数据转换为包含位姿矩阵的BoundingBox格式并写入 {output_file}")
    
    # 显示结果
    with open(output_file, 'r') as f:
        result = json.load(f)
        print("\n输出文件内容:")
        print(json.dumps(result, indent=2))