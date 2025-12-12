import json
import os

class Config:
    _instance = None
    
    def __new__(cls, config_file="config/config.json"):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config_file="config/config.json"):
        if self._initialized:
            return
        with open(config_file) as f:
            config_data = json.load(f)
            self.camera_extrinsic = config_data["camera_extrinsic"]
            self.lidar_extrinsic = config_data["lidar_extrinsic"]
            self.camera_intrinsic = config_data["camera_intrinsic"]
            self.target_path = config_data["target_path"]
            
            # 先初始化基本配置，然后再处理Target
            target = Target(self.target_path)
            self.cloud = target.cloud
            self.image = target.image
            self.label = target.label
        self._initialized = True
    
    def get_lidar_extrinsic(self):
        return self.lidar_extrinsic

def get_config():
    """获取全局单例配置实例"""
    return Config()


class Target:
    """
    最简配置模式，只处理单个文件
    """
    def __init__(self, target_path=None, index=0):
        # 如果没有提供target_path，则从配置中获取
        if target_path is None:
            config = get_config()
            target_path = config.target_path
            
        # 只使用一个固定的文件组合，而不是读取整个目录
        self.image = os.path.join(target_path, f"{index}.jpg")
        self.cloud = os.path.join(target_path, f"{index}.pcd")
        self.label = os.path.join(target_path, f"{index}.json")
        
        # 确保文件存在
        if not os.path.exists(self.image):
            self.image = None
        
        if not os.path.exists(self.cloud):
            self.cloud = None
        if not os.path.exists(self.label):
            self.label = None