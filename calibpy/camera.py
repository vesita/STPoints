from PIL import Image
import cv2
import numpy as np

try:
    from calibpy.fast_config import get_config
except:
    from fast_config import get_config
    
    
class Camera:
    def __init__(self):
        config = get_config()
        self.intrinsic = np.array(config.camera_intrinsic).reshape(3, 3)
        self.extrinsic = np.array(config.camera_extrinsic).reshape(4, 4)
        self.image = Image.open(config.image)
        self.chessboard_size = (6, 7)  # 棋盘格内角点数 (宽, 高)
    
    def get_caliboard(self):
        # 使用6*7（内角点数）进行标定
        # 使用opencv获取角点位置
        # 将PIL图像转换为OpenCV格式
        img = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2BGR)
        
        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 查找棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)
        
        if ret:
            # 精细化角点位置
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # 将角点坐标从图像坐标转换为(x, y)坐标形式
            points_2d = []
            for corner in corners:
                u, v = corner.ravel()
                points_2d.append([u, v])
                
            return np.array(points_2d, dtype=np.float32)
        else:
            # 如果找不到角点，返回空数组而不是None
            return np.array([])
    
    # 消除畸变后标定板在图像中的位置
    def ideal_target(self):
        points = self.get_caliboard()
        if len(points) == 0:
            return np.array([])
        
        # 使用内参矩阵的逆来消除畸变影响
        dedistortion = np.linalg.inv(self.intrinsic)
        # 将2D点转换为齐次坐标进行变换
        homogenous_points = np.hstack([points, np.ones((points.shape[0], 1))])
        return np.array([dedistortion @ p for p in homogenous_points])