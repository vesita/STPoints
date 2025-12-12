

import numpy as np


def ro_x(theta):
    """绕X轴旋转矩阵"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])
    

def ro_y(theta):
    """绕Y轴旋转矩阵"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])
    

def ro_z(theta):
    """绕Z轴旋转矩阵"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])
    

def rotation_matrix(rx, ry, rz):
    """根据欧拉角（弧度）计算旋转矩阵，ZYX顺序"""
    Rz = ro_z(rz)
    Ry = ro_y(ry)
    Rx = ro_x(rx)
    return Rx @ Ry @ Rz