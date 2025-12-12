import json
import numpy as np
import cv2
import os

# load calib
calib_path = 'data/calib/calib/camera/image.json'
label_path = 'data/calib/label/000002.json'

with open(calib_path, 'r') as f:
    calib = json.load(f)

intrinsic = np.array(calib['intrinsic']).reshape(3,3).astype(np.float64)
# dist may be long; take first 8 or full
dist = np.array(calib.get('distortion_coefficients', calib.get('distortion', [0,0,0,0])) , dtype=np.float64)

extrinsic = np.array(calib['extrinsic']).reshape(4,4).astype(np.float64)

with open(label_path,'r') as f:
    labels = json.load(f)

psr = labels[0]['psr']
pos = psr['position']
rot = psr['rotation']
scale = psr['scale']

# Implement front-end psr_to_xyz local ordering (from util.psr_to_xyz)
# Front-end uses euler_angle_to_rotate_matrix(eu,tr) which composes ZYX order
# We'll implement that to get transform matrix

def euler_angle_to_rotate_matrix(eu, tr, order='ZYX'):
    theta = [eu['x'], eu['y'], eu['z']]
    cx = np.cos(theta[0]); sx = np.sin(theta[0])
    cy = np.cos(theta[1]); sy = np.sin(theta[1])
    cz = np.cos(theta[2]); sz = np.sin(theta[2])
    R_x = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
    R_y = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    R_z = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
    mats = {'X':R_x,'Y':R_y,'Z':R_z}
    R = mats[order[2]].dot(mats[order[1]].dot(mats[order[0]]))
    M = np.eye(4)
    M[:3,:3]=R
    M[0,3]=tr['x']; M[1,3]=tr['y']; M[2,3]=tr['z']
    return M

# construct box local coords as in util.psr_to_xyz
x = scale['x']/2.0
y = scale['y']/2.0
z = scale['z']/2.0
local = [
    x, y, -z, 1,   x, -y, -z, 1,
    x, -y, z, 1,   x, y, z, 1,
    -x, y, -z, 1,   -x, -y, -z, 1,
    -x, -y, z, 1,   -x, y, z, 1
]
local = np.array(local).reshape(-1,4)

M = euler_angle_to_rotate_matrix(rot, pos)
world = (M.dot(local.T)).T[:,0:3]

# Method A: OpenCV projectPoints using extrinsic R,t from extrinsic matrix
R_cv = extrinsic[:3,:3]
t_cv = extrinsic[:3,3].reshape(3,1)
# cv2.projectPoints expects rvec,tvec where rvec is Rodrigues for camera rotation from object
# Here object points are in LiDAR/world and we want to map to camera: X_cam = R_cv * X + t_cv
# We can use rvec = Rodrigues(R_cv)
rvec, _ = cv2.Rodrigues(R_cv)

proj_cv, _ = cv2.projectPoints(world.reshape(-1,1,3), rvec, t_cv, intrinsic, dist)
proj_cv = proj_cv.reshape(-1,2)

# Method B: frontend style (no distortion)
# compute homogeneous
hom = np.hstack([world, np.ones((world.shape[0],1))])
imgpos = (extrinsic.dot(hom.T)).T[:,0:3]
# apply intrinsic (3x3) to 3D points
imgpos2 = (intrinsic.dot(imgpos.T)).T
uv = imgpos2[:,0:2] / imgpos2[:,2:3]

print('cv2 projected points:\n', proj_cv)
print('frontend projected points (no-distort):\n', uv)

# Compare differences
print('\nDifferences (frontend - cv):')
print(uv - proj_cv)

# Print centroid positions
print('\nworld centroid:', np.mean(world,axis=0))
print('camera translation t_cv:', t_cv.ravel())

# Save to file for inspection
np.savetxt('tools/proj_cv.txt', proj_cv, fmt='%.4f')
np.savetxt('tools/proj_frontend.txt', uv, fmt='%.4f')

print('\nSaved proj results to tools/proj_cv.txt and tools/proj_frontend.txt')
