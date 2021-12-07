"""
    A module for loading camera matrices from json files 
"""

from scipy.linalg import null_space
from pathlib import Path 
import json 
import numpy as np 

from util import pflat 

def build_camera_matrices(folder:Path, output_K=False):
    txt_path = folder / 'cameras.json'
    text = txt_path.read_text()
    cams_obj = json.loads(text)
    f = cams_obj['instrinsics']['f']
    Cx = cams_obj['instrinsics']['Cx']
    Cy = cams_obj['instrinsics']['Cy']

    cameras = dict()
    for cam in cams_obj['cams']:
        values = {'f':f, 'Cx':Cx, 'Cy':Cy}
        for key in ('x', 'y', 'z', 'pitch', 'roll', 'yaw'):
            values[key] = cam[key]
        
        cam_id = int(cam['id'])
        P, K = build_cam(values)
        cameras[cam_id] = P
    
    if output_K:
        return cameras, K

    return cameras

def build_cam(values):
    flip = np.array([[ 0, 1, 0 ], [ 0, 0, -1 ], [ 1, 0, 0 ]], dtype=np.float32)

    x = values['x']
    y = values['y']
    z = values['z']
    pitch = values['pitch']
    roll = values['roll']
    yaw = values['yaw']
    f = values['f']
    Cx = values['Cx']
    Cy = values['Cy']

    K = np.array([[f, 0, Cx], [0, f, Cy], [0, 0, 1]], dtype=np.float64)

    c_y = np.cos(np.radians(yaw))
    s_y = np.sin(np.radians(yaw))
    c_r = np.cos(np.radians(roll))
    s_r = np.sin(np.radians(roll))
    c_p = np.cos(np.radians(pitch))
    s_p = np.sin(np.radians(pitch))
    matrix = np.identity(4)
    matrix[0, 3] = x
    matrix[1, 3] = y
    matrix[2, 3] = z
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r
    matrix = np.linalg.inv(matrix)
    
    P = K @ flip @ matrix[:3, :]
    
    # Verify that camera's translation is correct 
    cen = np.array([x,y,z,1]).reshape((4,1))
    C = pflat(null_space(P))
    assert(np.allclose(C, cen))

    return P, K


def euler_angles(phi, theta, psi):
    sin = np.sin
    cos = np.cos
    R = [[cos(theta)*cos(psi),  -cos(phi)*sin(psi)+sin(phi)*sin(theta)*cos(psi),  sin(phi)*sin(psi)+cos(phi)*sin(theta)*cos(psi)],
         [cos(theta)*sin(psi), cos(phi)*cos(psi)+sin(phi)*sin(theta)*sin(psi), -sin(phi)*cos(psi)+cos(phi)*sin(theta)*sin(psi)],
         [-sin(theta),            sin(phi)*cos(theta),             cos(phi)*cos(theta)]]
    return np.array(R, dtype=np.float32)

def is_visible(x, y, z, P, im_h=720, im_w=1280):
    point = np.array([x, y, z, 1.0], dtype=np.float32).reshape((4,1))
    proj = pflat(P @ point).flatten()
    px, py = proj[0:2]
    if px >= 0 and px <= im_w and py >= 0 and py <= im_h:
        return True 
    
    return False 

def is_obj_visible(x, y, z, height, P, im_h=720, im_w=1280, min_height=10.0):
    point = np.array([x, y, z, 1.0], dtype=np.float32).reshape((4,1))
    proj = pflat(P @ point).flatten()
    px, py = proj[0:2]
    if px >= 0 and px <= im_w and py >= 0 and py <= im_h:
        delta_height = np.array([0,0,height,0], dtype=np.float32).reshape((4,1))
        new_point = point + delta_height
        proj2 = pflat(P @ new_point)
        py2 = proj2[1]

        pixel_height = abs(py2 - py)
        if pixel_height >= min_height:
            return True 
    
    return False 