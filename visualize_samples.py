""" Optional script for visualizing the ground points and ground truth positions 
    for a random sample of generated images. This also works as a demo for how 
    to interpret camera data into a projection matrix.

    Install dependencies by 
    python -m pip install imageio opencv-python numpy
"""

from pathlib import Path
import cv2
import imageio as iio 
import numpy as np 
import argparse
from random import choice 

import numpy as np
from scipy.linalg import null_space

def set_from_string(some_dict, text):
    name, value = text.split('=')
    some_dict[name] = float(value)
    return some_dict

def build_camera_matrices(folder:Path, output_K=False):
    txt_path = folder / 'camera_calibration.txt'
    text = txt_path.read_text()
    
    lines = [x for x in text.split('\n') if x]
    intrinsics_line = lines[-1]
    cam_lines = lines[:-1]

    intrinsics_splot = good_lstrip(intrinsics_line, 'Intrinsics: ').split(', ')
    base_values = dict()
    for i in range(3):
        base_values = set_from_string(base_values, intrinsics_splot[i])
    assert('f'  in base_values)
    assert('Cx' in base_values)
    assert('Cy' in base_values)

    cameras = dict()
    for line in cam_lines:
        values = dict()
        values.update(base_values)

        cam_id = line.split(':')[0]
        new_line = good_lstrip(line, f"{cam_id}:")
        splot = new_line.split(';')
        loc_str = good_lstrip(splot[0], 'Location(').rstrip(')')
        loc_splot = loc_str.split(', ')
        for i in range(3):
            values = set_from_string(values, loc_splot[i])
        
        rot_str = good_lstrip(splot[1], 'Rotation(').rstrip(')')
        rot_splot = rot_str.split(', ')
        for i in range(3):
            values = set_from_string(values, rot_splot[i])
        
        for name in ['x', 'y', 'z', 'pitch', 'roll', 'yaw']:
            assert(name in values)
        
        P, K = build_cam(values)
        cameras[cam_id] = P
    
    if output_K:
        return cameras, K

    return cameras

def build_cam(values):
    flip = np.array([[ 0,  1,  0 ], [ 0,  0, -1 ], [ 1,  0,  0 ]], dtype=np.float32)

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


def good_lstrip(text, intro):
    assert(len(intro) <= len(text))
    l = len(intro)
    first = text[:l]
    assert(first == intro)

    return text[l:]

def intr(x):
    return int(round(float(x)))

def pflat(x):
    if len(x.shape) == 1:
        x /= x[-1]
    else:
        x /= x[-1, :]
    return x

def euler_angles(phi, theta, psi):
    sin = np.sin
    cos = np.cos
    R = [[cos(theta)*cos(psi),  -cos(phi)*sin(psi)+sin(phi)*sin(theta)*cos(psi),  sin(phi)*sin(psi)+cos(phi)*sin(theta)*cos(psi)],
         [cos(theta)*sin(psi), cos(phi)*cos(psi)+sin(phi)*sin(theta)*sin(psi), -sin(phi)*cos(psi)+cos(phi)*sin(theta)*sin(psi)],
         [-sin(theta),            sin(phi)*cos(theta),             cos(phi)*cos(theta)]]
    return np.array(R, dtype=np.float32)

def read_positions(pos_path:Path):
    text = pos_path.read_text()
    lines = [x for x in text.split('\n') if x]

    instances = list()

    for line in lines:
        splot = line.split(';')
        ru_type = splot[0]
        track_id = int(splot[1])
        loc_str = good_lstrip(splot[2], 'Location(').rstrip(')')

        xs, ys, zs = loc_str.split(', ')
        x = float(good_lstrip(xs, 'x='))
        y = float(good_lstrip(ys, 'y='))
        z = float(good_lstrip(zs, 'z='))
        X = np.array([x, y, z, 1], dtype=np.float32).reshape((4,1))

        instance = {'X': X, 'type': ru_type, 'id': track_id}
        instances.append(instance)
    
    return instances

def main(folder:Path, cam_index:int):
    all_scenarios = [f for f in (folder / 'scenarios').glob('*') if f.is_dir()]
    for scenario in all_scenarios:
        visualize_scenario(scenario, folder / 'visualization', cam_index)

def visualize_scenario(scenario:Path, out_folder:Path, cam_index:int):
    all_positions = [f for f in (scenario / 'positions').glob('*.txt') if f.is_file()]
    all_positions.sort(key=lambda f: int(f.stem))
    
    n_frames = len(all_positions)

    out_folder.mkdir(exist_ok=True)
    with iio.get_writer(out_folder / f"{scenario.name}.mp4", fps=25) as vid:
        for pos_path in all_positions:
            frame_no = int(pos_path.stem)
            instances = read_positions(pos_path)

            all_cams = [f for f in (scenario / 'images').glob('cam*') if f.is_dir()]
            all_cams.sort(key=lambda f: int(good_lstrip(f.name, 'cam')))
            cam = all_cams[cam_index]

            projection_matrices = build_camera_matrices(scenario)
            cam_num = good_lstrip(cam.name, 'cam')
            proj_matrix = projection_matrices[cam_num]

            ground_points = np.genfromtxt(scenario / 'ground_points.txt', delimiter=',', dtype=np.float32).T
            n = ground_points.shape[1]
            new_ground = np.ones((4, n), dtype=np.float32)
            new_ground[0:3, :] = ground_points

            im = iio.imread(cam / f"{pos_path.stem}.jpg")

            proj_ground = proj_matrix @ new_ground
            for i_ground in range(n):
                ground_point = pflat(proj_ground[:, i_ground])
                x = intr(ground_point[0])
                y = intr(ground_point[1])
                cv2.drawMarker(im, (x, y), (255,0,128), cv2.MARKER_CROSS, 4, 2, cv2.LINE_AA)
            
            for instance in instances:
                X = instance['X']
                ru_type = instance['type']
                ru_id = instance['id']

                pos2D = pflat(proj_matrix @ X)
                x, y, _ = pos2D
                x = intr(x)
                y = intr(y)

                if x >= 0 and x <= 1280 and y >= 0 and y <= 720:
                    cv2.drawMarker(im, (x,y), (0,255,255), cv2.MARKER_TRIANGLE_UP, 8, 2, cv2.LINE_AA)
                    cv2.putText(im, f"{ru_type}{ru_id}", (x,y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,0,0), 2, cv2.LINE_AA)
                    cv2.putText(im, f"{ru_type}{ru_id}", (x,y), cv2.FONT_HERSHEY_PLAIN, 1.5, (255,255,255), 1, cv2.LINE_AA)

            cv2.putText(im, f"Frame {frame_no}", (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(im, f"Frame {frame_no}", (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (255,255,255), 1, cv2.LINE_AA)
            
            vid.append_data(im)
            if frame_no%20 == 0:
                print(f"{frame_no} / {n_frames} ({100*frame_no/n_frames:.1f}%) scenario: {scenario.name}")

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--folder", help="Where the data is located", default="./output")
    args.add_argument("--cam", help="Which camera (default: 0)", type=int, default=0)
    args = args.parse_args()
    folder = Path(args.folder)
    cam_index = args.cam
    main(folder, cam_index)