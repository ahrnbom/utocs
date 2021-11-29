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
import numpy as np
import json

from util import good_lstrip, pflat, intr
from cameras import build_camera_matrices

def read_positions(pos_path:Path):
    text = pos_path.read_text()
    instances = json.loads(text)
    for instance in instances:
        x = instance['x']
        y = instance['y']
        z = instance['z']
        X = np.array([x, y, z, 1], dtype=np.float32)
        instance['X'] = X
        # X is used to make it easy to project into cameras
    return instances


def main(folder:Path, cam_index:int):
    all_scenarios = [f for f in (folder / 'scenarios').glob('*') if f.is_dir()]
    all_scenarios.sort()
    for scenario in all_scenarios:
        visualize_scenario(scenario, folder / 'visualization', cam_index)


def visualize_scenario(scenario:Path, out_folder:Path, cam_index:int):
    all_positions = [f for f in (scenario / 'positions').glob('*.json') if f.is_file()]
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
            cam_num = int(good_lstrip(cam.name, 'cam'))
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
                cv2.drawMarker(im, (x, y), (255,0,128), cv2.MARKER_STAR, 4, 2, cv2.LINE_AA)
                
                # To number the ground points:
                #cv2.putText(im, str(i_ground), (x,y+10), cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,255), 1, cv2.LINE_AA)
            
            up = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)

            for instance in instances:
                X = instance['X']
                ru_type = instance['type']
                ru_id = instance['id']

                l, w, h = [instance[key] for key in "lwh"]
                fx, fy, fz = [instance[key] for key in ['forward_x', 'forward_y', 'forward_z']]
                forward = np.array([fx, fy, fz, 0.0], dtype=np.float32)
                
                right = np.cross(forward[0:3], up[0:3])
                right = np.array([*right, 0.0], dtype=np.float32)

                pos2D = pflat(proj_matrix @ X)
                x, y, _ = pos2D
                x = intr(x)
                y = intr(y)

                if x >= 0 and x <= 1280 and y >= 0 and y <= 720:
                    cv2.drawMarker(im, (x,y), (0,255,255), cv2.MARKER_CROSS, 8, 2, cv2.LINE_AA)
                    cv2.putText(im, f"{ru_type}{ru_id}", (x,y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,0,0), 2, cv2.LINE_AA)
                    cv2.putText(im, f"{ru_type}{ru_id}", (x,y), cv2.FONT_HERSHEY_PLAIN, 1.5, (255,255,255), 1, cv2.LINE_AA)

                    draw3Dbox(im, proj_matrix, X, l, w, h, forward, right, up)

            cv2.putText(im, f"Frame {frame_no}", (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(im, f"Frame {frame_no}", (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (255,255,255), 1, cv2.LINE_AA)
            
            vid.append_data(im)
            if frame_no%20 == 0:
                print(f"{frame_no} / {n_frames} ({100*frame_no/n_frames:.1f}%) scenario: {scenario.name}")

def draw3Dbox(im, P, X, l, w, h, forward, right, up):
    points3D = list()
    for il in (-0.5, 0.5):
        dl = il * l * forward 
        for iw in (-0.5, 0.5):
            dw = iw * w * right 
            for ih in (0.0, 1.0):
                dh = ih * h * up 
                point3D = X + dl + dw + dh 
                points3D.append(point3D)

    for indices in [(0,1,3,2), (4,5,7,6), (0,4,5,1), (2,6,7,3), (0,4,6,2), (1,5,7,3)]:
        p1 = points3D[indices[0]]
        p2 = points3D[indices[1]]
        p3 = points3D[indices[2]]
        p4 = points3D[indices[3]]

        for pair in [(p1, p2), (p2, p3), (p3, p4), (p4, p1)]:
            a, b = pair 
            
            a2D = pflat(P @ a)
            ax, ay, _ = a2D
            ax = intr(ax)
            ay = intr(ay)

            b2D = pflat(P @ b)
            bx, by, _ = b2D
            bx = intr(bx)
            by = intr(by)
            cv2.line(im, (ax, ay), (bx, by), (100,255,100), 1, cv2.LINE_AA)

    # Show forward direction
    Xf = X + l/2.0 * forward 
    Xf2D = pflat(P @ Xf)
    xf, yf, _ = Xf2D
    xf = intr(xf)
    yf = intr(yf)
    cv2.drawMarker(im, (xf, yf), (255,0,0), cv2.MARKER_TRIANGLE_UP, 8, 2, cv2.LINE_AA)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--folder", help="Where the data is located", default="./output")
    args.add_argument("--cam", help="Which camera (default: 0)", type=int, default=0)
    args = args.parse_args()
    folder = Path(args.folder)
    cam_index = args.cam
    main(folder, cam_index)