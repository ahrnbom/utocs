"""
    This script visualizes user generated tracks alongside the ground truth
    in both pixel and world coordinates (top-down) as videos
"""

from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import imageio as iio 
import numpy as np 
import argparse
import numpy as np
import json

from util import good_lstrip, pflat, intr, long_str
from cameras import build_camera_matrices
from visualize_samples import build3Dbox, draw3Dbox, read_positions
from eval import get_seqnums

def brighter(color):
    r = (color[0] + 255) // 2
    g = (color[1] + 255) // 2
    b = (color[2] + 255) // 2
    return (r, g, b)

def get_colors(categories:List[str]):
    colors = dict()
    n = len(categories)
    for i, cat in enumerate(categories):
        hue = 255*i/(n+2)
        col = np.empty((1,1,3)).astype("uint8")
        col[0][0][0] = hue
        col[0][0][1] = 128 + (i%9)*9 # Saturation
        col[0][0][2] = 150 + (i%2)*50 # Value (brightness)
        cvcol = cv2.cvtColor(col, cv2.COLOR_HSV2BGR)
        col = (int(cvcol[0][0][0]), int(cvcol[0][0][1]), int(cvcol[0][0][2]))
        colors[cat] = col 
    return colors 

def visualize_attempt(folder:Path, gt_folder:Path, classes:List[str], 
                      seq_num:int, out_path:Path, cam_num=0):
    
    colors = get_colors(classes)

    im_folder = gt_folder / '..' / 'images' / f"cam{cam_num}"
    images = list(im_folder.glob('*.jpg'))
    images.sort()
    n_ims = len(images)

    cameras = build_camera_matrices(gt_folder / '..')
    cam = cameras[cam_num]

    with iio.get_writer(out_path, fps=20) as vid:
        for im_path in images:
            frame_no = int(im_path.stem)
            image = iio.imread(im_path)
            attempt = read_positions(folder / f"{long_str(frame_no, 6)}.json")
            gt = read_positions(gt_folder / f"{long_str(frame_no, 6)}.json")

            frame1 = render_pixel_frame(image, classes, frame_no, attempt, gt, 
                                        cam, colors)
            frame2 = render_topdown_frame(frame1.shape, classes, attempt, gt, 
                                          colors)
            
            frame = np.vstack([frame1, frame2])
            vid.append_data(frame)

            if frame_no%100 == 0:
                print(f"{frame_no+1} {100.0*frame_no/n_ims:.2f}%")

def render_topdown_frame(dims:Tuple, classes:List[str], attempt:List[Dict],
                         ground_truth:List[Dict], colors:Dict):
    
    up = np.array([0,0,1,0], dtype=np.float32)

    for at in attempt:
        class_name = at['type']
        if not class_name in classes:
            continue 

        X, l, w, h = [at[key] for key in "Xlwh"]
        forward = np.array([at['forward_x'], at['forward_y'], 
                           at['forward_z'], 0],
                           dtype=np.float32)
        right = np.array([*np.cross(forward[0:3], up[0:3]), 0.0], 
                         dtype=np.float32)

        color = colors[class_name]
        points3D = build3Dbox(X, l, w, h, forward, right, up)
        

def render_pixel_frame(image:np.ndarray, classes:List[str], frame_no:int, 
                       attempt:List[Dict], ground_truth:List[Dict], 
                       cam:np.ndarray, colors:Dict):

    up = np.array([0,0,1,0], dtype=np.float32)

    for at in attempt:
        class_name = at['type']
        if not class_name in classes:
            continue 

        X, l, w, h = [at[key] for key in "Xlwh"]
        forward = np.array([at['forward_x'], at['forward_y'], 
                           at['forward_z'], 0],
                           dtype=np.float32)
        right = np.array([*np.cross(forward[0:3], up[0:3]), 0.0], 
                         dtype=np.float32)

        color = colors[class_name]
        draw3Dbox(image, cam, X, l, w, h, forward, right, up, color)

    for gt in ground_truth:
        class_name = gt['type']
        if not class_name in classes:
            continue 

        X, l, w, h = [gt[key] for key in "Xlwh"]
        forward = np.array([gt['forward_x'], gt['forward_y'], 
                           gt['forward_z'], 0],
                           dtype=np.float32)
        right = np.array([*np.cross(forward[0:3], up[0:3]), 0.0], 
                         dtype=np.float32)

        color = colors[class_name]
        color = brighter(color)
        draw3Dbox(image, cam, X, l, w, h, forward, right, up, color)
    
    # Thicker black first, then thin white, very readable 
    cv2.putText(image, f"Frame {frame_no}", (10,20), cv2.FONT_HERSHEY_PLAIN, 
                1.5, (0,0,0), 2, cv2.LINE_AA)
    cv2.putText(image, f"Frame {frame_no}", (10,20), cv2.FONT_HERSHEY_PLAIN, 
                1.5, (255,255,255), 1, cv2.LINE_AA)

    return image 

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--folder", type=str,
                      help="Path to folder with the tracks to be evaluated")
    args.add_argument("--gt_folder",  default="./output", type=str,
                      help="Path of ground truth",)
    args.add_argument("--set", type=str, default='test',
                      help="Either 'training', 'validation' or 'test")
    args.add_argument("--classes", type=str, default="",
                      help="Set to a comma-separated list of classes to only " \
                           "visualize those")
    args.add_argument("--export", action="store_true", help="Include to write "\
                      "the video in the folder with the track. Otherwise, "\
                      "it will be written in the folder 'output'")
    args = args.parse_args()

    folder = Path(args.folder)
    gt_folder = Path(args.gt_folder)
    which_set = args.set

    if args.classes:
        classes = args.classes.split(',')
    else:
        classes = ['car', 'truck', 'bus', 'bicyclist', 'pedestrian']

    assert which_set in ['training', 'validation', 'test']
    seq_nums = get_seqnums()
    
    for seq_num in seq_nums[which_set]:
        seq = gt_folder / 'scenarios' / long_str(seq_num, 4)
        _folder = folder / seq.name
        _gtfolder = seq / 'positions'
        print(f"Visualizing {seq_num} from {which_set} for run {folder}")

        if args.export:
            out_path = folder / f"utocs_visualization_{seq_num}.mp4"
        else:
            out_folder = Path('output') / 'visualized_attempts'
            out_folder.mkdir(exist_ok=True, parents=True)
            out_path = out_folder / "utocs_visualization_" \
                                    f"{folder.name}_{seq_num}.mp4"

        visualize_attempt(_folder, _gtfolder, classes, seq_num, out_path)