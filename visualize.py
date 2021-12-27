"""
    This script visualizes user generated tracks alongside the ground truth
    in both pixel and world coordinates (top-down) as videos. Can also be used
    to only visualize the ground truth 
"""

from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import imageio as iio 
import numpy as np 
import argparse
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
from scipy.linalg import null_space 

from util import normalize_numpy_vector, pflat, intr, long_str
from cameras import build_camera_matrices
from eval import get_seqnums

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

# Splits an int in two approximately equal ints that sum up to x 
def int_split2(x):
    if x%2 == 0:
        return x//2, x//2
    else:
        return x//2, (x//2)+1

def good_resize(im, new_h, new_w):
    h, w, _ = im.shape 

    h_scale = new_h/h 
    w_scale = new_w/w

    # Scale by whichever dimension needs the least change, and then pad 
    scale = min(h_scale, w_scale)
    w2, h2 = intr(w*scale), intr(h*scale)
    im = cv2.resize(im, (w2, h2))

    dh = int_split2(new_h - h2)
    dw = int_split2(new_w - w2)

    new_im = np.pad(im, (dh, dw, (0,0)))
    return new_im 

def brighter(color):
    r = (color[0] + 255) // 2
    g = (color[1] + 255) // 2
    b = (color[2] + 255) // 2
    return (r, g, b)

def darker(color):
    r = 2*color[0]//3
    g = 2*color[1]//3
    b = 2*color[2]//3
    return (r, g, b)

def matplotlib_color(color, alpha=1.0):
    new_color = [c/256.0 for c in color]
    new_color.append(alpha)
    return new_color

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

def build3Dbox(X, l, w, h, forward, right, up):
    points3D = list()
    for il in (-0.5, 0.5):
        dl = il * l * forward 
        for iw in (-0.5, 0.5):
            dw = iw * w * right 
            for ih in (0.0, 1.0):
                dh = ih * h * up 
                point3D = X + dl + dw + dh 
                points3D.append(point3D)
    return points3D


def draw3Dbox(im, P, X, l, w, h, forward, right, up, color=(255,0,0), text=""):
    points3D = build3Dbox(X, l, w, h, forward, right, up)

    for indices in [(0,1,3,2), (4,5,7,6), (0,4,5,1), (2,6,7,3), (0,4,6,2), 
                    (1,5,7,3)]:

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
            cv2.line(im, (ax, ay), (bx, by), color, 1, cv2.LINE_AA)

    # Show forward direction
    Xf = X + l/2.0 * forward 
    Xf2D = pflat(P @ Xf)
    xf, yf, _ = Xf2D
    xf = intr(xf)
    yf = intr(yf)
    point = (xf, yf)
    cv2.drawMarker(im, point, darker(color), cv2.MARKER_TRIANGLE_UP, 8, 2, 
                   cv2.LINE_AA)

    if text:
        cv2.putText(im, text, point, cv2.FONT_HERSHEY_PLAIN, 1.0, darker(color),
                    2, cv2.LINE_AA)
        cv2.putText(im, text, point, cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,255),
                    1, cv2.LINE_AA)

def visualize(folder:Path, gt_folder:Path, classes:List[str], 
              seq_num:int, out_path:Path, cam_num=0):
    
    colors = get_colors(classes)

    im_folder = gt_folder / '..' / 'images' / f"cam{cam_num}"
    images = list(im_folder.glob('*.jpg'))
    images.sort()
    n_ims = len(images)

    cameras = build_camera_matrices(gt_folder / '..')
    cam = cameras[cam_num]

    ground = np.genfromtxt(gt_folder / '..' / 'ground_points.txt', 
                           delimiter=',', dtype=np.float32).T

    with iio.get_writer(out_path, fps=20) as vid:
        for im_path in images:
            frame_no = int(im_path.stem)
            image = iio.imread(im_path)
            if folder is not None:
                attempt = read_positions(folder /   
                                         f"{long_str(frame_no, 6)}.json")
                gt_text = False # avoids clutter
            else:
                attempt = None
                gt_text = True
            gt = read_positions(gt_folder / f"{long_str(frame_no, 6)}.json")

            frame1 = render_pixel_frame(image, classes, frame_no, attempt, gt, 
                                        cam, colors, ground, gt_text=gt_text)
            frame2 = render_topdown_frame(frame1.shape, classes, attempt, gt, 
                                          cam, colors, ground, gt_text=gt_text)
            
            frame = np.vstack([frame1, frame2])
            vid.append_data(frame)

            if frame_no%100 == 0:
                print(f"Frame {frame_no+1}, {100.0*frame_no/n_ims:.2f}%, " \
                      f"Sequence: {seq_num}")

def rotated_rectangle(x, y, l, w, phi, edge_color, face_color):
    positions = list()
    base_pos = np.array([x, y], dtype=np.float32)
    forward = np.array([np.cos(phi), np.sin(phi)], dtype=np.float32)
    right = np.array([[0, -1], [1, 0]], dtype=np.float32) @ forward 
    for il, ii in zip((-0.5, 0.5), (1.0, -1.0)):
        ll = il * l * forward 
        for iw in (-0.5, 0.5):
            ww = ii*iw * w * right 
            new_pos = base_pos + ll + ww 
            positions.append((new_pos[0], new_pos[1]))
    xy = np.array(positions, dtype=np.float32)
    rect = patches.Polygon(xy, closed=False, ec=edge_color, fc=face_color)
    return rect 

def render_topdown_frame(dims:Tuple, classes:List[str], attempt:List[Dict],
                         ground_truth:List[Dict], cam:np.ndarray, colors:Dict, 
                         ground:np.ndarray, gt_text=False):
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.grid(True)
    ax.set_aspect('equal')

    plt.plot(ground[0,:], ground[1,:], '.', ms=1.0)
    minx, maxx = np.min(ground[0,:]), np.max(ground[0,:])
    miny, maxy = np.min(ground[1,:]), np.max(ground[1,:])

    for gt in ground_truth:
        class_name = gt['type']
        if not class_name in classes:
            continue 

        X, l, w = [gt[key] for key in "Xlw"]
        phi = np.arctan2(gt['forward_y'], gt['forward_x'])
        x, y = X.flatten()[0:2]

        color = brighter(colors[class_name])
        edge_color = matplotlib_color(color)
        face_color = matplotlib_color(color, 0.35)
        rect = rotated_rectangle(x, y, l, w, phi, edge_color, face_color)
        ax.add_patch(rect)

        if gt_text:
            plt.text(x, y, f"{gt['type']}{gt['id']}")

        minx = min(minx, x)
        miny = min(miny, y)
        maxx = max(maxx, x)
        maxy = max(maxy, y)

    if attempt is not None:
        for at in attempt:
            class_name = at['type']
            if not class_name in classes:
                continue 

            X, l, w = [at[key] for key in "Xlw"]
            phi = np.arctan2(at['forward_y'], at['forward_x'])
            x, y = X.flatten()[0:2]

            color = colors[class_name]
            edge_color = matplotlib_color(color)
            face_color = matplotlib_color(color, 0.75)
            rect = rotated_rectangle(x, y, l, w, phi, edge_color, face_color)
            ax.add_patch(rect)

            plt.text(x, y, f"{at['type']}{at['id']}")

            minx = min(minx, x)
            miny = min(miny, y)
            maxx = max(maxx, x)
            maxy = max(maxy, y)

    # Draw camera 
    cam_cen = pflat(null_space(cam)).flatten()
    cam_dir = normalize_numpy_vector(cam[2, 0:3]) * 7.5 
    plt.arrow(cam_cen[0], cam_cen[1], cam_dir[0], cam_dir[1], width=0.01,
              head_width=1.0)

    plt.xlim(minx - 2, maxx + 2)
    plt.ylim(miny - 2, maxy + 2)

    # Convert to image as a numpy array 
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, 0:3] # we only need RGB
    buf = good_resize(buf, dims[0], dims[1])

    plt.close()

    return buf 

def render_pixel_frame(image:np.ndarray, classes:List[str], frame_no:int, 
                       attempt:List[Dict], ground_truth:List[Dict], 
                       cam:np.ndarray, colors:Dict, ground:np.ndarray, 
                       gt_text=False):

    # Draw ground points 
    n = ground.shape[1]
    new_ground = np.ones((4, n), dtype=np.float32)
    new_ground[0:3, :] = ground
    ground2D = pflat(cam @ new_ground)
    for i in range(n):
        gnd = ground2D[:, i]
        pnt = (intr(gnd[0]), intr(gnd[1]))
        cv2.drawMarker(image, pnt, (255,255,255), cv2.MARKER_CROSS, 2)

    up = np.array([0,0,1,0], dtype=np.float32)

    # Draw ground truth 
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
        
        text = ""
        if gt_text:
            text=f"{gt['type']}{gt['id']}"
        draw3Dbox(image, cam, X, l, w, h, forward, right, up, color, text=text)

    # Draw the attempted tracks 
    if attempt is not None:
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
            draw3Dbox(image, cam, X, l, w, h, forward, right, up, color, 
                      text=f"{at['type']}{at['id']}")
        
    # Thicker black first, then thin white, very readable 
    cv2.putText(image, f"Frame {frame_no}", (10,20), cv2.FONT_HERSHEY_PLAIN, 
                1.5, (0,0,0), 2, cv2.LINE_AA)
    cv2.putText(image, f"Frame {frame_no}", (10,20), cv2.FONT_HERSHEY_PLAIN, 
                1.5, (255,255,255), 1, cv2.LINE_AA)

    return image 

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--folder", type=str,
                      help="Path to folder with the tracks to be visualized, " \
                           "can also be left empty to only visualize GT",
                           default="")
    args.add_argument("--gt_folder",  default="./output", type=str,
                      help="Path of ground truth",)
    args.add_argument("--set", type=str, default='test',
                      help="Either 'training', 'validation' or 'test''")
    args.add_argument("--seqs", type=str, default="", help="Set to only run "\
                      "these sequences. Should be comma-separated, like "\
                      "'0003,0027'.")
    args.add_argument("--classes", type=str, default="",
                      help="Set to a comma-separated list of classes to only " \
                           "visualize those")
    args.add_argument("--export", action="store_true", help="Include to write "\
                      "the video in the folder with the track. Otherwise, "\
                      "it will be written in the folder 'output'")
    args.add_argument("--processes", help="Number of processes to spawn, " \
                      "0 means single-threaded.", default=0, type=int)
    args = args.parse_args()

    if not args.folder:
        folder = None
    else:
        folder = Path(args.folder)
    gt_folder = Path(args.gt_folder)
    which_set = args.set

    if args.classes:
        classes = args.classes.split(',')
    else:
        classes = ['car', 'truck', 'bus', 'bicyclist', 'pedestrian']

    assert which_set in ['training', 'validation', 'test', 'all']
    seq_nums = get_seqnums()
    
    seq_nums['all'] = seq_nums['training'] + seq_nums['validation'] + \
                      seq_nums['test']

    to_visualize = list()

    seqs_to_run = seq_nums[which_set]
    if args.seqs:
        seqs_to_run = [int(s) for s in args.seqs.split(',')]

    for seq_num in seqs_to_run:
        seq = gt_folder / 'scenarios' / long_str(seq_num, 4)
        if folder is not None:
            _folder = folder / seq.name
        else:
            _folder = None 
        _gtfolder = seq / 'positions'

        if args.export:
            if folder is None:
                raise ValueError("Export is incompatible with not having " \
                                 "a folder with tracks to visualize")

            out_path = folder / f"utocs_visualization_{seq_num}.mp4"
        else:
            if folder is None:
                out_folder = Path('output') / 'visualized_gt'
                out_folder.mkdir(exist_ok=True, parents=True)
                out_path = out_folder / f"{seq_num}.mp4"
            else:
                out_folder = Path('output') / 'visualized_attempts'
                out_folder.mkdir(exist_ok=True, parents=True)
                out_path = out_folder / "utocs_visualization_" \
                                        f"{folder.name}_{seq_num}.mp4"

        to_visualize.append( (_folder, _gtfolder, classes, seq_num, out_path) )
    
    if args.processes == 0:
        for vis in to_visualize:
            visualize(*vis)
    else:
        from multiprocessing import Pool 
        import os 
        os.nice(10)
        with Pool(args.processes) as pool:
            pool.starmap(visualize, to_visualize)
