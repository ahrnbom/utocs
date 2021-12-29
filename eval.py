"""
    Use MOTMetrics to compute score between ground truth and a list of tracks
"""

import motmetrics as mm 
from dataclasses import dataclass
import numpy as np
from scipy.linalg.decomp_svd import null_space 
import shapely.geometry
import shapely.affinity
from typing import List
from pathlib import Path 
import json 
import argparse

from util import long_str, vector_dist, pflat, print_table
from cameras import build_camera_matrices


@dataclass
class RoadUser:
    X:np.ndarray
    type:str
    id:int
    shape:np.ndarray
    forward:np.ndarray

def get_seqnums():
    sets_lines = [l for l in Path('sets.txt').read_text().split('\n') if l]
    seq_nums = dict()
    for line in sets_lines:
        splot = line.split(': ')
        set_name = splot[0]
        seq_nums[set_name] = [int(v) for v in splot[1].split(' ')]
    return seq_nums 

def euclidean_dist3D(gt:RoadUser, ru:RoadUser):
    if gt.type == ru.type:
        dist = vector_dist(gt.X[0:2], ru.X[0:2])
        return dist 
    else:
        return np.nan

def iou_dist3D(gt:RoadUser, ru:RoadUser, min_iou:float) -> float:
    if gt.type == ru.type:
        iou = iou3D(gt, ru)
        if iou < min_iou:
            return np.nan
        return 1.0 - iou # IOU "distance" is defined like this
    else:
        return np.nan

def iou3D(A:RoadUser, B:RoadUser) -> float:
    a_rect = rotated_rectangle(A)
    b_rect = rotated_rectangle(B)
    intersection = a_rect.intersection(b_rect).area
    union = a_rect.union(b_rect).area
    iou = intersection / union 
    return iou

def rotated_rectangle(ru:RoadUser):
    x, y = ru.X[0:2]
    l, w = ru.shape[0:2]

    # "Native" shapely solution, more than twice as slow for some reason
    #phi = np.arctan2(ru.forward[1], ru.forward[0])
    #box = shapely.geometry.box(-l/2.0, -w/2.0, l/2.0, w/2.0)
    #box = shapely.affinity.rotate(box, 360.0*phi/(2.0*np.pi))
    #box = shapely.affinity.translate(box, x, y)
    #return box 

    positions = list()
    base_pos = np.array([x, y], dtype=np.float32)
    forward = ru.forward[0:2]
    right = np.array([[0, -1], [1, 0]], dtype=np.float32) @ forward 
    for il, ii in zip((-0.5, 0.5), (1.0, -1.0)):
        ll = il * l * forward 
        for iw in (-0.5, 0.5):
            ww = ii*iw * w * right 
            new_pos = base_pos + ll + ww 
            positions.append((new_pos[0], new_pos[1]))
    box = shapely.geometry.Polygon(positions)
    return box 

# Loads road users from JSON file
def load_instances(folder:Path, frame_no:int, classes=None) -> List[RoadUser]:
    file_path = folder / f"{long_str(frame_no, 6)}.json"
    file_text = file_path.read_text()
    objs = json.loads(file_text)
    rus = list()
    for obj in objs:

        # Exclude this object if not in the predefined list of included classes
        if classes is not None:
            if not obj['type'] in classes:
                continue

        X = np.array([obj['x'], obj['y'], obj['z']], dtype=np.float32)
        shape = np.array([obj['l'], obj['w'], obj['h']], dtype=np.float32)
        forward = np.array([obj['forward_x'], obj['forward_y'], 
                            obj['forward_z']], 
                            dtype=np.float32)
        ru =  RoadUser(X, obj['type'], obj['id'], shape, forward)
        rus.append(ru)
    return rus 

def evaluate_scenario(tr_folder:Path,
                      gt_folder:Path,
                      min_iou:float=0.25,
                      name:str='some_tracker',
                      classes=None,
                      verbose=True,
                      metric='iou') -> float:

    acc = mm.MOTAccumulator(auto_id=True)

    frames = [int(f.stem) for f in gt_folder.glob('*.json')]
    frames.sort()
    
    for frame_no in frames:
        gt_instances = load_instances(gt_folder, frame_no, classes)
        gt_ids = [g.id for g in gt_instances]

        tr_instances = load_instances(tr_folder, frame_no, classes)
        tr_ids = [t.id for t in tr_instances]

        # Present distances to the format expected by MOTMetrics
        dists = list()
        for g in gt_instances:
            these_dists = list()
            for t in tr_instances:
                if metric == 'iou':
                    dist = iou_dist3D(g, t, min_iou)
                elif metric == 'euclidean':
                    dist = euclidean_dist3D(g, t)
                these_dists.append(dist)
            dists.append(these_dists)

        acc.update(gt_ids, tr_ids, dists)
    
    # Compute the metrics
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=['mota'], name=name)

    if verbose:
        print(summary)

    return float(summary['mota'])

def main():
    args = argparse.ArgumentParser()

    args.add_argument("--folder", type=str,
                      help="Path to folder with the tracks to be evaluated, " \
                           "can also be multiple ones separated by commas")
    args.add_argument("--gt_folder",  default="./output", type=str,
                      help="Path of ground truth",)
    args.add_argument("--set", type=str, default='test',
                      help="Either 'training', 'validation' or 'test")
    args.add_argument("--classes", type=str, default="",
                      help="Set to a comma-separated list of classes to only " \
                           "evaluate those")
    args.add_argument("--iou_thresh", type=float, default=0.25,
                      help="Usually 0.25, how much the road user's rotated " \
                           "rectangles must overlap to consider it a hit, " \
                            "only relevant if metric is iou")
    args.add_argument("--metric", type=str, default="iou",
                      help="Either 'iou' or 'euclidean'")
    args = args.parse_args()

    folders = [Path(f) for f in args.folder.split(',')]
    for folder in folders:
        assert folder.is_dir(), f"Cannot file folder {folder}"
    gt_folder = Path(args.gt_folder)
    iou_thresh = args.iou_thresh
    which_set = args.set
    metric = args.metric

    assert metric in ['iou', 'euclidean']

    if args.classes:
        classes = args.classes.split(',')
    else:
        # None means all classes are included
        classes = None 

    assert which_set in ['training', 'validation', 'test']
    seq_nums = get_seqnums()
    seqs = seq_nums[which_set]
    motas = np.zeros((len(seqs), len(folders)), dtype=np.float32)
    for i_seq, seq_num in enumerate(seqs):
        seq = gt_folder / 'scenarios' / long_str(seq_num, 4)
        for i_folder, folder in enumerate(folders):
            mota = evaluate_scenario(folder / seq.name, 
                                    seq / 'positions',
                                    iou_thresh, name=f"{folder.name}{seq_num}", 
                                    classes=classes,
                                    verbose=False,
                                    metric=metric)
            
            motas[i_seq, i_folder] = mota 
        
        print_table(seqs[:i_seq+1], [f.name for f in folders], 
                    motas[:i_seq+1, :])
        print("")

    for i_folder, folder in enumerate(folders):    
        mean_mota = np.mean(motas[:, i_folder])
        print(f"Average MOTA for method {folder.name} is {mean_mota}")

if __name__=="__main__":
    main()

