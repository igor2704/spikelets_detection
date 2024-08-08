import cv2
import numpy as np

from collections import defaultdict
from sklearn.metrics import DistanceMetric

from detection_models.utils.normal_mask_utils import get_central_points_normal_mask
from detection_models.utils.binary_mask_utils import get_central_points


def compute_confusion(points, pred_points, max_distance):
    if len(pred_points) * len(points) == 0:
        return { 'TP': 0, 'FP': 0, 'FN': len(points) }
    
    euclidean = DistanceMetric.get_metric('euclidean')
    distances = euclidean.pairwise(points, pred_points)

    pairs = list()

    for tidx, _ in enumerate(points):
        for pidx, _ in enumerate(pred_points):
            pairs.append((distances[tidx][pidx], tidx, pidx))

    sorted_by_distance_pairs = sorted(pairs, key=lambda x:x[0])

    correct_pairs = defaultdict(list)
    distributed_pred = set()

    for distance, tidx, pidx in sorted_by_distance_pairs:
        if distance > max_distance:
            break
        if pidx not in distributed_pred:
            correct_pairs[tidx].append(pidx)
            distributed_pred.add(pidx)

    confusion = { 'TP': 0, 'FP': 0,  'FN': 0 }        
    confusion['FN'] = len(points) - len(correct_pairs)

    for neighbors in correct_pairs.values():
        if len(neighbors) > 1:
            confusion['FP'] += len(neighbors) - 1

        confusion['TP'] += 1
        
    for i in range(len(pred_points)):
        if i not in distributed_pred:
            confusion['FP'] += 1
    return confusion
    
def detection_score(name, mask, pred_mask, distance, 
                    normal_mask=False, true_mask_scale=3, pred_mask_scale=5.5):
    if not normal_mask:
        points = get_central_points(mask)
        pred_points = get_central_points(pred_mask)
    else:
        points = get_central_points_normal_mask(mask, scale=true_mask_scale)
        pred_points = get_central_points_normal_mask(pred_mask, scale=pred_mask_scale)

    confusions = {}
    confusions = compute_confusion(points, pred_points, distance)
    
    return confusions
