import torch
import numpy as np

from detection_models.utils.normalise_distance import detection_score
from detection_models.utils.normal_mask_utils import get_central_points_normal_mask
from detection_models.utils.binary_mask_utils import get_central_points


def iou(mask1: torch.Tensor, mask2: torch.Tensor) -> float:
    intersection = (mask1 * mask2).sum()
    if intersection == 0:
        return 0.0
    union = torch.logical_or(mask1, mask2).to(torch.int).sum()
    return float(intersection / union)

def lerok_metrics(mask_batch: np.ndarray, prediction_batch: np.ndarray,
                  distance_batch: np.ndarray, normal_mask: bool = False,
                  true_mask_scale=3, pred_mask_scale=5.5) -> dict[str, list[float]]:
    precisions = []
    recalls = []
    f1s = []
    for true_mask, pred_mask, distance in zip(mask_batch, 
                                              prediction_batch, 
                                              distance_batch):
        confusions = detection_score('', true_mask[0],
                                     pred_mask[0], distance,
                                     normal_mask=normal_mask,
                                     true_mask_scale=true_mask_scale,
                                     pred_mask_scale=pred_mask_scale)

        if confusions["TP"] + confusions["FP"] == 0:
            continue
        precision = confusions["TP"] / (confusions["TP"] + confusions["FP"])
        recall = confusions["TP"] / (confusions["TP"] + confusions["FN"])

        if precision + recall == 0:
            fscore = 0
        else:
            fscore = 2 * (precision*recall) / (precision+recall)

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(fscore)

    if len(f1s) == 0:
        f1s = [0]
    if len(precisions) == 0:
        precisions = [0]
    if len(recalls) == 0:
        recalls = [0]
    return {'f1': np.mean(f1s),
            'precision': np.mean(precisions),
            'recall': np.mean(recalls)}

def mae_mse(true_mask_batch: np.ndarray, pred_mask_batch: np.ndarray,
        normal: bool = False, scale_tr_msk: float = 3, scale_pred_msk: float = 5.5):
    true_mask_count, pred_mask_count = [], []
    for true_mask, pred_mask in zip(true_mask_batch, pred_mask_batch):
        if normal:
            true_mask_count.append(len(get_central_points_normal_mask(true_mask[0], scale_tr_msk)))
            pred_mask_count.append(len(get_central_points_normal_mask(pred_mask[0], scale_pred_msk)))
        else:
            true_mask_count.append(len(get_central_points(true_mask[0])))
            pred_mask_count.append(len(get_central_points(pred_mask[0])))
    true_mask_count = np.array(true_mask_count)
    pred_mask_count = np.array(pred_mask_count)
    return {'mse': ((true_mask_count - pred_mask_count) ** 2).mean(),
            'mae': (np.abs(true_mask_count - pred_mask_count)).mean()}
      