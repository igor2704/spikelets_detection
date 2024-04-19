import random
import numpy as np

import os
import hydra
import json
from omegaconf import DictConfig

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from detection_models.models.unet_normal import NormalUNETSpikeletsNet
from detection_models.utils.normal_mask_utils import get_central_points_normal_mask
from detection_models.datasets_and_augmentations.dataset_unet_normal import NormalSpikeletsDataset


@hydra.main(version_base=None, config_path='.', config_name='config_normal')
def main(cfg: DictConfig):
    seed_value = 42
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    test_ds = NormalSpikeletsDataset(cfg.pathes.test_path, cfg.pathes.test_mask_path, cfg.pathes.test_mask_segment_path,
                                    cfg.pathes.standard_mask_path, cfg.pathes.test_json_coords_path, cfg.augmentation_params,
                                    train=False,
                                    min_val=cfg.mask_params.min_val,
                                    coeff=cfg.mask_params.coeff,
                                    standard_mask_scale=cfg.mask_params.standard_mask_scale)
    test_dl = DataLoader(test_ds, cfg.test_params.batch_size,
                          num_workers=cfg.test_params.num_workers, shuffle=False)
    model = NormalUNETSpikeletsNet(loss=cfg.model_params.loss,
                                    lr=cfg.model_params.lr,
                                    max_epochs=cfg.training_params.max_epochs,
                                    encoder_name=cfg.model_params.encoder_name,
                                    wandb_log=False,
                                    local_log=cfg.training_params.local_log,
                                    warmup_duration=cfg.model_params.warmup_duration,
                                    warmup_start_factor=cfg.model_params.warmup_start_factor,
                                    cos_annealing_eta_min=cfg.model_params.cos_annealing_eta_min,
                                    threshold_scale_true_mask=cfg.model_params.threshold_scale_true_mask,
                                    threshold_scale_pred_mask=cfg.model_params.threshold_scale_pred_mask
                                    ).to(cfg.test_params.device)
    model.load(cfg.test_params.model_path)
    model.eval()
    dct = model.test(test_dl, cfg.test_params.device)
    for key, val in dct.items():
        dct[key] = float(val)
    with open(cfg.test_params.out_json_file_name, 'w') as f:
        json.dump(dct, f)
        
        
if __name__ == '__main__':
    main()
