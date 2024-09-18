import os
import hydra
import typing as tp
from omegaconf import DictConfig

import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from copy import deepcopy

import torch
from torch.utils.data import DataLoader

from detection_models.models.unet_normal import NormalUNETSpikeletsNet
from detection_models.utils.normal_mask_utils import get_central_points_normal_mask, get_max_contour_mask, get_crop
from detection_models.datasets_and_augmentations.dataset_unet_normal import NormalSpikeletsInferDataset


def get_coords(img: np.ndarray, mask: np.ndarray, 
               coords: tp.List[tp.Tuple[int, int]],
               mask_point_size: tp.Tuple[int, int] = (512, 224),
               gap_0: int = 0, gap_1: int = 0):
    crop_img = deepcopy(img)
    all_plant_mask = mask[..., 0].astype('int32') + mask[..., 1].astype('int32')
    all_plant_mask = np.where(all_plant_mask > 0, 1, 0).astype('uint8')
    for i in range(3):
        crop_img[..., i] *= all_plant_mask
        
    plant_mask = get_max_contour_mask(all_plant_mask)
    plant_mask = np.where(plant_mask > 0, 1, 0)
    body_mask = get_max_contour_mask(mask[..., 1])  
    body_mask = np.where(body_mask > 0, 1, 0)
    crop_body = get_crop(body_mask, gap_0, gap_1)
    crop_plant = get_crop(plant_mask, gap_0, gap_1)
    
    d_cr = int((crop_body[0] + crop_plant[0]) / 2)
    u_cr = int((crop_body[1] + crop_plant[1]) / 2)
    l_cr = int((crop_body[2] + crop_plant[2]) / 2)
    r_cr = int((crop_body[3] + crop_plant[3]) / 2)
    
    w, h = abs(d_cr-u_cr), abs(l_cr-r_cr)
    coeff_h = h / mask_point_size[1]
    coeff_w = w / mask_point_size[0]
        
    new_coords = list()
    for i in range(len(coords)):
        new_coords.append([coords[i][1] * coeff_h + l_cr, 
                           coords[i][0] * coeff_w + d_cr])
    return new_coords 

@torch.no_grad()
def infer_kld(infer_dl, model, out_dir_path, in_dir_path, in_mask_dir_path, device='cuda', scale=5.5):
    model.eval()
    model.to(device)
    pred_count = list()
    names_lst = list()
    centers_lst = list()
    for batch in tqdm(infer_dl):
        imgs, names = batch
        imgs = imgs.to(device)
        predicton = model.predict(imgs)
        for pred, name in zip(predicton, names):
            centers = get_central_points_normal_mask(pred, scale=scale)
            names_lst.append(name)
            
            img_path = os.path.join(in_dir_path, name + '.jpg')
            segmentation_mask_path = os.path.join(in_mask_dir_path, name + '.png')
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            segmentation_mask = cv2.imread(segmentation_mask_path)
            segmentation_mask = cv2.cvtColor(segmentation_mask, cv2.COLOR_BGR2RGB)
            centers = get_coords(img, segmentation_mask, centers, pred.shape)
            
            for i, point in enumerate(centers):
                centers[i] = [int(point[0]), int(point[1])]
                
            pred_count.append(len(centers))
            centers_lst.append(centers)
            plt.figure()
            plot = sns.heatmap(pred)
            fig = plot.get_figure()
            fig.savefig(os.path.join(out_dir_path, f'{name}.jpg'))
    df = pd.DataFrame(list(zip(names_lst, pred_count)), 
                      columns=['Name', 'Spikelets Num'], index=None)
    df.to_csv(os.path.join(out_dir_path, 'spikelets_count.csv'), index=False)
    df_coords = pd.DataFrame(list(zip(names_lst, centers_lst)), 
                             columns=['Name', 'Spikelets Centers'], index=None)
    df_coords.to_csv(os.path.join(out_dir_path, 'coordinates.csv'), index=False)

@hydra.main(version_base=None, config_path='.', config_name='config_normal')
def main(cfg: DictConfig):
    # os.environ['KMP_DUPLICATE_LIB_OK']='True'
    infer_ds = NormalSpikeletsInferDataset(cfg.infer_params.in_dir_path, 
                                           cfg.infer_params.segmentation_mask_dir_path, 
                                           cfg.augmentation_params)
    infer_dl = DataLoader(infer_ds, cfg.infer_params.batch_size,
                          num_workers=cfg.infer_params.num_workers, shuffle=False)
    model = NormalUNETSpikeletsNet(wandb_log=False,
                                   local_log=False).to(cfg.infer_params.device)
    model.load(cfg.infer_params.model_path)
    infer_kld(infer_dl, model, cfg.infer_params.out_dir_path,
              cfg.infer_params.in_dir_path, cfg.infer_params.segmentation_mask_dir_path,
              device=cfg.infer_params.device, scale=cfg.model_params.threshold_scale_pred_mask)

if __name__ == '__main__':
    main()
    