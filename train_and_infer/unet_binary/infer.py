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

from detection_models.models.unet_binary import UNETSpikeletsNet
from detection_models.utils.binary_mask_utils import get_central_points, get_max_contour_mask, get_crop, get_radius
from detection_models.datasets_and_augmentations.dataset_unet_binary import SpikeletsInferDataset


def get_coords_mask_circle(img: np.ndarray, mask: np.ndarray, binary_mask:np.ndarray,
               coords: tp.List[tp.Tuple[int, int]], mode:str,
               mask_point_size: tp.Tuple[int, int] = (512, 224),
               gap_0: int = 0, gap_1: int = 0, color=(255, 0, 0), thickness=5):
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
        new_coords.append([coords[i][0] * coeff_h + l_cr, 
                           coords[i][1] * coeff_w + d_cr])

    new_mask = np.zeros_like(img[:,:,0])
    new_mask[d_cr:u_cr, l_cr:r_cr] = cv2.resize(binary_mask, img[d_cr:u_cr, l_cr:r_cr][:,:,0].shape[::-1]) * 255
    new_mask = np.where(new_mask > 0, 255, 0)
    if 'circle' in mode:
        r = get_radius(new_mask)
        for coord in new_coords:
            cv2.circle(img, (int(coord[0]), int(coord[1])), int(r), color, thickness) 
    return new_coords, new_mask, img

@torch.no_grad()
def infer_bce(infer_dl, model, out_dir_path, in_dir_path, in_mask_dir_path, 
              device='cuda', mode='mask', color=(255, 0, 0), thickness=5):
    # mode: 'mask' 'circle' 'mask and circle'
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
            centers = get_central_points(pred)
            names_lst.append(name)
            
            # jpg
            img_path = os.path.join(in_dir_path, name + '.jpg')
            segmentation_mask_path = os.path.join(in_mask_dir_path, name + '.png')
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            segmentation_mask = cv2.imread(segmentation_mask_path)
            segmentation_mask = cv2.cvtColor(segmentation_mask, cv2.COLOR_BGR2RGB)
#             y, x = np.nonzero(mask)
#             h = y.max() - y.min()
#             w = x.max() - x.min()
#             if h < w:
#                 img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
#                 segmentation_mask = cv2.rotate(segmentation_mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
            centers, new_mask, img = get_coords_mask_circle(img, segmentation_mask, pred, 
                                                            centers, mask_point_size=pred.shape,
                                                            mode=mode, color=color, thickness=thickness)
            
            for i, point in enumerate(centers):
                centers[i] = [int(point[0]), int(point[1])]
                
            pred_count.append(len(centers))
            
            if 'mask' in mode:
                cv2.imwrite(os.path.join(out_dir_path, f'mask_{name}.jpg'), new_mask)
            if 'circle' in mode:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(out_dir_path, f'circle_{name}.jpg'), img)
            centers_lst.append(centers)
            
    df = pd.DataFrame(list(zip(names_lst, pred_count)), 
                      columns=['Name', 'Spikelets Num'], index=None)
    df.to_csv(os.path.join(out_dir_path, 'spikelets_count.csv'), index=False)
    df_coords = pd.DataFrame(list(zip(names_lst, centers_lst)), 
                             columns=['Name', 'Spikelets Centers'], index=None)
    df_coords.to_csv(os.path.join(out_dir_path, 'coordinates.csv'), index=False)

@hydra.main(version_base=None, config_path='.', config_name='config_binary')
def main(cfg: DictConfig):
    # os.environ['KMP_DUPLICATE_LIB_OK']='True'
    infer_ds = SpikeletsInferDataset(cfg.infer_params.in_dir_path, 
                                     cfg.infer_params.segmentation_mask_dir_path, 
                                     cfg.augmentation_params)
    infer_dl = DataLoader(infer_ds, cfg.infer_params.batch_size,
                          num_workers=cfg.infer_params.num_workers, shuffle=False)
    model = UNETSpikeletsNet(encoder_name=cfg.model_params.encoder_name,
                             wandb_log=False,
                             local_log=False).to(cfg.infer_params.device)
    model.load(cfg.infer_params.model_path)
    infer_bce(infer_dl, model, cfg.infer_params.out_dir_path,
              cfg.infer_params.in_dir_path, cfg.infer_params.segmentation_mask_dir_path,
              device=cfg.infer_params.device, mode=cfg.infer_params.mode,
              thickness=cfg.infer_params.thickness)

if __name__ == '__main__':
    main()