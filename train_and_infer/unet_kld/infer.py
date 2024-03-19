import os
import hydra
from omegaconf import DictConfig

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from detection_models.models.unet_kld import NormalUNETSpikeletsNet
from detection_models.utils.normal_mask_utils import get_central_points_normal_mask
from detection_models.datasets_and_augmentations.dataset_unet_kld import NormalSpikeletsInferDataset


@torch.no_grad()
def infer_kld(infer_dl, model, out_dir_path, device='cuda', scale=5.5):
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
            for i, point in enumerate(centers):
                centers[i] = [int(point[0]), int(point[1])]
            names_lst.append(name)
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

@hydra.main(version_base=None, config_path='.', config_name='config_kld')
def main(cfg: DictConfig):
    infer_ds = NormalSpikeletsInferDataset(cfg.infer_params.in_dir_path, 
                                           cfg.infer_params.segmentation_mask_dir_path, 
                                           cfg.augmentation_params)
    infer_dl = DataLoader(infer_ds, cfg.infer_params.batch_size,
                          num_workers=cfg.infer_params.num_workers, shuffle=False)
    model = NormalUNETSpikeletsNet().to(cfg.infer_params.device)
    model.load(cfg.infer_params.model_path)
    infer_kld(infer_dl, model, cfg.infer_params.out_dir_path, 
              device=cfg.infer_params.device, scale=cfg.model_params.threshold_scale_pred_mask)

if __name__ == '__main__':
    main()
    