import random
import os
import wandb

import cv2
import numpy as np
from copy import deepcopy
from tqdm import tqdm

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision.utils import make_grid, save_image

import segmentation_models_pytorch as smp

from detection_models.utils.metrics import iou, lerok_metrics, mae_mse
from detection_models.utils.get_params import get_lr
    

class UNETSpikeletsNet(pl.LightningModule):
    
    def __init__(self, 
                 lr: float = 9e-4,
                 max_epochs: int = 100, 
                 encoder_name: str = 'efficientnet-b4', 
                 wandb_log: bool = False,
                 local_log: bool = False,
                 warmup_duration: int = 5,
                 warmup_start_factor: float = 1e-4,
                 cos_annealing_eta_min: float = 1e-5,
                 dir_path_example_output: str = 'result'):
        super(UNETSpikeletsNet, self).__init__()
        self.model = smp.Unet(encoder_name=encoder_name, activation=None)
        self.loss = nn.BCELoss()
        self.lr = lr
        self.max_epochs = max_epochs
        self.warmup_duration = warmup_duration
        self.warmup_start_factor = warmup_start_factor,
        self.wandb_log = wandb_log
        self.cos_annealing_eta_min = cos_annealing_eta_min
        self.last_img_val_batch = None
        self.last_result_val_batch = None
        self.dir_path_example_output = dir_path_example_output
        self.number_epoch = 0
        self._save_imgs = False
        self.local_log = local_log
        self._val_metrics = {'val_loss': [],
                             'val_f1': [],
                             'val_precision': [],
                             'val_recall': [],
                             'mae': [],
                             'mse': [],
                             'iou': []}
        self.best_mae = 9e5
        if type(self.warmup_start_factor) in [list, tuple]:
            self.warmup_start_factor = self.warmup_start_factor[0]

    def training_step(self, batch, batch_idx):
        self.model.train()
        image, mask, _ = batch
        mask = mask.float().unsqueeze(1)
            
        predictions = torch.sigmoid(self.model(image))
        mask = (mask > 0.5).float()

        loss = self.loss(predictions, mask)
        
        preds = (predictions > 0.5).float()
        
        ious = iou(mask, preds)
        lr = get_lr(self.optimizer)
        
        if self.local_log:
            self.log('train_iou', ious, prog_bar=True)
            self.log('train_bce_loss', loss, prog_bar=True)
            self.log('lr_by_step', lr, prog_bar=True)
        if self.wandb_log:
            wandb.log({'train bce loss by step':loss, 'train iou by step':iou, 'lr by step':lr})
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        self.model.eval()
        image, mask, distance_batch = val_batch
        mask = mask.float().unsqueeze(1)
            
        predictions = torch.sigmoid(self.model(image))
        mask = (mask > 0.5).float()
        
        loss = self.loss(predictions, mask)
        
        predictions = (predictions > 0.5).float()
        
        if not self._save_imgs:
            self.last_img_val_batch = make_grid(image, nrow=4)
            self.last_result_val_batch = make_grid(predictions, nrow=4)
            self.last_mask_val_batch = make_grid(mask, nrow=4)
            self._save_imgs = True
        
        lerok_metrics_dct = lerok_metrics(mask.detach().cpu().numpy(), 
                                          predictions.detach().cpu().numpy(), 
                                          distance_batch.cpu().numpy(),
                                          False)
        mae_mse_dct = mae_mse(mask.detach().cpu().numpy(), 
                              predictions.detach().cpu().numpy())
        ious = iou(mask, predictions)
        self._val_metrics['val_loss'].append(float(loss))
        self._val_metrics['val_f1'].append(lerok_metrics_dct['f1'])
        self._val_metrics['val_precision'].append(lerok_metrics_dct['precision'])
        self._val_metrics['val_recall'].append(lerok_metrics_dct['recall'])
        self._val_metrics['mae'].append(mae_mse_dct['mae'])
        self._val_metrics['mse'].append(mae_mse_dct['mse'])
        self._val_metrics['iou'].append(float(ious))
    
    def on_validation_epoch_end(self):
        avg_loss = np.mean(self._val_metrics['val_loss'])
        avg_f1 = np.mean(self._val_metrics['val_f1'])
        avg_precision = np.mean(self._val_metrics['val_precision'])
        avg_recall = np.mean(self._val_metrics['val_recall'])
        avg_mae = np.mean(self._val_metrics['mae'])
        avg_mse = np.mean(self._val_metrics['mse'])
        avg_iou = np.mean(self._val_metrics['iou'])
        lr = get_lr(self.optimizer)
        if avg_mae < self.best_mae:
            self.best_mae = avg_mae
        
        if self.local_log:
            self.log('val_bce_loss', avg_loss, prog_bar=True)
            self.log('val_f1', avg_f1, prog_bar=True)
            self.log('val_precision', avg_precision, prog_bar=True)
            self.log('val_recall', avg_recall, prog_bar=True)
            self.log('val_mse', avg_mse, prog_bar=True)
            self.log('val_mae', avg_mae, prog_bar=True)
            self.log('iou', avg_iou, prog_bar=True)
            self.log('lr_by_epoch', lr, prog_bar=True)
        
        if self.wandb_log:
            wandb.log({ 'val bce loss by epoch':avg_loss, 
                       'val f1 by epoch': avg_f1,
                       'val precision by epoch': avg_precision,
                       'val recall by epoch': avg_recall,
                       'val mae by epoch': avg_mae,
                       'val mse by epoch': avg_mse,
                       'val iou by epoch': avg_iou,
                       'lr by epoch': lr})
            
        self.number_epoch += 1
        self._save_imgs = False
        if self.last_img_val_batch is not None:
            if self.wandb_log:
                wandb.log({'input images': wandb.Image(self.last_img_val_batch)})
        if self.last_result_val_batch is not None:
            if self.wandb_log:
                wandb.log({'output masks': wandb.Image(self.last_result_val_batch)})
        if self.last_mask_val_batch is not None:
             if self.wandb_log:
                wandb.log({'real masks': wandb.Image(self.last_mask_val_batch)})
        return avg_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.optimizer = optimizer
        
        if self.warmup_duration > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=self.warmup_start_factor,
                total_iters=self.warmup_duration,
                verbose=True
            )
            cos_annealing = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.max_epochs - self.warmup_duration,
                verbose=True,
                eta_min=self.cos_annealing_eta_min
            )
            lr_scheduler = {
                "scheduler": torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    [warmup, cos_annealing],
                    milestones=[self.warmup_duration],
                    verbose=True
                ),
                "interval": "epoch",
                "frequency": 1,
                "name": "cos_warmup_lr",
            }
        else:
            lr_scheduler = {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer,
                        T_max=self.max_epochs,
                        verbose=True,
                        eta_min=self.cos_annealing_eta_min
                    ),
                "interval": "epoch",
                "frequency": 1,
                "name": "cos_warmup_lr",
            }
        return [optimizer], [lr_scheduler]   
  

    def predict(self, x: torch.tensor):
        with torch.no_grad():
            return torch.squeeze((torch.sigmoid(self.model(x)) > 0.5), 1).float().detach().cpu().numpy()
    
    @torch.no_grad()
    def test(self, test_dl, device='cuda'):
        self.model.eval()
        precisions = []
        recalls = []
        f1s = []
        ious = []
        losses = []
        maes = []
        mses = []
        for batch in tqdm(test_dl):
            image, mask, distance_batch = batch
            image = image.to(device)
            mask = mask.to(device)
            mask = mask.float().unsqueeze(1)
            mask = (mask > 0.5).float()

            predictions = torch.sigmoid(self.model(image))
            loss = self.loss(predictions, mask)
            losses.append(loss.cpu().float())
            predictions = (predictions > 0.5).float()
            
            lerok_metrics_dct = lerok_metrics(mask.cpu().numpy(), 
                                          predictions.cpu().numpy(), 
                                          distance_batch.cpu().numpy(),
                                          False)
            mae_mse_dct = mae_mse(mask.detach().cpu().numpy(), 
                                  predictions.detach().cpu().numpy())
            maes.append(mae_mse_dct['mae'])
            mses.append(mae_mse_dct['mse'])
            f1s.append(lerok_metrics_dct['f1'])
            precisions.append(lerok_metrics_dct['precision'])
            recalls.append(lerok_metrics_dct['recall'])
            ious.append(iou(mask, predictions))
        return {'loss': np.mean(losses),
                'iou': np.mean(ious),
                'f1': np.mean(f1s),
                'precision': np.mean(precisions),
                'recall': np.mean(recalls),
                'mae': np.mean(maes),
                'mse': np.mean(mses)}
        
    def load(self, path: str, device: str = 'cuda'):
        checkpoint = torch.load(path, map_location=device)
        self.load_state_dict(checkpoint["state_dict"])
        return self
    