import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from detection_models.datasets_and_augmentations.dataset_unet_normal import NormalSpikeletsDataset
from detection_models.models.unet_normal import NormalUNETSpikeletsNet


@hydra.main(version_base=None, config_path='.', config_name='config_normal')
def main(cfg: DictConfig):
    wandb_log = True
    if cfg.pathes.wandb_project == '' or  cfg.pathes.wandb_name == '':
        wandb_log = False
    if wandb_log:
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        run = wandb.init(config=cfg_dict, project=cfg.pathes.wandb_project, 
                         name=cfg.pathes.wandb_name)
    train_ds = NormalSpikeletsDataset(cfg.pathes.train_path, cfg.pathes.train_mask_path, cfg.pathes.train_mask_segment_path,
                                      cfg.pathes.standard_mask_path, cfg.pathes.train_json_coords_path,
                                      cfg.augmentation_params, 
                                      min_val=cfg.mask_params.min_val,
                                      coeff=cfg.mask_params.coeff,
                                      standard_mask_scale=cfg.mask_params.standard_mask_scale)
    val_ds = NormalSpikeletsDataset(cfg.pathes.val_path, cfg.pathes.val_mask_path, cfg.pathes.val_mask_segment_path,
                                    cfg.pathes.standard_mask_path, cfg.pathes.val_json_coords_path, cfg.augmentation_params,
                                    train=False,
                                    min_val=cfg.mask_params.min_val,
                                    coeff=cfg.mask_params.coeff,
                                    standard_mask_scale=cfg.mask_params.standard_mask_scale)
    train_dl = DataLoader(train_ds, cfg.training_params.batch_size,
                          num_workers=cfg.training_params.num_workers, shuffle=True)
    val_dl = DataLoader(val_ds, cfg.training_params.batch_size,
                        num_workers=cfg.training_params.num_workers, shuffle=False)
    
    MyTrainingModuleCheckpoint = ModelCheckpoint(dirpath=cfg.pathes.dirpath_models_saving,
                                            filename='unet_normal_{epoch}-{val_f1:.3f}' + f'_{cfg.model_params.encoder_name}',
                                            monitor='val_mae',
                                            mode='min',
                                            save_top_k=1)
    MyEarlyStopping = EarlyStopping(monitor='val_mae', mode='min', 
                                    patience=cfg.training_params.patience, verbose=True)
    callbacks = [MyEarlyStopping, MyTrainingModuleCheckpoint]

    trainer = pl.Trainer(
        max_epochs=cfg.training_params.max_epochs,
        devices=cfg.training_params.devices,
        accelerator=cfg.training_params.accelerator,
        callbacks=callbacks
    )
    module = NormalUNETSpikeletsNet(loss=cfg.model_params.loss,
                                    lr=cfg.model_params.lr,
                                    max_epochs=cfg.training_params.max_epochs,
                                    encoder_name=cfg.model_params.encoder_name,
                                    wandb_log=wandb_log,
                                    local_log=cfg.training_params.local_log,
                                    warmup_duration=cfg.model_params.warmup_duration,
                                    warmup_start_factor=cfg.model_params.warmup_start_factor,
                                    cos_annealing_eta_min=cfg.model_params.cos_annealing_eta_min,
                                    threshold_scale_true_mask=cfg.model_params.threshold_scale_true_mask,
                                    threshold_scale_pred_mask=cfg.model_params.threshold_scale_pred_mask)
    if wandb_log:
        wandb.watch(module)
    trainer.fit(module, train_dl, val_dl)
    
    if wandb_log:
        wandb.finish()


if __name__ == '__main__':
    main()
