import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from detection_models.datasets_and_augmentations.dataset_unet_binary import SpikeletsDataset
from detection_models.models.unet_binary import UNETSpikeletsNet


@hydra.main(version_base=None, config_path='.', config_name='config_binary')
def main(cfg: DictConfig):
    wandb_log = True
    if cfg.pathes.wandb_project == '' or  cfg.pathes.wandb_name == '':
        wandb_log = False
    if wandb_log:
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        run = wandb.init(config=cfg_dict, project=cfg.pathes.wandb_project, 
                         name=cfg.pathes.wandb_name)
    train_ds = SpikeletsDataset(cfg.pathes.train_path, cfg.pathes.train_mask_path, 
                                cfg.pathes.train_mask_segment_path, cfg.augmentation_params, 
                                cfg.model_params.radius)
    val_ds = SpikeletsDataset(cfg.pathes.val_path, cfg.pathes.val_mask_path, 
                              cfg.pathes.val_mask_segment_path, cfg.augmentation_params, 
                              cfg.model_params.radius, train=False)
    train_dl = DataLoader(train_ds, cfg.training_params.batch_size,
                          num_workers=cfg.training_params.num_workers, shuffle=True)
    val_dl = DataLoader(val_ds, cfg.training_params.batch_size,
                        num_workers=cfg.training_params.num_workers, shuffle=False)
    
    MyTrainingModuleCheckpoint = ModelCheckpoint(dirpath=cfg.pathes.dirpath_models_saving,
                                            filename='unet_binary_{epoch}-{val_f1:.3f}' + f'_{cfg.model_params.encoder_name}',
                                            monitor='val_f1',
                                            mode='max',
                                            save_top_k=1)
    MyEarlyStopping = EarlyStopping(monitor='val_f1', mode='max', 
                                    patience=cfg.training_params.patience, verbose=True)
    callbacks = [MyEarlyStopping, MyTrainingModuleCheckpoint]

    trainer = pl.Trainer(
        max_epochs=cfg.training_params.max_epochs,
        devices=cfg.training_params.devices,
        accelerator=cfg.training_params.accelerator,
        callbacks=callbacks
    )
    module = UNETSpikeletsNet(lr=cfg.model_params.lr,
                              max_epochs=cfg.training_params.max_epochs,
                              encoder_name=cfg.model_params.encoder_name,
                              wandb_log=wandb_log,
                              local_log=cfg.training_params.local_log,
                              warmup_duration=cfg.model_params.warmup_duration,
                              warmup_start_factor=cfg.model_params.warmup_start_factor,
                              cos_annealing_eta_min=cfg.model_params.cos_annealing_eta_min)
    if wandb_log:
        wandb.watch(module)
    trainer.fit(module, train_dl, val_dl)
    
    if wandb_log:
        wandb.finish()
    
    
if __name__ == '__main__':
    main()
    