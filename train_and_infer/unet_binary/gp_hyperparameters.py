import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt.plots import plot_convergence, plot_objective, plot_evaluations

from detection_models.datasets_and_augmentations.dataset_unet_binary import SpikeletsDataset
from detection_models.models.unet_binary import UNETSpikeletsNet


@hydra.main(version_base=None, config_path='.', config_name='config_binary')
def main(cfg: DictConfig):
    
    number_iter = 0
    space  = [Real(cfg.hyperparameters.lr[0], cfg.hyperparameters.lr[1], 
                   'log-uniform', name='lr'),
              Real(cfg.hyperparameters.cos_annealing_eta_min[0], cfg.hyperparameters.cos_annealing_eta_min[1],
                   'log-uniform', name='cos_annealing_eta_min'),
              Real(cfg.hyperparameters.radius[0], cfg.hyperparameters.radius[1], 
                   'log-uniform', name='radius'),
              Categorical(categories=cfg.hyperparameters.encoders, name='encoder') 
              ]
    
    @use_named_args(space)
    def objective(lr, cos_annealing_eta_min, radius, encoder):
        nonlocal number_iter
        if number_iter == 0:
            number_iter += 1
            return 4.76219 * (2 - 0.9065)
        params_dct = {'lr': lr,
                      'cos_annealing_eta_min': cos_annealing_eta_min,
                      'radius': radius,
                      'encoder': encoder}
        run = wandb.init(config=params_dct, project='GP_bce', 
                                      name=f'GP_{number_iter}')
        train_ds = SpikeletsDataset(cfg.pathes.train_path, cfg.pathes.train_mask_path, 
                                    cfg.pathes.train_mask_segment_path, cfg.augmentation_params, 
                                    radius)
        val_ds = SpikeletsDataset(cfg.pathes.val_path, cfg.pathes.val_mask_path, 
                                  cfg.pathes.val_mask_segment_path, cfg.augmentation_params, 
                                  radius, train=False)
        train_dl = DataLoader(train_ds, cfg.training_params.batch_size,
                              num_workers=cfg.training_params.num_workers, shuffle=True)
        val_dl = DataLoader(val_ds, cfg.training_params.batch_size,
                            num_workers=cfg.training_params.num_workers, shuffle=False)

        MyEarlyStopping = EarlyStopping(monitor='val_f1', mode='max', 
                                        patience=7, verbose=True)
        callbacks = [MyEarlyStopping]

        trainer = pl.Trainer(
            max_epochs=cfg.hyperparameters.epoches,
            devices=cfg.training_params.devices,
            accelerator=cfg.training_params.accelerator,
            callbacks=callbacks
        )
        module = UNETSpikeletsNet(lr=lr,
                                  max_epochs=cfg.hyperparameters.epoches,
                                  encoder_name=encoder,
                                  wandb_log=True,
                                  local_log=cfg.training_params.local_log,
                                  warmup_duration=cfg.model_params.warmup_duration,
                                  warmup_start_factor=cfg.model_params.warmup_start_factor,
                                  cos_annealing_eta_min=cos_annealing_eta_min)

        wandb.watch(module)
        trainer.fit(module, train_dl, val_dl)
        wandb.finish()
        
        number_iter += 1
        if module.best_f1 > 0.7:
            print(f'mae: {module.best_mae} f1: {module.best_f1}')
            return module.best_mae * (2 - module.best_f1)
        else:
            print(f'mae: {module.best_mae} f1: {module.best_f1}')
            return 10000000
    
    number_iter = 0
    res_gp = gp_minimize(objective, space, n_calls=cfg.hyperparameters.n_calls, 
                         random_state=42, n_initial_points=cfg.hyperparameters.n_initial_points)
    
    
if __name__ == '__main__':
    main()