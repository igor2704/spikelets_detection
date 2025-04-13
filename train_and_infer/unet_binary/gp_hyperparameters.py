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
            return 2.96414 * (2 - 0.86067)
        elif number_iter == 1:
            number_iter += 1
            return 4.19675 * (2 - 0.8027)
        elif number_iter == 2:
            number_iter += 1
            return 5 * 37.77377 * (2 - 0.0081023)
        elif number_iter == 3:
            number_iter += 1
            return 13.38867 * (2 - 0.88383)
        elif number_iter == 4:
            number_iter += 1
            return 5 * 53.25668 * (2 - 0.0069558)
        elif number_iter == 5:
            number_iter += 1
            return 21.61789 * (2 - 0.87394)
        elif number_iter == 6:
            number_iter += 1
            return 5 * 40.57066 * (2 - 0.72734)
        elif number_iter == 7:
            number_iter += 1
            return 8.80119 * (2 - 0.89251)
        elif number_iter == 8:
            number_iter += 1
            return 30.4169 * (2 - 0.80546)
        elif number_iter == 9:
            number_iter += 1
            return 5 * 45.12631 * (2 - 0.5358)
        elif number_iter == 10:
            number_iter += 1
            return 5 * 10.26095 * (2 - 0.50574)
        elif number_iter == 11:
            number_iter += 1
            return 5 * 23.75424 * (2 - 0.7818)
        elif number_iter == 12:
            number_iter += 1
            return 8.97583 * (2 - 0.88978)
        elif number_iter == 13:
            number_iter += 1
            return 7.2141 * (2 - 0.8496)
        elif number_iter == 14:
            number_iter += 1
            return 15.25994 * (2 - 0.8605)
        elif number_iter == 15:
            number_iter += 1
            return 14.52761 * (2 - 0.8847)
        elif number_iter == 16:
            number_iter += 1
            return 5 * 26.07448 * (2 - 0.77549)
        elif number_iter == 17:
            number_iter += 1
            return 5 * 25.425 * (2 - 0.74141)
        elif number_iter == 18:
            number_iter += 1
            return 30.69287 * (2 - 0.80405)
        elif number_iter == 19:
            number_iter += 1
            return 5 * 107.68242 * (2 - 0.26986)
        
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
                                        patience=15, verbose=True)
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
        if module.best_f1 > 0.8:
            print(f'mae: {module.best_mae} f1: {module.best_f1}')
            return module.best_mae * (2 - module.best_f1)
        else:
            print(f'mae: {module.best_mae} f1: {module.best_f1}')
            return 5 * module.best_mae * (2 - module.best_f1)
    
    number_iter = 0
    res_gp = gp_minimize(objective, space, n_calls=cfg.hyperparameters.n_calls, 
                         random_state=42, n_initial_points=cfg.hyperparameters.n_initial_points)
    
    
if __name__ == '__main__':
    main()