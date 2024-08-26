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

from detection_models.models.unet_normal import NormalUNETSpikeletsNet
from detection_models.datasets_and_augmentations.dataset_unet_normal import NormalSpikeletsDataset



@hydra.main(version_base=None, config_path='.', config_name='config_normal')
def main(cfg: DictConfig):
    
    number_iter = 0
    space  = [Real(cfg.hyperparameters.lr[0], 
                   cfg.hyperparameters.lr[1], 
                   'log-uniform', name='lr'),
              Real(cfg.hyperparameters.thr_scale_pred_msk[0], 
                   cfg.hyperparameters.thr_scale_pred_msk[1],
                   'log-uniform', name='threshold_scale_pred_mask'),
              Real(cfg.hyperparameters.cos_annealing_eta_min[0], 
                   cfg.hyperparameters.cos_annealing_eta_min[1],
                   'log-uniform', name='cos_annealing_eta_min'),
              Real(cfg.hyperparameters.coeff[0], 
                   cfg.hyperparameters.coeff[1],
                   'log-uniform', name='coeff'),
              Categorical(categories=cfg.hyperparameters.encoders, name='encoder')
              ]
    
    @use_named_args(space)
    def objective(lr, threshold_scale_pred_mask, cos_annealing_eta_min, coeff, encoder):
        nonlocal number_iter
        if number_iter == 0:
            number_iter += 1
            return 2.91 * (2 - 0.8634)
        elif number_iter == 1:
            number_iter += 1
            return 14.682 * (2 - 0.6246)
        elif number_iter == 2:
            number_iter += 1
            return 10000000
        elif number_iter == 3:
            number_iter += 1
            return 10000000
        elif number_iter == 4:
            number_iter += 1
            return 6.078 * (2 - 0.7239)
        elif number_iter == 5:
            number_iter += 1
            return 2.867 * (2 - 0.8715)
        elif number_iter == 6:
            number_iter += 1
            return 5.497 * (2 - 0.8338)
        elif number_iter == 7:
            number_iter += 1
            return 4.727 * (2 - 0.7582)
        elif number_iter == 8:
            number_iter += 1
            return 3.686 * (2 - 0.8638)
        elif number_iter == 9:
            number_iter += 1
            return 2.625 * (2 - 0.8802)
        elif number_iter == 10:
            number_iter += 1
            return 3.745 * (2 - 0.8217)
        elif number_iter == 11:
            number_iter += 1
            return 10000000
        elif number_iter == 12:
            number_iter += 1
            return 3.943 * (2 - 0.8092)
        elif number_iter == 13:
            number_iter += 1
            return 10000000
        elif number_iter == 14:
            number_iter += 1
            return 5.177 * (2 - 0.7809)
        params_dct = {'lr': lr,
                      'scale coeff': coeff,
                      'cos_annealing_eta_min': cos_annealing_eta_min,
                      'threshold_scale_pred_mask': threshold_scale_pred_mask, 
                      'encoder': encoder}
        run = wandb.init(config=params_dct, project='GP_kld', 
                                      name=f'GP_{number_iter}')
        train_ds = NormalSpikeletsDataset(cfg.pathes.train_path, cfg.pathes.train_mask_path, cfg.pathes.train_mask_segment_path,
                                      cfg.pathes.standard_mask_path, cfg.pathes.train_json_coords_path,
                                      cfg.augmentation_params, 
                                      min_val=cfg.mask_params.min_val,
                                      coeff=coeff,
                                      standard_mask_scale=cfg.mask_params.standard_mask_scale)
        val_ds = NormalSpikeletsDataset(cfg.pathes.val_path, cfg.pathes.val_mask_path, cfg.pathes.val_mask_segment_path,
                                    cfg.pathes.standard_mask_path, cfg.pathes.val_json_coords_path, cfg.augmentation_params,
                                    train=False,
                                    min_val=cfg.mask_params.min_val,
                                    coeff=coeff,
                                    standard_mask_scale=cfg.mask_params.standard_mask_scale)
        train_dl = DataLoader(train_ds, cfg.training_params.batch_size,
                              num_workers=cfg.training_params.num_workers, shuffle=True)
        val_dl = DataLoader(val_ds, cfg.training_params.batch_size,
                            num_workers=cfg.training_params.num_workers, shuffle=False)

        MyEarlyStopping = EarlyStopping(monitor='val_mae', mode='min', 
                                        patience=7, verbose=True)
        callbacks = [MyEarlyStopping]

        trainer = pl.Trainer(
            max_epochs=cfg.hyperparameters.epoches,
            devices=cfg.training_params.devices,
            accelerator=cfg.training_params.accelerator,
            callbacks=callbacks
        )
        module = NormalUNETSpikeletsNet(loss='kld',
                                        lr=lr,
                                        max_epochs=cfg.hyperparameters.epoches,
                                        encoder_name=encoder,
                                        wandb_log=True,
                                        local_log=cfg.training_params.local_log,
                                        warmup_duration=cfg.model_params.warmup_duration,
                                        warmup_start_factor=cfg.model_params.warmup_start_factor,
                                        cos_annealing_eta_min=cos_annealing_eta_min,
                                        threshold_scale_true_mask=cfg.model_params.threshold_scale_true_mask,
                                        threshold_scale_pred_mask=threshold_scale_pred_mask)
        wandb.watch(module)
        trainer.fit(module, train_dl, val_dl)
        wandb.finish()
        
        number_iter += 1
        if module.best_f1 > 0.6:
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