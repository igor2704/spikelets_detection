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
            return 3.41336 * (2 - 0.84601)
        elif number_iter == 1:
            number_iter += 1
            return 5 * 7.67212 * (2 - 0.76376)
        elif number_iter == 2:
            number_iter += 1
            return 5 * 21.11551 * (2 - 0.60535)
        elif number_iter == 3:
            number_iter += 1
            return 5 * 7.69302 * (2 - 0.79589)
        elif number_iter == 4:
            number_iter += 1
            return 4.11391 * (2 - 0.81705)
        elif number_iter == 5:
            number_iter += 1
            return 2.82129 * (2 - 0.89331)
        elif number_iter == 6: # 
            number_iter += 1
            return 5 * 14.58585 * (2 - 0.24309)
        elif number_iter == 7:
            number_iter += 1
            return 3.99521 * (2 - 0.81298)
        elif number_iter == 8:
            number_iter += 1
            return 4.56067 * (2 - 0.89359)
        elif number_iter == 9:
            number_iter += 1
            return 3.08658 * (2 - 0.89301)
        elif number_iter == 10:
            number_iter += 1
            return 5 * 7.09243 * (2 - 0.63782)
        elif number_iter == 11:
            number_iter += 1
            return 5 * 17.67704 * (2 - 0.18849)
        elif number_iter == 12:
            number_iter += 1
            return 5 * 17.8773 * (2 - 0.084844)
        elif number_iter == 13:
            number_iter += 1
            return 5 * 8.69248 * (2 - 0.78852)
        elif number_iter == 14:
            number_iter += 1
            return 3.5596 * (2 - 0.8476)
        elif number_iter == 15:
            number_iter += 1
            return 2.75273 * (2 - 0.90418)
        elif number_iter == 16:
            number_iter += 1
            return 8.35006 * (2 - 0.89398)
        elif number_iter == 17:
            number_iter += 1
            return 2.45315 * (2 - 0.90538)
        elif number_iter == 18:
            number_iter += 1
            return 2.96872 * (2 - 0.88485)
        elif number_iter == 19:
            number_iter += 1
            return 4.23405 * (2 - 0.8493)
        elif number_iter == 20:
            number_iter += 1
            return 2.18625 * (2 - 0.89988)
        elif number_iter == 21:
            number_iter += 1
            return 5 * 18.76922 * (2 - 0.79589)
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