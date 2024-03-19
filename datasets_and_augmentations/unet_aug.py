import hydra
import albumentations as A
from omegaconf import DictConfig

from albumentations.pytorch import ToTensorV2


def get_transforms(cfg: DictConfig):
    transform_train = A.Compose([
        A.Resize(cfg.image_height, cfg.image_width),
        A.HorizontalFlip(p=cfg.p_horizontal_flip),
        A.VerticalFlip(p=cfg.p_vertical_flip),
        A.Rotate(p=cfg.p_rotate, limit=cfg.limit_rotate),
        A.GaussianBlur(blur_limit=tuple(cfg.blur_limit), p=cfg.p_blur),
        A.GaussNoise(p=cfg.p_gauss_noise),
        A.RandomBrightnessContrast(p=cfg.p_random_brightness_contrast),
        A.RGBShift(r_shift_limit=cfg.r_shift_limit, g_shift_limit=cfg.g_shift_limit, 
                   b_shift_limit=cfg.b_shift_limit, p=cfg.p_rgb_shift),
        A.ColorJitter(brightness=cfg.brightness, contrast=cfg.contrast, 
                      saturation=cfg.saturation, hue=cfg.hue, p=cfg.p_color_jitter),
        A.ToGray(p=cfg.p_to_gray),
        A.Normalize(),
        ToTensorV2()
    ])


    transform_test = A.Compose([
        A.Resize(cfg.image_height, cfg.image_width),
        A.Normalize(),
        ToTensorV2()
    ])
    return {'size': (cfg.image_height, cfg.image_width),
            'train': transform_train,
            'test': transform_test}