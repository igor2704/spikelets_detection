import os
import json
import cv2
import numpy as np
from omegaconf import DictConfig

from torch.utils.data import Dataset

from detection_models.datasets_and_augmentations.unet_aug import get_transforms
from detection_models.utils.normal_mask_utils import generate_masks_with_colorchecker_scale, get_random_crops, get_max_contour_mask
from detection_models.utils.binary_mask_utils import get_radius


class NormalSpikeletsDataset(Dataset):
    def __init__(self, 
                 image_dir_path: str, 
                 point_mask_dir_path: str | None, 
                 segmentation_dir_path: str,
                 standard_mask_npy_path: str,
                 coords_json_path: str,
                 cfg_aug: DictConfig,
                 train: bool = True,
                 infer: bool = False,
                 min_val: float = 1e-35,
                 coeff: float = 25000,
                 standard_mask_scale: float = 30000):
        filenames_image = set([name.split('.')[0] for name in os.listdir(image_dir_path)])
        filenames_mask_points = set([name.split('.')[0] for name in os.listdir(point_mask_dir_path)])
        filenames_mask_seg = set([name.split('.')[0] for name in os.listdir(segmentation_dir_path)])
        self.filenames = list(filenames_image.intersection(filenames_mask_points).intersection(filenames_mask_seg))
        self.filenames = [name for name in self.filenames if name not in [' ', '']]
        self.image_dir_path = image_dir_path
        self.point_mask_dir_path = point_mask_dir_path
        self.segmentation_dir_path = segmentation_dir_path
        self.train = train
        self.standard_mask = np.load(standard_mask_npy_path)
        with open(coords_json_path, 'r') as f:
            self.coords_dct = json.load(f)
        self.infer = infer
        self.min_val = min_val
        self.coeff = coeff
        self.standard_mask_scale = standard_mask_scale
        self.cfg_aug = cfg_aug
        self.transform = get_transforms(self.cfg_aug)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir_path, self.filenames[index] + '.jpg')
        point_mask_path = os.path.join(self.point_mask_dir_path, self.filenames[index] + '.png')
        segmentation_mask_path = os.path.join(self.segmentation_dir_path, self.filenames[index] + '.png')
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        point_mask = cv2.imread(point_mask_path)
        point_mask = cv2.cvtColor(point_mask, cv2.COLOR_BGR2GRAY)
        segmentation_mask = cv2.imread(segmentation_mask_path)
        segmentation_mask = cv2.cvtColor(segmentation_mask, cv2.COLOR_BGR2RGB)
        
        crop_img, crop_mask, new_coords = get_random_crops(img, segmentation_mask, point_mask, 
                                                            self.coords_dct[self.filenames[index]], 
                                                            random=self.train)
        
        normal_mask = generate_masks_with_colorchecker_scale(crop_mask.shape,
                                                             segmentation_mask[..., 2], 
                                                             new_coords,
                                                             self.standard_mask,
                                                             min_val=self.min_val,
                                                             coeff=self.coeff,
                                                             standard_mask_scale=self.standard_mask_scale)
        
        crop_mask = cv2.resize(crop_mask, self.transform['size'])
        distance = 3 * get_radius(crop_mask)
        
        if self.train:
            augmentations = self.transform['train'](image=crop_img, mask=normal_mask)
        else:
            augmentations = self.transform['test'](image=crop_img, mask=normal_mask)
        
        if not self.infer: 
            return augmentations['image'], augmentations['mask'] / augmentations['mask'].sum(), distance
        else:
            name = self.filenames[index]
            return augmentations['image'], augmentations['mask'] / augmentations['mask'].sum(), distance, name 
    
    def __len__(self):
        return len(self.filenames)

    
class NormalSpikeletsInferDataset(Dataset):
    def __init__(self, 
                 image_dir_path: str, 
                 segmentation_dir_path: str,
                 cfg_aug: DictConfig,
                 rewrite_rotate: bool = False):
        filenames_image = set([name.split('.')[0] for name in os.listdir(image_dir_path)])
        filenames_mask_seg = set([name.split('.')[0] for name in os.listdir(segmentation_dir_path)])
        self.filenames = list(filenames_image.intersection(filenames_mask_seg))
        self.filenames = [name for name in self.filenames if name not in [' ', '']]
        self.image_dir_path = image_dir_path
        self.segmentation_dir_path = segmentation_dir_path
        self.cfg_aug = cfg_aug
        self.transform = get_transforms(self.cfg_aug)
        self.rewrite_rotate = rewrite_rotate

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir_path, self.filenames[index] + '.jpg')
        segmentation_mask_path = os.path.join(self.segmentation_dir_path, self.filenames[index] + '.png')
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        segmentation_mask = cv2.imread(segmentation_mask_path)
        segmentation_mask = cv2.cvtColor(segmentation_mask, cv2.COLOR_BGR2RGB)
        mask = get_max_contour_mask(segmentation_mask[..., 2])
        y, x = np.nonzero(mask)
        h = y.max() - y.min()
        w = x.max() - x.min()
        if h < w:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            segmentation_mask = cv2.rotate(segmentation_mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
            if self.rewrite_rotate:
                cv2.imwrite(img_path, img)
                cv2.imwrite(segmentation_mask_path, segmentation_mask)
        
        crop_img, _, _ = get_random_crops(img, segmentation_mask, np.zeros_like(img), [], random=False)
        
        augmentations = self.transform['test'](image=crop_img)
        
        return augmentations['image'], self.filenames[index]
    
    def __len__(self):
        return len(self.filenames)
    