import os
import json
import cv2
import numpy as np
from omegaconf import DictConfig

from torch.utils.data import Dataset

from detection_models.datasets_and_augmentations.unet_aug import get_transforms
from detection_models.utils.binary_mask_utils import get_random_crops, get_max_contour_mask, get_radius, generate_binary_mask


class SpikeletsDataset(Dataset):
    def __init__(self, 
                 image_dir_path: str, 
                 point_mask_dir_path: str | None, 
                 segmentation_dir_path: str,
                 cfg_aug: DictConfig,
                 radius: float = 1,
                 train: bool = True):
        filenames_image = set([name.split('.')[0] for name in os.listdir(image_dir_path)])
        filenames_mask_points = set([name.split('.')[0] for name in os.listdir(point_mask_dir_path)])
        filenames_mask_seg = set([name.split('.')[0] for name in os.listdir(segmentation_dir_path)])
        self.filenames = list(filenames_image.intersection(filenames_mask_points).intersection(filenames_mask_seg))
        self.filenames = [name for name in self.filenames if name not in [' ', '']]
        self.image_dir_path = image_dir_path
        self.point_mask_dir_path = point_mask_dir_path
        self.segmentation_dir_path = segmentation_dir_path
        self.train = train
        self.radius = radius
        self.transform = get_transforms(cfg_aug)

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
        
        if np.all(np.equal(point_mask, np.zeros_like(point_mask))) or len(np.array(point_mask).shape) < 2:
            print(img_path)
            print(segmentation_mask_path)
            print(point_mask_path)
        
        crop_img, crop_mask = get_random_crops(img, segmentation_mask, point_mask, random=self.train)
        crop_mask_dist = cv2.resize(crop_mask, self.transform['size'])
        distance = 3 * get_radius(crop_mask_dist)
        
        crop_mask = generate_binary_mask(crop_mask, self.radius)
        if np.all(np.equal(crop_mask, np.zeros_like(crop_mask))) or len(np.array(crop_mask).shape) < 2:
            print(img_path)
            print(segmentation_mask_path)
            print(point_mask_path)
        
        if self.train:
            augmentations =  self.transform['train'](image=crop_img, mask=crop_mask)
        else:
            augmentations = self.transform['test'](image=crop_img, mask=crop_mask)
        
        return augmentations['image'], augmentations['mask'], distance
    
    def __len__(self):
        return len(self.filenames)

    
class SpikeletsInferDataset(Dataset):
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
        
        crop_img, _ = get_random_crops(img, segmentation_mask, np.zeros_like(img), random=False)
        
        augmentations = self.transform['test'](image=crop_img)
        
        return augmentations['image'], self.filenames[index]
    
    def __len__(self):
        return len(self.filenames)
    