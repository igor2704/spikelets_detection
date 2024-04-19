import cv2
import typing as tp 
import numpy as np
from copy import deepcopy
from scipy.ndimage.filters import maximum_filter, minimum_filter


def get_centers_normal_mask(normal_mask: np.ndarray, scale=3):
    max_val = normal_mask.max()
    max_filter = maximum_filter(normal_mask, 3)
    max_filter = np.where(max_filter < max_val/scale, -1, max_filter)
    return np.where(normal_mask == max_filter)

def get_central_points_normal_mask(mask, scale=3):
    x, y = get_centers_normal_mask(mask, scale)
    all_coords = list(zip(x, y))
    coords = []
    flag = True
    for i, coord in enumerate(all_coords):
        x, y = coord
        for coord_2 in all_coords[i+1:]:
            x_2, y_2 = coord_2
            if np.abs(x - x_2) + np.abs(y - y_2) < 10:
                flag = False
        if flag:
            coords.append((x, y))
        flag = True
    return coords

def get_max_contour_mask(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = contours[np.argmax([cv2.contourArea(contour) for contour in contours])]
    return cv2.fillPoly(np.zeros(mask.shape, dtype='uint8'), pts=[max_contour], color=1)

def get_crop(mask: np.ndarray, eps_x: int = 10, eps_y: int = 5) -> np.ndarray:
    mask_nonzero_x, mask_nonzero_y = np.nonzero(mask)
    d_cr = int(np.max((np.min(mask_nonzero_x) - eps_x, 0)))
    l_cr = int(np.max((np.min(mask_nonzero_y) - eps_y, 0)))
    u_cr = int(np.min((np.max(mask_nonzero_x) + eps_x, mask.shape[0] - 1)))
    r_cr = int(np.min((np.max(mask_nonzero_y) + eps_y, mask.shape[1] - 1)))
    return d_cr, u_cr, l_cr, r_cr

def mask_max_contour_area(mask: np.array):
    contour_mask = get_max_contour_mask(mask)
    return np.sum(np.where(contour_mask > 0, 1, 0))

def generate_normal_mask(shape: tp.Tuple[int, int], 
                         scale: float, 
                         standard_mask: np.ndarray,
                         standart_mask_scale: float = 30000) -> np.ndarray:
    """
    Generate normal distribution mask with mean = (x, y)
    and covariance matrix = [[scale 0]
                             [0 scale]]
    Args:
        shape: mask shape.
        scale: mask scale. covariance matrix = [[scale 0]
                                                [0 scale]]
        standard_mask: mask with normal distribution.
        center_0: y
        center_1: x
        standart_mask_scale: standard_mask scale.
    
    Returns:
        np.ndarray: mask with normal distribution 
    """
    standard_shape = standard_mask.shape
    # convert standard_mask distribution to our distribution
    new_mask = standard_mask * 2 * np.pi * standart_mask_scale
    new_mask = np.power(new_mask, standart_mask_scale/scale)
    new_mask /= (scale * 2 * np.pi)
    return new_mask

def generate_pad_normal_mask(shape: tp.Tuple[int, int],
                             standard_mask: np.ndarray,
                             center_0: float,
                             center_1: float,
                             scale: float,
                             standart_mask_scale: float = 30000) -> np.ndarray:
    """
    Generate normal distribution mask with mean = (x, y)
    and covariance matrix = [[scale 0]
                             [0 scale]]
    Args:
        shape: mask shape.
        scale: mask scale. covariance matrix = [[scale 0]
                                                [0 scale]]
        standard_mask: mask with normal distribution.
        center_0: y
        center_1: x
        standart_mask_scale: standard_mask scale.
    
    Returns:
        np.ndarray: mask with normal distribution 
    """
    standard_shape = standard_mask.shape
    new_mask = generate_normal_mask(shape, scale, standard_mask, standart_mask_scale)
    pad_mask = np.zeros(shape, dtype=np.float64)
    # bounds
    up_b = max(center_0 - standard_shape[0] // 2, 0)
    down_b = min(center_0 + standard_shape[0] // 2, shape[0])
    left_b = max(center_1 - standard_shape[1] // 2, 0)
    right_b = min(center_1 + standard_shape[1] // 2, shape[1])
    # mask bounds
    up_b_m = max(standard_shape[0] // 2 - center_0 , 0)
    down_b_m = min(shape[0] - center_0 + standard_shape[0] // 2, standard_shape[0])
    left_b_m = max(standard_shape[1] // 2 - center_1, 0)
    right_b_m = min(shape[1] - center_1 + standard_shape[1] // 2, standard_shape[1])
#     if up_b >= down_b or left_b >= right_b or up_b_m >= down_b_m or left_b_m >= right_b_m:
#         print('-------------------------------------------------------------------------')
#         print('crop')
#         print(up_b, down_b, left_b, right_b)
#         print('crop mask')
#         print(up_b_m, down_b_m, left_b_m, right_b_m)
#         print('centers')
#         print(center_0, center_1)
#         print('shape')
#         print(shape[0], shape[1])
#         print('-------------------------------------------------------------------------')
    pad_mask[up_b:down_b, left_b:right_b] = new_mask[up_b_m:down_b_m, left_b_m:right_b_m]
    return pad_mask

def generate_sum_normal_mask(shape: tp.Tuple[int, int],
                             center_coords: tp.Sequence,
                             scale: float,
                             standard_mask: np.ndarray,
                             min_val: float = 1e-35,
                             standard_mask_scale: float = 30000) -> np.ndarray:
    """
    Generate sum of normal distribution mask with mean = center_coords
    and covariance matrix = [[scale 0]
                             [0 scale]]
    Args:
        shape: mask shape.
        scale: mask scale. covariance matrix = [[scale 0]
                                                [0 scale]]
        standard_mask: mask with normal distribution.
        center_coords: coordinates of center [(x, y), ...]
        standart_mask_scale: standard_mask scale.
        min_val: minimum value
    Returns:
        np.ndarray: mask with normal distribution 
    """
    mask = np.zeros(shape, dtype=np.float64)
    for center in center_coords:
        x, y = center
        x = int(x)
        y = int(y)
        mask += generate_pad_normal_mask(shape, standard_mask, 
                                         y, x, scale, standard_mask_scale)
    return mask.clip(min_val, 1)

def generate_masks_with_colorchecker_scale(shape: tp.Tuple[int, int],
                                           colorchecker_mask: np.ndarray,
                                           coords: tp.Sequence[int],
                                           standard_mask: np.ndarray,
                                           min_val: float = 1e-35,
                                           coeff: float = 25000,
                                           standard_mask_scale: float = 30000)-> np.ndarray:
    scale = mask_max_contour_area(colorchecker_mask) / coeff
    return generate_sum_normal_mask(shape, coords,
                                    scale, standard_mask, min_val, standard_mask_scale)

def get_random_crops(img: np.ndarray, mask: np.ndarray, mask_points: np.ndarray, coords: tp.List[tp.Tuple[int, int]],
                    gap_0: int = 0, gap_1: int = 0, random: bool = True) -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    crop_img = deepcopy(img)
    all_plant_mask = mask[..., 0].astype('int32') + mask[..., 1].astype('int32')
    all_plant_mask = np.where(all_plant_mask > 0, 1, 0).astype('uint8')
    for i in range(3):
        crop_img[..., i] *= all_plant_mask
        
    plant_mask = get_max_contour_mask(all_plant_mask)
    plant_mask = np.where(plant_mask > 0, 1, 0)
    body_mask = get_max_contour_mask(mask[..., 1])  
    body_mask = np.where(body_mask > 0, 1, 0)
    crop_body = get_crop(body_mask, gap_0, gap_1)
    crop_plant = get_crop(plant_mask, gap_0, gap_1)
    if not random:
        d_cr = int((crop_body[0] + crop_plant[0]) / 2)
        u_cr = int((crop_body[1] + crop_plant[1]) / 2)
        l_cr = int((crop_body[2] + crop_plant[2]) / 2)
        r_cr = int((crop_body[3] + crop_plant[3]) / 2)
    else:
        d_cr = np.random.randint(min(crop_body[0], crop_plant[0]),
                                 max(crop_body[0], crop_plant[0]) + 1)
        u_cr = np.random.randint(min(crop_body[1], crop_plant[1]),
                                 max(crop_body[1], crop_plant[1]) + 1)
        l_cr = np.random.randint(min(crop_body[2], crop_plant[2]),
                                 max(crop_body[2], crop_plant[2]) + 1)
        r_cr = np.random.randint(min(crop_body[3], crop_plant[3]),
                                 max(crop_body[3], crop_plant[3]) + 1)
    new_coords = list()
    for i in range(len(coords)):
        if coords[i][0] - l_cr < 0 or coords[i][1] - d_cr < 0:
            continue
        new_coords.append([coords[i][0] - l_cr, coords[i][1] - d_cr])
    return crop_img[d_cr:u_cr, l_cr:r_cr, :], mask_points[d_cr:u_cr, l_cr:r_cr], new_coords
