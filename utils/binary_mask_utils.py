import cv2
import typing as tp
import numpy as np
from copy import deepcopy


def get_central_points(mask):
#     print(mask.shape)
#     print(type(mask))
#     print(mask.min())
#     print(mask.max())
    contours, _ = cv2.findContours(mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE )
    central_points = list()

    for contour in contours:
        try:
            moments = cv2.moments(contour)

            cx = int(moments['m10']/moments['m00'])
            cy = int(moments['m01']/moments['m00'])

            central_points.append([cx, cy])
        except:
            pass

    return central_points

def get_max_contour_mask(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = contours[np.argmax([cv2.contourArea(contour) for contour in contours])]
    return cv2.fillPoly(np.zeros(mask.shape, dtype='uint8'), pts=[max_contour], color=1)

def get_radius(mask: np.ndarray) -> np.ndarray:
    try:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_contour = contours[np.argmax([cv2.contourArea(contour) for contour in contours])]
        area = cv2.contourArea(max_contour)
    except:
        area = 0
    return np.sqrt(area / np.pi)

def get_crop(mask: np.ndarray, eps_x: int = 350, eps_y: int = 200) -> np.ndarray:
    mask_nonzero_x, mask_nonzero_y = np.nonzero(mask)
    d_cr = int(np.max((np.min(mask_nonzero_x) - eps_x, 0)))
    l_cr = int(np.max((np.min(mask_nonzero_y) - eps_y, 0)))
    u_cr = int(np.min((np.max(mask_nonzero_x) + eps_x, mask.shape[0] - 1)))
    r_cr = int(np.min((np.max(mask_nonzero_y) + eps_y, mask.shape[1] - 1)))
    return d_cr, u_cr, l_cr, r_cr

def get_random_crops(img: np.ndarray, mask: np.ndarray, mask_points: np.ndarray,
                    gap_0: int = 30, gap_1: int = 15, random: bool = True) -> tuple[np.ndarray, np.ndarray]:
    crop_img = deepcopy(img)
    all_plant_mask = mask[..., 0] + mask[..., 1]
    all_plant_mask = np.where(all_plant_mask > 0, 1, 0).astype('uint8')
    for i in range(3):
        crop_img[..., i] *= all_plant_mask
        
    plant_mask = get_max_contour_mask(mask[..., 0] + mask[..., 1])
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
        
    return crop_img[d_cr:u_cr, l_cr:r_cr, :], mask_points[d_cr:u_cr, l_cr:r_cr]
