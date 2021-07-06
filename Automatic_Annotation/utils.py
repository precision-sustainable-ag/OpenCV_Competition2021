import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import itertools


def otsu_thresh(mask, kernel_size=(3,3)):
    mask_blur = cv2.GaussianBlur(mask, kernel_size,0).astype('uint8')
    ret3,mask_th3 = cv2.threshold(mask_blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return mask_th3


######################################################
############# VEGETATION INDICES #####################
######################################################

def make_exg(rgb_img, exg_thresh=False):
    # rgb_img: np array in [RGB] channel order
    # exr: single band vegetation index as np array
    # EXG = 2 * G - R - B
    img = rgb_img.astype(float)
    blue = img[:,:,2]
    green = img[:,:,1]
    red = img[:,:,0]
    exg = 2*green - red - blue
    if exg_thresh:
        exg = np.where(exg < 0, 0, exg).astype('uint8') # Thresholding removes low negative values
    return exg

######################################################
############ MORPHOLOGICAL OPERATIONS ################
######################################################


def filter_by_component_size(mask: np.int8, top_n: int) -> 'list[np.ndarray]':
    # calculate size of individual components and chooses based on min size
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    # size of components except 0 (background)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1
    # Determines number of components to segment
    # Sort components from largest to smallest
    top_n_sizes = sorted(sizes, reverse=True)[:top_n]
    try:
        min_size = min(top_n_sizes) - 1
    except:
        min_size = 0
    list_filtered_masks = []
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            filtered_mask = np.zeros((output.shape))
            filtered_mask[output == i + 1] = 255
            list_filtered_masks.append(filtered_mask)

    return list_filtered_masks

##########################################################
################### EXTRACT FOREGROUND ###################
##########################################################
  
def create_foreground(img, mask, add_padding=False, crop_to_content=True):
    # applys mask to create RGBA foreground using PIL

    if len(np.array(mask).shape) == 3:
        mask = np.asarray(mask)[:,:,0]
    else:
        mask = np.asarray(mask)
    rgba = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
    # extract from image using mask
    rgba[:, :, 3][mask==0] = 0
    
    foreground = Image.fromarray(rgba)
    # crop foreground to content
    if add_padding:
        pil_crop_frground = foreground.crop((foreground.getbbox()[0] - 3,foreground.getbbox()[1] - 3, foreground.getbbox()[2] + 3, foreground.getbbox()[3] + 3 ))
    else:
        if crop_to_content:
            pil_crop_frground = foreground.crop(foreground.getbbox())
        else:
            pil_crop_frground = foreground
    return pil_crop_frground