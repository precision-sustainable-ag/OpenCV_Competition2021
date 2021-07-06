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

########################################################
###################### CLASSIFY  #######################
########################################################

def specify_species(img_list, species):
    """ Gets list of image paths by specific row numbers. 
    Uses species name to call specific rows. """

    all_imgs = [str(k) for k in img_list]

    if species == 'clover':
        row1  = [str(k) for k in all_imgs if 'row1_' in k]
        row2 = [str(k) for k in all_imgs if 'row2_' in k]
        row3 = [str(k) for k in all_imgs if 'row3_' in k]
        row4 = [str(k) for k in all_imgs if 'row4_' in k]
        row5 = [str(k) for k in all_imgs if 'row5_' in k]
        target = itertools.chain(row1, row2, row3, row4, row5)
        target = list(set(target))
        target_posix = [Path(k) for k in target]
    
    if species == 'cowpea':
        row1  = [str(k) for k in all_imgs if 'row5' in k]
        row2 = [str(k) for k in all_imgs if 'row6' in k]
        row3 = [str(k) for k in all_imgs if 'row7' in k]
        row4 = [str(k) for k in all_imgs if 'row8' in k]
        target = itertools.chain(row1, row2, row3, row4)
        target = list(set(target))
        target_posix = [Path(k) for k in target]
    
    if species == 'horseweed':
        row1  = [str(k) for k in all_imgs if 'row4' in k]
        row2 = [str(k) for k in all_imgs if 'row5' in k]
        row3 = [str(k) for k in all_imgs if 'row6' in k]
        row4 = [str(k) for k in all_imgs if 'row7' in k]
        row5 = [str(k) for k in all_imgs if 'row8' in k]
        target = itertools.chain(row1, row2, row3, row4, row5)
        target = list(set(target))
        target_posix = [Path(k) for k in target]

    if species == 'goosefoot' or species == 'sunflower' or species == 'velvetleaf':
        row1  = [str(k) for k in all_imgs if 'row4' in k]
        row2 = [str(k) for k in all_imgs if 'row5' in k]
        row3 = [str(k) for k in all_imgs if 'row6' in k]
        row4 = [str(k) for k in all_imgs if 'row7' in k]
        row5 = [str(k) for k in all_imgs if 'row8' in k]
        row6 = [str(k) for k in all_imgs if 'row9' in k]
        target = itertools.chain(row1, row2, row3, row4, row5, row6)
        target = list(set(target))
        target_posix = [Path(k) for k in target]
    
    if species == 'grasses':
        row1  = [str(k) for k in all_imgs if 'row8' in k]
        row2 = [str(k) for k in all_imgs if 'row9' in k]
        row3 = [str(k) for k in all_imgs if 'row10' in k]
        row4 = [str(k) for k in all_imgs if 'row11' in k]
        row5 = [str(k) for k in all_imgs if 'row12' in k]
        target = itertools.chain(row1, row2, row3, row4, row5)
        target = list(set(target))
        target_posix = [Path(k) for k in target]

    if species == 'all':
        row1  = [str(k) for k in all_imgs if 'row1' in k]
        row2 = [str(k) for k in all_imgs if 'row2' in k]
        row3 = [str(k) for k in all_imgs if 'row3' in k]
        row4 = [str(k) for k in all_imgs if 'row4' in k]
        row5 = [str(k) for k in all_imgs if 'row5' in k]
        row6  = [str(k) for k in all_imgs if 'row6' in k]
        row7 = [str(k) for k in all_imgs if 'row7' in k]
        row8 = [str(k) for k in all_imgs if 'row8' in k]
        row9 = [str(k) for k in all_imgs if 'row9' in k]
        row10 = [str(k) for k in all_imgs if 'row10' in k]
        row11 = [str(k) for k in all_imgs if 'row11' in k]
        row12 = [str(k) for k in all_imgs if 'row12' in k]
        
        target = itertools.chain(
            row1, row2, row3, row4, row5, row6,
            row7, row8, row9, row10, row11, row12 )
        
        target = list(set(target))
        target_posix = [Path(k) for k in target]

    return target_posix