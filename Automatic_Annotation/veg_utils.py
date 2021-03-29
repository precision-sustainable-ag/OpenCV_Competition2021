import os
import torch
import cv2
import numpy as np
from PIL import Image
import random
from scipy import ndimage
import matplotlib.pyplot as plt
import time
from skimage import morphology
import matplotlib.patches as patches
import glob
from pathlib import Path




######################################################
################### PREPROCESSINGS ###################
######################################################

def get_images(img_dir):
    # Creates list of image paths of multiple file tpyes in directoty
    types = ('*.jpeg', '*.jpg','*.JPEG','*.JPG','*.png','*.PNG') # the tuple of file types
    img_list = []
    for exten in types:
        glob_dir = Path(img_dir) / exten
        img_list.extend(glob.glob(str(glob_dir)))
    return img_list


def hist_equa(bgr_img, color_CLAHE=False):
    # brg_img: np array in [BGR] channel order
    b,g,r = cv2.split(bgr_img)
    
    if color_CLAHE:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))            
        equ_r = clahe.apply(r)
        equ_g = clahe.apply(g)
        equ_b = clahe.apply(b)
    else:
        equ_r = cv2.equalizeHist(r)
        equ_g = cv2.equalizeHist(g)
        equ_b = cv2.equalizeHist(b)
    
    # RGB 
    equ = cv2.merge((equ_r, equ_g,equ_b ))
    
    # Returns RGB color image
    return equ

def CLAHE_hist(img: "np.int8"):
    # Contrast limiting adaptive histogram equalization (CLAHE).
    # Contrast amplification is limited to reduce noise amplification.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))            
    clahe_img = clahe.apply(img)

    return clahe_img

def otsu_thresh(mask, kernel_size=(5,5)):
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

def make_ndi(rgb_img):
    # rgb_img: np array in [RGB] channel order
    # exr: single band vegetation index as np array
    # NDI = (G + R) / (G - R) 
    img = rgb_img.astype(float)

    blue = img[:,:,2]
    green = img[:,:,1]
    red = img[:,:,0]
    
    ndi = (green + red) / (green - red)
    print("Max ndi: ", ndi.max())
    print("Min ndi: ", ndi.min())

    return ndi

def make_exr(rgb_img):
    # rgb_img: np array in [RGB] channel order
    # exr: single band vegetation index as np array
    # EXR = 1.4 * R - G
    img = rgb_img.astype(float)

    blue = img[:,:,2]
    green = img[:,:,1]
    red = img[:,:,0]
    
    exr = 1.4 * red - green
    exr = np.where(exr < 0, 0, exr).astype('uint8') # Thresholding removes low negative values
    return exr

def exg_minus_exr(rgb_img):
    exg = make_exg(rgb_img)
    exr = make_exr(rgb_img)

    exgr = exg - exr
    exgr = np.where(exgr < 25, 0, exgr).astype('uint8')
    
    return exgr



######################################################
############ MORPHOLOGICAL OPERATIONS ################
######################################################

def reduce_holes(mask, kernel_size, min_object_size, min_hole_size, iterations, dilation=True):
    # takes in ExG 2D np array (reduces holes from ExG)
    # and outputs boolean mask
    mask = morphology.remove_small_holes(morphology.remove_small_objects(mask, min_object_size),min_hole_size)
    mask = morphology.opening(mask, morphology.disk(3))
    mask = mask.astype('float32')
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,kernel_size)
    if dilation:
        mask = cv2.dilate(mask,kernel,iterations = iterations)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, (5,5))
    cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, (7,7))
    return cleaned_mask

def filter_topn_components(mask: np.int8, top_n: int) -> np.ndarray:

    # calculate size of individual components and chooses based on min size
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    # size of components except 0 (background)
    sizes = stats[1:, -1]; nb_components = nb_components - 1
    # determines number of components to segment
    top_n_sizes = sorted(sizes, reverse=True)[:top_n]
    min_size = min(top_n_sizes) - 1

    filtered_mask = np.zeros((output.shape))
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            filtered_mask[output == i + 1] = 255

    return filtered_mask

def filter_by_componenet_size(mask: np.int8, top_n: int) -> 'list[np.ndarray]':
    # calculate size of individual components and chooses based on min size
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    # size of components except 0 (background)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1
    # Determines number of components to segment
    # Sort components from largest to smallest
    top_n_sizes = sorted(sizes, reverse=True)[:top_n]
    min_size = min(top_n_sizes) - 1
    list_filtered_masks = []
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            filtered_mask = np.zeros((output.shape))
            filtered_mask[output == i + 1] = 255
            list_filtered_masks.append(filtered_mask)

    return list_filtered_masks


def label_mask(mask, top_n):
    component_mask = filter_by_concomponents(mask.astype('uint8'), top_n)
    # labels components
    blobs_labels = measure.label(component_mask, background=0, connectivity=2)
    return blobs_labels

##########################################################
################### EXTRACT FOREGROUND ###################
##########################################################


def create_foreground(img, mask, add_padding=False):
    # applys mask to create RGBA foreground using PIL
    
    if len(np.array(mask).shape) == 3:
        mask = np.asarray(mask)[:,:,0]
    else:
        mask = np.asarray(mask)
    rgba = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
    # extract from image using mask
    rgba[:, :, 3][mask[:,:]==0] = 0
    foreground = Image.fromarray(rgba)
    # crop foreground to content
    if add_padding:    
        pil_crop_frground = foreground.crop((foreground.getbbox()[0] - 3,foreground.getbbox()[1] - 3, foreground.getbbox()[2] + 3, foreground.getbbox()[3] + 3 ))
    else:
        pil_crop_frground = foreground.crop(foreground.getbbox())
    
    return pil_crop_frground

def extract_save_ind_frgrd(
    img: np.array, 
    top_masks: "list(np.ndarray)", 
    imgp: str,
    save_frgd_dir: str,
    testing=False):
    
    assert type(top_masks) is list, "top_masks is not a list of np.arrays"
    
    frgd_id = 0
    # Crop image using mask to create RGBA foreground (optional: 3 px padding)    
    for component_mask in top_masks:
        exg_frgd = create_foreground(img, component_mask, add_padding=True)     

        if not os.path.exists(save_frgd_dir):
            os.makedirs(save_frgd_dir)

        fname_prefix = os.path.splitext(os.path.basename(imgp))[0]

        frgd_path = os.path.join(save_frgd_dir, fname_prefix + "_" + str(frgd_id) + ".png" )
        exg_frgd.save(frgd_path)
        frgd_id += 1
        if testing:
            bgr_imgpath = os.path.join(save_frgd_dir, fname_prefix + "_ORIGINAL" + ".png" )
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(bgr_imgpath, img_bgr)
    return exg_frgd

def extract_detection_foreground(
    img,
    imgp, 
    component_mask, 
    save_frgd_dir,
    detection_id, 
    testing=False):

    
    # Crop image using mask to create RGBA foreground (optional: 3 px padding)    
    # for component_mask in top_masks:
    exg_frgd = create_foreground(img, component_mask, add_padding=True)     

    if not os.path.exists(save_frgd_dir):
        os.makedirs(save_frgd_dir)

    fname_prefix = os.path.splitext(os.path.basename(imgp))[0]

    frgd_path = os.path.join(save_frgd_dir, fname_prefix + "_" + "detection_" + str(detection_id) + ".png" )
    exg_frgd.save(frgd_path)
    if testing:
        bgr_imgpath = os.path.join(save_frgd_dir, fname_prefix + "_ORIGINAL" + ".png" )
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(bgr_imgpath, img_bgr)
    



########################################################
############### OBJECT DETECTIOON UTILS ################
########################################################

def xywh_2_xyxy(x, img_shape):
    
    # h, w = img_shape[:2]
    
    x = np.array(x)
    y = np.zeros_like(x)

    center_x = x[...,0]
    center_y = x[...,1]
    width = x[..., 2]
    height = x[..., 3]

    y[...,0] = center_x - width/2
    y[...,1] = center_y - height/2
    y[...,2] = center_x + width/2
    y[...,3] = center_y + height/2
    
    return y


def vegimg2label_path(img_path):
    # Define label paths as a function of image paths
    # Get single labels, some images don't have detection results
    label_name = os.path.splitext(os.path.basename(img_path))[0] + '.txt'
    img_dir = os.path.dirname(img_path)
    label_dir = os.path.join(img_dir, 'labels')
    label_path = os.path.join(label_dir, label_name)
    if not os.path.exists(label_path):
        return None
    return label_path


def extract_boxes(label_path, img_shape):
    h, w = img_shape[:2]
    list_bboxes = []
    with open(label_path, 'r') as f:
        lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels

        for j, x in enumerate(lb):
            # # print(x[1:])
            b = x[1:] * [w, h, w, h]  # box
            # # b[2:] = b[2:] * 1.2 + 3  # pad
            # b = [int(item) for item in b]
            # # print(b)
            # # b = xywh_2_xyxy(b.reshape(-1, 4)).ravel().astype(np.float)
            # b[[0, 2]] = np.clip(b[[0, 2]], 0, w)  # clip boxes outside of image
            # b[[1, 3]] = np.clip(b[[1, 3]], 0, h)
            list_bboxes.append(b)
    
        # print(list_bboxes)  
    
    return list_bboxes

def bbox_areas(bboxes):
    areas = []
    for box in bboxes:
        # grab the coordinates of the bounding boxes
        x1 = box[0]
        y1 = box[1]
        x2 = box[2]
        y2 = box[3]
        # Compute the area of each bounding boxes and store in list
        area = (x2 - x1) * (y2 - y1)    
        areas.append(area)
    return areas

def box_area(box):
    
    # grab the coordinates of the bounding boxes
    x1 = box[0]
    y1 = box[1]
    x2 = box[2]
    y2 = box[3]
    # Compute the area of each bounding boxes and store in list
    area = (x2 - x1) * (y2 - y1)    
    return area




# plt.figure(figsize=(28, 20))
# plt.suptitle("Vegetation Indices", fontsize=48)
# plt.subplot(231); plt.imshow(exg_mask);plt.title('EXG w/o\n CLAHE Equalization', fontsize=38)
# plt.subplot(232); plt.imshow(CLAHE_equ_exg_mask); plt.title("EXG with\n CLAHE Equalization",fontsize=38)
# plt.subplot(234); plt.imshow(exgr_WO_mask);plt.title('EXG-EXR w/o\n CLAHE Equalization', fontsize=38)
# plt.subplot(235); plt.imshow(CLAHE_equ_exgr_mask); plt.title("EXG-EXR with\n CLAHE Equalization",fontsize=38)
# plt.subplot(233); plt.imshow(img);plt.title('Original Image', fontsize=38)

# plt.tight_layout()

# plt.savefig(opt.save_foreground_dir + "/" +str(frgd_id) + "VEG_INDEX_Figure.png")
# plt.close()

# plt.figure(figsize=(36, 28))
# plt.suptitle("Foregrounds", fontsize=48)
# plt.subplot(231); plt.imshow(exg_frgd);plt.title('EXG w/o\n CLAHE Equalization', fontsize=38)
# plt.subplot(232); plt.imshow(exg_w_CLAHE_frgd); plt.title("EXG with\n CLAHE Equalization",fontsize=38)
# plt.subplot(234); plt.imshow(exgr_WO_frgd);plt.title('EXG-EXR w/o\n CLAHE Equalization', fontsize=38)
# plt.subplot(235); plt.imshow(exgr_w_CLAHE_frgd); plt.title("EXG-EXR with\n CLAHE Equalization",fontsize=38)
# plt.subplot(233); plt.imshow(img);plt.title('Original Image', fontsize=38)

# plt.tight_layout()

# plt.savefig(opt.save_foreground_dir + "/" +str(frgd_id) + "VEG_FOREGROUNDS_INDEX_Figure.png")

# frgd_id += 1