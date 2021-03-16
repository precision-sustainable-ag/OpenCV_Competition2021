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


# TODO: Find principal bbox from multiple 


def hist_equa(rgb_img):
    # rgb_img: np array in [RGB] channel order
    
    r,g,b = cv2.split(rgb_img)
    
    equ_b = cv2.equalizeHist(b)
    equ_g = cv2.equalizeHist(g)
    equ_r = cv2.equalizeHist(r)
    
    equ = cv2.merge((equ_r, equ_g,equ_b ))
    
    return equ

def make_exg(rgb_img):
    # rgb_img: np array in [RGB] channel order
    # exr: single band vegetation index as np array
    # EXG = 2 * G - R - B
    img = rgb_img.astype(float)
    blue = img[:,:,2]
    green = img[:,:,1]
    red = img[:,:,0]
    exg = (2*green) - red - blue
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

    return exr

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

def filter_by_concomponents(mask, top_n):
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

def label_mask(mask, top_n):
    component_mask = filter_by_concomponents(mask.astype('uint8'), top_n)
    # labels components
    blobs_labels = measure.label(component_mask, background=0, connectivity=2)
    return blobs_labels

def predict_bbox(imgp_list,model,classes, conf_thres = 0.7, nms_thres = 0.4, img_size=416):
    """ 
    Takes in list of images and trained model to predict bbox and localize vegetation of interest.
    Creates a list of images, bbox, and color IDs. 
    """
    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    ############ PREDICTION ################
    for img_path in imgp_list:
        # Read in image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(img_size,img_size)).transpose(2,0,1)
        img = img.reshape(1,3,img_size,img_size)
        # Make predictions
        with torch.no_grad():
            if torch.cuda.is_available():
                input_img = torch.from_numpy(img).type(torch.cuda.FloatTensor)/255
            else:
                input_img = torch.from_numpy(img).type(torch.FloatTensor)/255
            unalt_detections = model(input_img) # center x, center y, width, height
            detections = non_max_suppression(unalt_detections, conf_thres, nms_thres)
        # Append to results
        imgs.append(img_path)
        img_detections.extend(detections)
    

    ########### IMAGE/BBOX PAIRS #############
    img_bboxcoord_pairs = []
    for img_path, detection in zip(imgs, img_detections):
        img_bboxcoord = {}
        color_id = 1
        img = np.array(Image.open(img_path))
        # Draw bounding boxes and labels of detections
        if detection is not None:
            # Recale bbox
            rescaled_detection = rescale_boxes(detection, img_size, img.shape[:2])
            unique_labels = rescaled_detection[:, -1].cpu().unique().numpy() # remove if needed
            # Compile img/bbox pairs
            img_bboxcoord = {"image_path": img_path, "bboxes": [], "class_pred": classes[int(unique_labels)], "color_id": []}
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in rescaled_detection:
                n_x1 = max(0,int(np.array(x1)))
                n_x2 = max(0,int(np.array(x2)))
                n_y1 = max(0,int(np.array(y1)))
                n_y2 = max(0,int(np.array(y2)))
                # Only append bbox if it exists
                if n_x1 < n_x2 and n_y1 < n_y2:
                    bbox_coord = [n_x1,n_y1,n_x2, n_y2]
                    img_bboxcoord["bboxes"].append(bbox_coord)
                    img_bboxcoord["color_id"].append(color_id)
                    color_id += 1
        if not img_bboxcoord:
            continue
        # Append to results
        img_bboxcoord_pairs.append(img_bboxcoord)
    return img_bboxcoord_pairs

def extract_from_bbox(img_bbox_pairs, exg_thresh,
                    show_plots=False, save_plots=False, save_foregrounds=False, save_masks=False):
    
    timestr = time.strftime("%Y%m%d-%H%M%S")
    
    for i in img_bbox_pairs:
        imgp = i["image_path"]
        # Load image
        img = np.array(Image.open(imgp))
        img = cv2.imread(imgp)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        n_cls_preds = len(i["class_pred"])
        # Set new save prefix
        new_fname_prefix = os.path.splitext(os.path.basename(imgp))[0]
        # Bounding-box colors for plots
        cmap = plt.get_cmap("tab20b")
        colors = [cmap(i) for i in np.linspace(0, 1, 50)]
        
        ############# EXTRACT VEGETATION #############
        color_id_num = 0
        for bbox in i["bboxes"]: #x1, y1, x2, y2
            # Crop original image using bbox coordinates
            target_veg = rgb_img[bbox[1]:bbox[3],bbox[0]:bbox[2]] # y1 y2 x1 x2 for opencv
            box_w = bbox[2] - bbox[0]
            box_h = bbox[3] - bbox[1]    
            # Image histogram Equalization
            # target_veg = hist_equa(target_veg) # Performs better without
            # Get ExG Mask
            exg_mask = make_exg(target_veg) # TODO: 'make_exr' or 'make_ndi' options
            exg_mask = np.where(exg_mask < exg_thresh, 0, exg_mask).astype('uint8') # Thresholding removes low negative values
            # Create a CLAHE object (Arguments are optional).
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))            
            clahe_exg = clahe.apply(exg_mask)
            # Otsu's Thresholding
            ex_blur = cv2.GaussianBlur(clahe_exg, (5,5),0).astype('uint8')
            ret3,exg_th3 = cv2.threshold(ex_blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            # Get top N component
            top_mask = filter_by_concomponents(exg_th3, 1)
            # Crop image using mask to create RGBA foreground
            exg_frgd = create_foreground(target_veg, top_mask)

            ########### SHOWING AND SAVING PLOTS ############
            if save_plots or show_plots:
                
                
                
                bbox_colors = random.sample(colors, len(i['bboxes']))
                color = bbox_colors[color_id_num]


                # Create a Rectangle patch
                rect_bbox = patches.Rectangle((bbox[0], bbox[1]), box_w, box_h, linewidth=2,
                edgecolor=color, facecolor="none")
                # Subplots
                fig, axs = plt.subplots(2,3,figsize=(20,16))
                # Bbox on original image
                axs[0,0].imshow(rgb_img)
                axs[0,0].set_title("Detection",  fontsize=32)
                axs[0,0].add_patch(rect_bbox) # Add the bbox to the plot
                axs[0,0].text(bbox[0], bbox[1], s=i['class_pred'],color="white", verticalalignment="top", bbox={"color": color, "pad": 0},) # Label
                # Remaining plots
                axs[0,1].imshow(target_veg); axs[0,1].set_title("Detection Inset",  fontsize=32)
                axs[0,2].imshow(exg_mask); axs[0,2].set_title("ExG",  fontsize=32); axs[0,2].axis('off')
                axs[1,0].imshow(exg_th3);  axs[1,0].set_title("Otsu' Threshold",  fontsize=32); axs[1,0].axis('off')
                axs[1,1].imshow(top_mask);  axs[1,1].set_title("ExG Mask",  fontsize=32); axs[1,1].axis('off')                                
                axs[1,2].imshow(exg_frgd); axs[1,2].set_title("Foreground",  fontsize=32); axs[1,2].axis('off')
                # Save figure
                if save_plots:
                    figure_dir = f"output/figures/figures_{timestr}"
                    os.makedirs(figure_dir, exist_ok=True)
                    figure_path = os.path.join(figure_dir,"figure_" + new_fname_prefix + str(color_id_num) + ".jpg" )
                    plt.savefig(figure_path)
                if show_plots:
                    plt.show()
                plt.close()
            
            ########## SAVING FOREGROUNDS ############
            if save_foregrounds:
                frgd_dir = f"output/foregrounds/foregrounds_{timestr}"
                os.makedirs(frgd_dir, exist_ok=True)
                frgd_path = os.path.join(frgd_dir, new_fname_prefix + "_" + str(color_id_num) + ".png" )
                exg_frgd.save(frgd_path)
            
            color_id_num += 1
    
def xywh_2_xyxy(x):
    x = np.array(x)
    y = np.zeros_like(x)

    center_x = x[...,0]
    center_y = x[...,1]
    width = x[..., 2]
    height = x[..., 3]

    y[...,0] = np.maximum(0, center_x - width/2).astype(np.int)
    y[...,1] = np.maximum(0,center_y - height/2).astype(np.int)
    y[...,2] = np.maximum(0,center_x + width/2).astype(np.int)
    y[...,3] = np.maximum(0,center_y + height/2).astype(np.int)
    
    y = torch.from_numpy(y)
    return y


def rescale_boxes(boxes, current_dim, original_shape):
    """ Rescales bounding boxes to the original shape """
    orig_h, orig_w = original_shape
    # The amount of padding that was added
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))
    # Image height and width after padding is removed
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x
    # Rescale bounding boxes to dimension of original image
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
    return boxes

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """
    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    prediction[..., :4] = xywh_2_xyxy(prediction[..., :4])
    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Object confidence times class confidence
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]
        # Sort by it
        image_pred = image_pred[(-score).argsort()]
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
        detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)
        # Perform non-maximum suppression
        keep_boxes = []
        while detections.size(0):
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
            label_match = detections[0, -1] == detections[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            weights = detections[invalid, 4:5]
            # Merge overlapping bboxes by order of confidence
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)

    return output

def get_images(img_dir):
    img_list = glob.glob(img_dir + "/*.jpeg")

    return img_list