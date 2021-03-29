import argparse
from veg_utils import *
import random
import cv2
from tqdm import tqdm

"""
Creates PNG foreground images by extracting vegation using extended green 
vegetation index and contrast limiting adaptive histogram equalization. User must define 
the number of largest components (top_n) to extract from image set. 
Individual components will be saved us'ing original image name appended 
with number to represent n component. If object detection can is applied to
localize vegetation, top_n will be ignored.  


Written by: Matthew Kutugata
Last updated: March 28th, 2021
"""

parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', type=str, required=True, help='image directory')
parser.add_argument('--save_foreground_dir', type=str, default='output/foregrounds', help='directory to save foregrounds')  # file/folder, 0 for webcam
parser.add_argument('--top_n', type=int, default=3, help='inference size (pixels)')
parser.add_argument('--exg_thresh', type=float, default=0, help='object confidence threshold')
parser.add_argument('--testing', action='store_true', help='view output of sample foregrounds and original image')
parser.add_argument('--num_test_imgs', type=int, default=3, help='number of testing images')
parser.add_argument('--use_detection', action='store_true', help='use object detection to localize target vegetation, top_n will be ignored')

opt = parser.parse_args()

# Returns list of images of multiple files types
imgs = get_images(opt.image_dir)

# For testing and viewing examples
if opt.testing:
    imgs = random.sample(imgs, opt.num_test_imgs)
else:
    imgs = sorted(imgs, reverse=True)
frgd_id = 0
for imgp in tqdm(imgs):
    
    # Read in image
    img_bgr = cv2.imread(imgp)
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    if opt.use_detection:
        # Get detection labels (bboxes) for individual image
        label_path = vegimg2label_path(imgp)
        if label_path:
            # Extract list of bbox coordinates in xyxy
            bboxes = extract_boxes(label_path, img.shape)
            bboxes = xywh_2_xyxy(bboxes, img.shape)
            detection_id = 0
            for box in bboxes:
                box = [int(item) for item in box]
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                if box_area(box) < 5000:
                    continue
                detection_img = img[y1:y2, x1:x2]
                if detection_img.shape[0] == 0 or detection_img.shape[1] == 0:
                    continue
                # Create ExG mask
                exg_mask = make_exg(detection_img, exg_thresh=True)
                # Contrast limiting adaptive histogram equalization (CLAHE)
                equ_exg_mask = CLAHE_hist(exg_mask)
                # Otsu's thresh
                exg_th3 = otsu_thresh(equ_exg_mask)
                # Extract and save foregrounds
                exg_frgd = create_foreground(detection_img, exg_th3, add_padding=True)     
                if not os.path.exists(opt.save_foreground_dir):
                    os.makedirs(opt.save_foreground_dir)
                fname_prefix = os.path.splitext(os.path.basename(imgp))[0]
                frgd_path = os.path.join(opt.save_foreground_dir, fname_prefix + "_" + "detection_" + str(detection_id) + ".png" )
                exg_frgd.save(frgd_path)
                detection_id += 1

    else:
        # Create ExG mask
        exg_mask = make_exg(img, exg_thresh=True)
        # Contrast limiting adaptive histogram equalization (CLAHE)
        equ_exg_mask = CLAHE_hist(exg_mask)
        # Otsu's thresh
        exg_th3 = otsu_thresh(equ_exg_mask)
        # Get list of top N components
        top_masks = filter_by_componenet_size(exg_th3,opt.top_n)
        # Extract and save foregrounds
        exg_frgd = extract_save_ind_frgrd(img,top_masks, imgp,opt.save_foreground_dir,testing=opt.testing)

        plt.figure(figsize=(28, 20))
        # plt.suptitle("ExG Mask", fontsize=48)
        plt.subplot(231); plt.imshow(img);plt.title('Original Image', fontsize=38)
        plt.subplot(232); plt.imshow(exg_mask); plt.title("ExG Mask",fontsize=38)
        plt.subplot(233); plt.imshow(equ_exg_mask);plt.title('CLAHE Equalized ExG', fontsize=38)
        plt.subplot(234); plt.imshow(exg_th3); plt.title("Otsu Thresholding",fontsize=38)
        plt.subplot(235); plt.imshow(top_masks[4]);plt.title('1 of n top masks', fontsize=38)
        plt.subplot(236); plt.imshow(exg_frgd);plt.title('1 of n foregrounds', fontsize=38); plt.axis('off')

        plt.tight_layout()

        plt.savefig(opt.save_foreground_dir + "/" +str(frgd_id) + "Figure.png")
        plt.close()
        frgd_id +=1 