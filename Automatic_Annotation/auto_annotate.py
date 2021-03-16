from veg_utils import *
import random
import numpy as np
import os


img_dir = "/home/admin_mkutugata/Desktop/opencv_data_local/raw/06_data_2"

save_frgd_dir = "/home/admin_mkutugata/Desktop/opencv_data_local/annotate_output/foregrounds/06_data_2/"

imgs = get_images(img_dir)
imgs = sorted(imgs, reverse=True)

# For testing
# num_test_imgs = 3
# imgs = random.sample(imgs, num_test_imgs)

exg_thresh = 0

save_frgds = True

frgd_id = 0
for imgp in imgs:
    # Read in image
    img = cv2.imread(imgp)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Create ExG mask
    exg_mask = make_exg(img) # TODO: 'make_exr' or 'make_ndi' options
    exg_mask = np.where(exg_mask < exg_thresh, 0, exg_mask).astype('uint8') # Thresholding removes low negative values
    
    # Create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))            
    clahe_exg = clahe.apply(exg_mask)
    
    # Otsu's Thresholding
    ex_blur = cv2.GaussianBlur(clahe_exg, (5,5),0).astype('uint8')
    ret3,exg_th3 = cv2.threshold(ex_blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    # Get top N component
    top_mask = filter_by_concomponents(exg_th3, 2)
    
    # Crop image using mask to create RGBA foreground (optional: 3 px padding)
    exg_frgd = create_foreground(img, top_mask, add_padding=True)

    if save_frgds:

        if not os.path.exists(save_frgd_dir):
            os.makedirs(save_frgd_dir)

        fname_prefix = os.path.splitext(os.path.basename(imgp))[0]

        frgd_path = os.path.join(save_frgd_dir, fname_prefix + "_" + str(frgd_id) + ".png" )
        exg_frgd.save(frgd_path)
        frgd_id += 1





