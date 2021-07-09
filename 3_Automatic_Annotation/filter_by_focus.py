# from preprocess_utils import move_images
import argparse
from pathlib import Path
import cv2
import shutil
from tqdm import tqdm

def lap_var(img_path):
    """ Returns laplacian variance of image
    to check if image is blurry."""
    img = cv2.imread(str(img_path))
    lap_var = cv2.Laplacian(img, cv2.CV_64F).var()
    return lap_var

def chk_args(src, dst):
    src = Path(src)
    assert src.is_dir(), "Source directory does not exist. Check your path."
    dst = Path(dst)
    if not dst.is_dir():
        print("Destination directory does not exist. Creating destination directory.")
        dst.mkdir(parents=True, exist_ok=True)
    

def move_images(opt,no_rows = False):
    """ Move only focused images from source 'week' directory
    and move them to a new destination directory. If images are
    not organized by row or stop, uses unique time stamp to label
    images, else uses source 'row' and 'stop' subdirectories to 
    rename files as such. Excludes duplicate images for each stop"""
    src = opt.image_dir
    dst = opt.dest_dir
    lpthresh = opt.lap_thresh
    
    chk_args(src, dst)

    week = src.split("/")[-1]
    wk = 'wk' + week[-1]
    if no_rows:
        glob_sym = '*/**'
    else: 
        glob_sym = '*/**'

    row_id = 0
    stop_id = 0
    paths = list(Path(src).joinpath().glob(glob_sym))
    for row in tqdm(paths):
        row_id += 1
        rglob_dir = 'color.png' if wk == 'wk3' else 'rgb*.png'
        for img in Path(row).rglob(rglob_dir):
            stop_id += 1
            img_lpvar = lap_var(img)
            if img_lpvar > lpthresh: # Threshold value
                if no_rows:
                    rown = img.parts[-3]
                    date_time = "".join(img.parts[-2].split("_")[:3]) + "_" + "".join(img.parts[-2].split("_")[-3:])
                    new_fname = f"{wk}_" + rown + "_" + date_time + '_rgb.png'
                else:                    
                    row_stop = "row" + "_".join(list(img.parts)[-4:-2][0:2])
                    date_time = "".join(img.parts[-2].split("_")[:3]) + "_" + "".join(img.parts[-2].split("_")[-3:])
                    new_fname = f"{wk}_" + row_stop + "_" + date_time + '_rgb.png'
                    
                dst_path = Path(dst) / new_fname
                shutil.copy(str(img),str(dst_path))
                break

if __name__ == "__main__":
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True, help='"week" parent directory')
    parser.add_argument('--dest_dir', type=str, required=True, help='Where images will be saved.')
    parser.add_argument('--lap_thresh', type=int, default=150, help='Variance threshold (should be atleast 150).')
    opt = parser.parse_args()

    # Move images filter by focus quality
    move_images(opt, no_rows = False)