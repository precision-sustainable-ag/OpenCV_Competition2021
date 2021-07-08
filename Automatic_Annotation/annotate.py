import cv2
from utils import specify_species, make_exg, otsu_thresh, filter_by_component_size, create_foreground
import argparse
from pathlib import Path

from skimage.segmentation import clear_border
import skimage.morphology as morphology
from skimage.morphology import opening, white_tophat
from skimage.morphology import black_tophat, disk

import itertools

class AnnotateBenchBot:
    def __init__(self):
        self.species_map = {
                            0:"clover", 1:"sunflower", 2:"cowpea",
                            3:"goosefoot", 4:"horseweed", 5:"grass",
                            6:"velvetleaf", 7:"grasses"
                            }

    def validate_args(self, args):
        # Check if directories exists
        self.image_dir = args.image_dir
        assert Path(self.image_dir).is_dir(), "data_dir does not exist."
        self.save_dir = args.annotation_dir 
        if not Path(self.save_dir).is_dir():
            print("Save directory does not exist. Creating save direcory.")
            Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        
        self.images_dir = Path(self.save_dir) / 'images'
        self.masks_dir = Path(self.save_dir) / 'masks' 

        if not Path(self.images_dir).is_dir() or Path(self.masks_dir).is_dir():
            print("Image and mask directorie does not exist. Creating image and mask subdirectories.")
            Path(self.images_dir).mkdir(parents=True, exist_ok=True)
            Path(self.masks_dir).mkdir(parents=True, exist_ok=True)

        self.target_species = args.target_species

        self.clear_border = args.clear_border

        self.top_n = args.top_n

        self.min_object_size = int(args.min_object_size)
        self.area_threshold = int(args.area_threshold)
        self.target_species = args.target_species.lower()

        self.species_list = self.specify_species(Path(self.image_dir).rglob('*rgb.png'), self.target_species)
        

    def get_VIs(self, imgpath):
        self.orig_img = cv2.imread(str(imgpath))
        self.rgbimg = cv2.cvtColor(self.orig_img, cv2.COLOR_BGR2RGB)
        self.exg_mask = make_exg(self.rgbimg, exg_thresh=True)
        # self.exg_mask = median(self.exg_mask, disk(3))
        # Otsu's thresh
        self.otsu_mask = otsu_thresh(self.exg_mask)
        # return rgbimg, exg_mask, otsu_mask
        
    def specify_species(self, img_list, species):
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

    def get_imgs_by_species(self):
        self.species_list = specify_species(Path(self.image_dir).rglob('*rgb.png'), self.target_species)

    def annotate(self):
        
        
        for imgpath in sorted(self.species_list):
            print(imgpath)
            # Load image
            self.get_VIs(imgpath)    

                        
            if self.clear_border:
                self.final_mask = clear_border(self.otsu_mask)
            else:
                self.final_mask = self.otsu_mask

            # Filter components by top largest sizes
            mask_list = filter_by_component_size(self.final_mask.astype('uint8'),self.top_n)
            num = 0 
            # Apply morphological closing operations to individual masks
            for component in mask_list:

                mask = morphology.remove_small_holes(
                    morphology.remove_small_objects(
                        component.astype('uint8'), 
                        self.min_object_size),
                        self.area_threshold)
                
                # Close using white tophat
                w_disk = disk(4)
                # Close using black tophat
                b_disk = disk(2)
                # Tophat closing
                w_tophat = white_tophat(mask, w_disk)
                b_tophat = black_tophat(mask, b_disk)
                # Map results to mask
                mask[w_tophat==255] = 255
                mask[b_tophat==255] = 255

                # Otsu's thresh
                exg_mask = otsu_thresh(mask.astype('uint8'))
                open_disk = disk(1)
                exg_mask = opening(exg_mask, open_disk)

                # Save foreground using original file name
                mask_name = imgpath.stem
                frgd = create_foreground(self.rgbimg, exg_mask)
                if not Path(self.save_dir).is_dir():
                    Path(self.save_dir).mkdir(parents=True, exist_ok=True)

                image_path = str(self.images_dir)  + "/" + mask_name + "0" + str(num) + '.png'
                mask_path = str(self.masks_dir)  + "/" + mask_name + "0" + str(num) + '.png'

                frgd.save(image_path)
                cv2.imwrite(mask_path, exg_mask)  
                num +=1
    
    def main(self, args):
        self.validate_args(args)
        self.annotate()
        print('Image annotation completed.')

if __name__ == "__main__":
    import argparse
 
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='sample', help='"week" parent directory')
    parser.add_argument('--annotation_dir', type=str, default='output/annotation/sample', help='"week" annotation output directory')
    parser.add_argument('--top_n', type=int, default=3, help='top N largest components from image')
    parser.add_argument('--target_species', default='clover', type=str, help='Name target specie by bench row')
    parser.add_argument('--clear_border', action='store_false' ,help='Remove components on the border')
    parser.add_argument('--area_threshold', default = 100, help='The maximum area, in pixels, of a contiguous hole that will be filled in the mask')
    parser.add_argument('--min_object_size', default = 100, help='The smallest allowable mask component size')
    args = parser.parse_args()

    annotate = AnnotateBenchBot()
    annotate.main(args)
