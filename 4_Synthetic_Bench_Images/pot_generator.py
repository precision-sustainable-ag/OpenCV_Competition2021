"""
Written by Matthew Kutugata (2021)
Contact: mattkutugata@tamu.edu
"""

import argparse
from pathlib import Path
import warnings
from tqdm import tqdm
import random
import numpy as np
import cv2
from utils import overlay

"""
Takes in images of bench backgrounds, pots, and annotations to create synthetic bench data.
Ouput includes RGB image and vegetation mask in seperate directories. 

Expected folder strucutre:
|-- pot
    |-- <filename>.png
|-- bench
    |-- <filename>.png
|-- annotations
    |-- week<num>
        |-- clover
            |-- wk<num>_row<num>_stop<num>_<timestamp>_frgd.png
|-- save_directory

A "images" and "masks" directory will be automatically generated insided your save directory.
"""

class BenchDataset:
    # Define properties that all BenchDataset objects must have
    def __init__(self):
        self.zero_padding = 6
        self.pot_alignment = [9, 6, 3]
        self.commonnames   = ["clover", "cowpea", "goosefoot", "grasses", "horseweed", "sunflower", "velvetleaf"]

    def _validate_args(self, args):
        # Validates input arguments and sets up class variables        
        # Validate the count
        assert args.count > 0, 'count must be greater than 0'
        self.count = args.count
        # Validate args
        self.mode = args.mode   
        self._validate_input_directory()
    
    def _validate_input_directory(self):
        # Validate input directories
        self.bench_dir = Path(args.bench_dir)
        assert self.bench_dir.exists(), f'bench directory does not exist: {args.bench_dir}'
        self.pot_dir = Path(args.pot_dir)
        assert self.pot_dir.exists(), f'pot directory does not exist: {args.pot_dir}'
        self.save_dir = Path(args.save_dir)
        if not self.save_dir.exists():
            print(self.save_dir.exists(), f'save directory does not exist, creating save directory: {args.save_dir}')
            self.save_dir.mkdir(parents=True, exist_ok=True)
        # Create list of week directories
        self.annotation_dir = Path(args.annotation_dir)
        assert self.annotation_dir.exists(), f'annotation directory does not exist: {args.pot_dir}'
        
        # Setup data dictionaries
        self._create_foreground_dict()
        self._create_bench_list()
        self._create_pot_list()

    def _create_foreground_dict(self):
        # Validates input foregrounds and processes them into a foregrounds dictionary.
        self.foregrounds_dict = dict()
        self.weeks = []
        self.commonnames = []
        if str(self.annotation_dir.parts[-1]).startswith("annotations"):
            for week_dir in self.annotation_dir.iterdir():
                self.weeks.append(week_dir.name)
                if not week_dir.is_dir():
                    warnings.warn(f'file found in week directory (expected common name directories), ignoring: {week_dir}')
                    continue
                # This is a super category directory
                for commonname_dir in sorted(Path(week_dir).iterdir()):
                    self.commonnames.append(commonname_dir.name)
                    if not commonname_dir.is_dir():
                        warnings.warn(f'file found in common name directory (expected category directories), ignoring: {commonname_dir}')
                        continue
                    # This is a category directory
                    for image_file in sorted(Path(commonname_dir).iterdir()):
                        if not image_file.is_file():
                            warnings.warn(f'a directory was found inside a common name directory, ignoring: {str(image_file)}')
                            continue
                        if image_file.suffix != '.png':
                            warnings.warn(f'foreground must be a .png file, skipping: {str(image_file)}')
                            continue
                        # Valid foreground image, add to foregrounds_dict
                        super_category = week_dir.name
                        category = commonname_dir.name
                        if super_category not in self.foregrounds_dict:
                            self.foregrounds_dict[super_category] = dict()
                        if category not in self.foregrounds_dict[super_category]:
                            self.foregrounds_dict[super_category][category] = []
                        self.foregrounds_dict[super_category][category].append(image_file)
        if str(self.annotation_dir.parts[-1]).startswith("week"):
            self.week = str(self.annotation_dir.parts[-1])

            for commonname_dir in self.annotation_dir.iterdir():
                self.commonnames.append(commonname_dir.name)
                if not commonname_dir.is_dir():
                    warnings.warn(f'file found in common name directory (expected category directories), ignoring: {commonname_dir}')
                    continue

                    for image_file in sorted(Path(commonname_dir).iterdir()):
                    if not image_file.is_file():
                        warnings.warn(f'a directory was found inside a common name directory, ignoring: {str(image_file)}')
                        continue
                    if image_file.suffix != '.png':
                        warnings.warn(f'foreground must be a .png file, skipping: {str(image_file)}')
                        continue
                    # Valid foreground image, add to foregrounds_dict
                    super_category = self.annotation_dir.name
                    category = commonname_dir.name
                    if super_category not in self.foregrounds_dict:
                        self.foregrounds_dict[super_category] = dict()
                    if category not in self.foregrounds_dict[super_category]:
                        self.foregrounds_dict[super_category][category] = []
                    self.foregrounds_dict[super_category][category].append(image_file)

            
            assert len(self.foregrounds_dict) > 0, 'no valid foregrounds were found'
            self._sort_weeks_and_commonnames()
    
    def _create_pot_list(self):
        self.pot_images = []
        for image_file in self.pot_dir.iterdir():
            if image_file.suffix != '.png':
                continue
            self.pot_images.append(image_file)
        assert len(self.pot_images) > 0, "No valid pot images were found"

    def _create_bench_list(self):
        self.bench_images = []
        for image_file in self.bench_dir.iterdir():
            if image_file.suffix != '.png':
                continue
            self.bench_images.append(image_file)
        self.bench_images = sorted(self.bench_images)
        assert len(self.bench_images) > 0, "No valid bench images were found"
    
    def _sort_weeks_and_commonnames(self):
        self.commonnames = sorted(list(set(self.commonnames)))
        self.weeks = sorted(self.weeks)


    def generate_bench_images(self, mode='random'): # 'random', 'by_week', 'by_commonname', or 'choas'
        plant_paths = sorted(Path(self.annotation_dir).rglob('*.png'))
        plant_paths = [str(i) for i in plant_paths]
        # print(plant_paths)
        image_num = 0
        for i in tqdm(range(self.count)):            
            # Get (lists) of image paths
            bench_path = random.choice(self.bench_images)    
            
            pot_alignment = random.choice(self.pot_alignment)
            pot_paths = np.random.choice(self.pot_images, pot_alignment, replace=False)

            # Check for mode
            # Not working
            if self.mode == 'random':
                # Get list of pot paths
                rdmn_plant_paths = np.random.choice(plant_paths, pot_alignment, replace=True)
            if self.mode == 'by_week':
                pass

            background, mask = overlay(
                                bench_path, 
                                pot_paths, 
                                rdmn_plant_paths,
                                pot_alignment, 
                                )


            save_imagedir = Path(self.save_dir,'images')
            save_maskdir = Path(self.save_dir,'masks')
            # Create file dir if needed for images and masks
            if not save_imagedir.exists():
                print(f'save image directory does not exist, creating directory: {save_imagedir}')
                save_imagedir.mkdir(parents=True, exist_ok=True)
            
            if not save_maskdir.exists():
                print(f'save mask directory does not exist, creating directory: {save_maskdir}')
                save_maskdir.mkdir(parents=True, exist_ok=True)
            
            # Create the file stem (used for both composite and mask)
            file_stem = f'{image_num:0{self.zero_padding}}'
            if self.week:
                save_imagepath = Path(save_imagedir,f'{self.week}_' + file_stem + '.png')
                save_maskpath = Path(save_maskdir,f'{self.week}_' + file_stem + '.png')                
            else:
                save_imagepath = Path(save_imagedir, file_stem + '.png')
                save_maskpath = Path(save_maskdir,file_stem + '.png')                
            # Save results
            cv2.imwrite(str(save_imagepath), background)
            cv2.imwrite(str(save_maskpath), mask)
            image_num += 1

    def main(self, args):
        self._validate_args(args)
        self.generate_bench_images()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--bench_dir", type=str, required=True, dest="bench_dir", default="bench", help="Location of empty bench images.")
    parser.add_argument("--pot_dir", type=str, required=True, dest="pot_dir", default="pots", help="Location of empty pot directory.")
    parser.add_argument("--annotation_dir", type=str, required=True, dest="annotation_dir", default="annotations", help="Location of annotation plants.")
    parser.add_argument("--save_dir", type=str, required=True, dest="save_dir", default="output", help="Location to save results.")
    parser.add_argument("--mode", type=str, dest="mode", default="random", help="NOT FUNCTIONAL 'random', 'by_week', 'by_commonname'")
    parser.add_argument("--count", type=int, dest="count", default=5, help="Number of images to create.")
    args = parser.parse_args()

    ben = BenchDataset()
    ben = ben.main(args)
