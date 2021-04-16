import numpy as np 
import pandas as pd 
from glob import glob 
import os 
import argparse 
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("-ad", "--annotationsdir", default='/home/pi/OpenCVData/annotations/week2/', help="Directory with annotations stored")
parser.add_argument("-dd","--datadir", default='/home/pi/OpenCVData/week2', help="Directory with data")
args = parser.parse_args()

annot_dir = args.annotationsdir
data_dir = args.datadir 

rgb_paths = []
depth_paths = []
annots = []
for filename in glob(annot_dir+'*.txt'):
    annot_filename = filename
    with open(annot_filename) as f: 
        lines = f.readlines()[0].strip().split(' ')
        lines = list(map(int, lines))
    annots.append(lines)
    rgb_folder_path = '_'.join(annot_filename.split('/')[-1].split('_')[:-2])
    rgb_image_num = annot_filename.split('/')[-1].split('_')[-2:][0]+'.png'
    depth_image_num = annot_filename.split('/')[-1].split('_')[-2:][0].replace('rgb', 'depth')+'.npy'
    # print(rgb_folder_path, rgb_image_num)
    rgb_paths.append(rgb_folder_path+'/'+rgb_image_num)
    depth_paths.append(rgb_folder_path+'/'+depth_image_num)
    # print(glob('_'.join(rgb_folder_path.split('_')[:-2])+'******/'+rgb_image_num))
    # print(glob('_'.join(path[0].split('_')[:-2])+'??????'+path[1]))

print(rgb_paths)
print(annots)
spreadsheet = pd.DataFrame(
    {'rgb_paths':rgb_paths,
     'annots':annots
    })
spreadsheet.to_csv('annotated_plants.csv')