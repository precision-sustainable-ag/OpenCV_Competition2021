# Semi-automatic annotation system

## Instructions:

1. After cloning the OpenCV_Competition2021 repo, `cd` into the Automatic_Annotation folder. 
2. In the command line, run the 'run.sh' file by type `bash run.sh`

### Preprocessing
Color images from BecnhBot platform were captured and stored by capture week, row, and camera stop. For each stop, 30-80 images were captured, many of which were blurry as camera was the automatic focus was being adjusted. Simple preprocessing was performed to remove duplicates and blurry images by using laplacian variance of each image. If images where below a given threshold (considered too blurry) they were not used for annotation. Instead, the first instance of an image above the given threshold was choosen for annotation and moved to a seperate directory for inspection. 

To illustrate the preprocessing, we have provided a sample of the BenchBot output 'week4_sample'. The 'filter_by_focus.py' script will perform this preprocessing and output a 'focus_output' folder with a single instance (instead of 30-80) for each stop location. 


After preprocessing, the 'annotate.py' script will create an "annotate_output" folder in your working directory. An "images" folder of foregrounds for creating synthetic dataset and a "masks" folder with binary masks for each extracted foreground will be populated with results. Masks have been generated as ancillary data which can be helpful for inspecting results and for experimental purposes. Masks are not needed for generating synthetic data. Mask file names remain the same except for a numbered suffix which has been added to represents the N component extracted from the original image.

Lastly, manual inspection, selection, and organization of foregrounds must be performed by the user to divide the results into the 7 species classes: 

1. clover
2. cowpea
3. grass
4. horsetail
5. sunflower
6. goosefoot
7. velvetleaf

For each week, this process took less than 10 minutes.

## Annotation Arguments

These arguments can be adjusted to account for some differences in features among the various species classes. 
```
    --image_dir', type=str, default='sample', help='location of images to generate foregrounds')
    --annotation_dir', type=str, default='output/annotation/sample', help='foreground output directory')
    --top_n', type=int, default=3, help='top N largest components from image')
    --target_species', default='all', type=str, help='Name target species from 7 classes')
    --clear_border', action='store_false' ,help='Remove components on the border')
    --area_threshold', default = 100, help='The maximum area, in pixels, of a contiguous hole that will be filled in the mask')
    --min_object_size', default = 100, help='The smallest allowable mask component size')
```
---

The BenchBotics semi-automatic annotation was designed to meet the data needs of training a convolutional neural network. The annotation system used images collected by the Oak-D sensor attached to our custom Bench Bot platform. 

---

For more details visit our [wiki](https://github.com/precision-sustainable-ag/OpenCV_Competition2021/wiki/3.-Annotation) page 
