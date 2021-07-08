# Semi-automatic annotation system

## Instructions:

1. After cloning the OpenCV_Competition2021 repo, `cd` into the Automatic_Annotation folder. 
2. In the command line, run the 'run.sh' file by type `bash run.sh`

An "output" folder should be created in your working directory. An "images" folder of foregrounds for creating synthetic dataset and a "masks" folder with binary masks for each extracted foreground will be populated with results. Masks have been generated as ancillary data which can be helpful for inspecting results and for experimental purposes. Masks are not needed for generating synthetic data. Mask file names remain the same except for a numbered suffix which has been added to represents the N component extracted from the original image.

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

## Why
A species-specific foreground dataset was created to meet the synthetic dataset generation needs of our project. The extraction process was designed by keeping in mind the unique requirements of training a convolutional neural network that could identify and segment three broad categories of agriculturally relevant plants. Because collecting real-world agricultural image data is so difficult, our approach involved labeling plant data using controlled greenhouse conditions. We created a semi-automatic pipeline that extracted target vegetation from our BenchBot images and created foregrounds of vegetation. Our library of species foregrounds could then be applied to generating a synthetic dataset for training our model. This section outlines the semi-automatic vegetation extraction pipeline used to develop that library.

![alt text](assets/benchbot.jpg?=raw)

## What
Plant foregrounds were extracted from BenchBot images by creating a binary mask of the target vegetation. Vegetation indices and a combination of thresholding or unsupervised classification were used to create masks. Noise generated during mask generation was then removed using morphological operations. Vegetation foregrounds were created by overlaying the resulting mask and extracting only the target plant data from the original color images. Plant foregrounds were then manually inspected and sorted by species. 

## How
Vegetation Indices: First, a vegetation index ¬– a simple operations of image channels designed to emphasize certain plant properties – was created using the original color image. We experimented with four color-based vegetation indices; Excess green index (ExG), Excess Red (ExR), Normalized difference index (NDI), and ExG minus ExR (ExG-ExR). Of the four, we found that ExG was the most helpful in extracting relevant vegetation information for our unique artificial setting. Thresholding was performed on the resulting single channel ExG image by changing negative values to zero.

#### Mask generation
Manual intervention was needed to identify the proper mask generation technique which was determined by factors including species, growth stage, and lighting conditions. In some cases, a simple Otsu’s thresholding of the ExG index could be applied to generate a binary mask of plant vegetation. For other conditions, however, K-means clustering, chosen for its simplicity and efficiency, performed better in capturing the relevant plant foreground. 

#### Morphological Operations
Morphological operations played an important role in removing unnecessary components and denoising in the masks. These operations varied in sophistication depending on species, growth stage, and lighting, and thus the technique that was used.  For Otsu’s thresholding, morphological operations were used to identify the top 5 largest connected components which were then cropped and denoised. Masks generated using K-means clustering used morphological closing operations to eliminate noise. For images of plants during the early growth stage, mask components that were connected to the border were removed. This was to eliminate unnaturally straight plants edges that could occur if vegetation was not entirely within the image frame. During later growth stages and for some species (i.e. grasses), however, most vegetation extended beyond the image frame and unnatural edges had to be retained. 

![alt text](assets/opencv_2021.jpg?=raw)

![alt text](assets/opencv_2021_reducing_nosie.jpg?=raw)

#### Foreground extraction and inspection
Vegetation was extracted using the resulting denoised vegetation mask. Plant foregrounds that were too small, visually indistinguishable, or that appeared to contain more than one species were manually removed. The remaining foregrounds were then sorted by species.

![alt text](assets/opencv_2021_foregrounds.jpg?=raw)


## When
Images were collected once for 6 weeks; the first two weeks were used for fine-tunning and calibration of BenchBot and Oak-D sensor platform, and the remaining weeks were used for the resulting foreground generation process. Each week, 100-200 images were be processed for foreground extraction and manual sorting.

## Where
All code was written in python. Vegetation indices were generated using Numpy, masks were created using a combination of SciKit Learn, OpenCV, and SciKit Image libraries. Images were collected in Maryland and images were organized and annotations were created in Texas.

---
