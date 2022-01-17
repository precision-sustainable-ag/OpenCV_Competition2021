# Semi-automatic annotation pipeline

Collecting and labeling images of weeds is time-consuming and limiting resulting in severely slowed development of datasets necessary for utilizing artificial intelligence in agriculture, particularly in weed identification, mapping and precision management applications. This project aims to reduce the burden of manually annotating images of weeds and other plants. We do this by: 
1. Developing a library of vegetation cut-outs
2. Generating a dataset of artificial images (using vegetation cut-outs) to train a deep learning object detection model, and 
3. Using detection results and simple image processing techniques to extract and classify vegetation.

No images need to be manually annotated only manually sorted into distinct classes, saving an estimated 22 hours per 1,000 cut-outs. Automatic annotation pipelines play an important role in developing robust datasets for trained AI models that can handle diverse scenes. The methodology devised here will be utilized in a weed image library pipeline currently being developed.

## Workflow

![](https://github.com/precision-sustainable-ag/OpenCV_Competition2021/blob/annotation_pipeline/4_Synthetic_Bench_Images/assets/workflow_v2_simplified_small.png)

## Instructions:

1. After cloning the OpenCV_Competition2021 repo, `cd` into the Synthetic Bench Images folder. 
2. In the command line, run the 'runpot.sh' file by type `bash runpot.sh`

An "output" folder should be created in your working directory. An "images" and "masks" folder should be created and be populated with synthetic bench images and masks, respectively.
Additionally, bounding box labels are provided for each plant and their classe name.

## pot_generator

### Arguments

```
    --bench_dir, type=str, required=True, dest="bench_dir", default="bench", help="Location of empty bench images.
    --pot_dir, type=str, required=True, dest="pot_dir", default="pots", help="Location of empty pot directory.
    --annotation_dir, type=str, required=True, dest="annotation_dir", default="annotations", help="Location of annotation plants.
    --save_dir, type=str, required=True, dest="save_dir", default="output", help="Location to save results.
    --count, type=int, dest="count", default=5, help="Number of images to create."
```
---

Synthetic BenchBot images were created to diversify trainset and further test trained model. Images were collected by the Oak-D sensor attached to our custom Bench Bot platform. 

For more details visit our [wiki](https://github.com/precision-sustainable-ag/OpenCV_Competition2021/wiki/4.-Synthetic-Bench-Images) page 
