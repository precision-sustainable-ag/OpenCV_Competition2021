# Synthetic Bench Images

## Instructions:

1. After cloning the OpenCV_Competition2021 repo, `cd` into the Synthetic Bench Images folder. 
2. In the command line, run the 'runpot.sh' file by type `bash runpot.sh`

An "output" folder should be created in your working directory. An "images" and "masks" folder should be created and be populated with synthetic bench images and masks, respectively.

## pot_genertor

These arguments can be adjusted to account for some differences in features among the various species classes. 
```
    --bench_dir, type=str, required=True, dest="bench_dir", default="bench", help="Location of empty bench images.
    --pot_dir, type=str, required=True, dest="pot_dir", default="pots", help="Location of empty pot directory.
    --annotation_dir, type=str, required=True, dest="annotation_dir", default="annotations", help="Location of annotation plants.
    --save_dir, type=str, required=True, dest="save_dir", default="output", help="Location to save results.
    --mode, type=str, dest="mode", default="random", help="NOT FUNCTIONAL 'random', 'by_week', 'by_commonname'
    --count, type=int, dest="count", default=5, help="Number of images to create."
```
---

The BenchBotics semi-automatic annotation was designed to meet the data needs of training a convolutional neural network. The annotation system used images collected by the Oak-D sensor attached to our custom Bench Bot platform. 

---

For more details visit our [wiki](https://github.com/precision-sustainable-ag/OpenCV_Competition2021/wiki/4.-Synthetic-Bench-Images) page 