# Synthetic Bench Images

## Instructions:

1. After cloning the OpenCV_Competition2021 repo, `cd` into the Synthetic Bench Images folder. 
2. In the command line, run the 'runpot.sh' file by type `bash runpot.sh`

An "output" folder should be created in your working directory. An "images" and "masks" folder should be created and be populated with synthetic bench images and masks, respectively.

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

---

For more details visit our [wiki](https://github.com/precision-sustainable-ag/OpenCV_Competition2021/wiki/4.-Synthetic-Bench-Images) page 
