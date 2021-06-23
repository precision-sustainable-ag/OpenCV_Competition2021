# Semi-automatic annotation system

The BenchBotics semi-automatic annotation was designed to meet the data needs of training a convolutional neural network. The annotation system used images collected by the Oak-D sensor attached to our custom Bench Bot platform. 

## Why
A species-specific foreground dataset was created to meet the synthetic dataset generation needs of our project. The extraction process was designed by keeping in mind the unique requirements of training a convolutional neural network that could identify and segment three broad categories of agriculturally relevant plants. Because collecting real-world agricultural image data is so difficult, our approach involved labeling plant data using controlled greenhouse conditions. We created a semi-automatic pipeline that extracted target vegetation from our BenchBot images and created foregrounds of vegetation. Our library of species foregrounds could then be applied to generating a synthetic dataset for training our model. This section outlines the semi-automatic vegetation extraction pipeline used to develop that library.

## What
Plant foregrounds were extracted from BenchBot images by creating a binary mask of the target vegetation. Vegetation indices and a combination of thresholding or unsupervised classification were used to create masks. Noise generated during mask generation was then removed using morphological operations. Vegetation foregrounds were created by overlaying the resulting mask and extracting only the target plant data from the original color images. Plant foregrounds were then manually inspected and sorted by species. 

![alt text](assets/benchbot.jpg?=raw)

Use in colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/precision-sustainable-ag/OpenCV_Competition2021/blob/master/Automatic_Annotation/automatic_annotate.ipynb)

![alt text](assets/detect.png?=raw)


![alt text](assets/figure.jpg?=raw)
