# Synthetic Data Creation

The BenchBotics ..

## Why
A species-specific foreground dataset was created to meet the synthetic dataset generation needs of our project. The extraction process was designed by keeping in mind the unique requirements of training a convolutional neural network that could identify and segment three broad categories of agriculturally relevant plants. Because collecting real-world agricultural image data is so difficult, our approach involved labeling plant data using controlled greenhouse conditions. We created a semi-automatic pipeline that extracted target vegetation from our BenchBot images and created foregrounds of vegetation. Our library of species foregrounds could then be applied to generating a synthetic dataset for training our model. This section outlines the semi-automatic vegetation extraction pipeline used to develop that library.


