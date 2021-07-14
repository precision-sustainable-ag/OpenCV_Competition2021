# Final Implementation

## Embedding Biomass Estimation Model

Install the following dependancies into RPi for the embedded biomass model and real-time plotting. 

*pip3 install joblib
*pip3 install scikit-learn==0.24.1
*pip3 install matplotlib
*sudo apt-get install python3-gi-cairo
Run the following to install the requirements required for the OAK-D Camera: 
*python3 install_requirements.py 

In the folder /models/BiomassModel, there are three CSV files(clover, grass, broadleaf) containing DataFrames with several recorded biomass samples, along with calculated correlating "features" such as Excess of Green(ExG), Excess of Red(ExR), Depth (CAnopy Height Measurement), and Segmentation Pixel count.

Use PCA.py for creating the feature vector with RGB, Depth, and semantic segmentation image (inference image) captured/processed by the OAK-D Camera. PCA.py is also used to create 3 individual Random Forest Regression models for Clover, Broadleaf, and Grass plants named: random_forest_clover.joblib, random_forest_broadleaf.joblib, random_forest_grass.joblib.

##Final Implementation

To run the final implementatio script it is neccesary to create a proper pipeline into the dephtAI framework. The RGB image is resized and center-cropped with final size of 512x512.

To use these models to create real-time biomass predictions it is necessary to input a feature vector which is calculated in the exact same way as the feature vector which was used during training. That means using the exact same sized RGB images, and correctly rectifying the Depth preview to correlate with the RGB images. When the program, color_segmentation.py, is ran(python3 color_segmentation.py), 4 separate windows are displayed including a RGB and Depth image of the greenhouse plant row, a segmentation mask showing the neural network's class predicitons pixel by pixel, and a real-time plot of the 3 biomass predictions from the corresponding random forest models, as shown below. 
![image](https://user-images.githubusercontent.com/70924969/125471205-2f4777a9-cd56-499f-b7a7-1d1df0f764f0.png)

  The pipeline for the OAK-D camera is built as follows: the RGB camera stream is created at 1080P resolution with a size of 512x512 pixels, and a depth stream is created which shows the disparity map between the two OAK-D stereo cameras at 720P resolution at a size of 720x1280 pixels. Next, the RGB stream is used as the input to the Deep Learning model, 4_class_model_mobilenet_v3_large_data4_combined_class_weights_512x512_without_softmax.blob, which outputs class predictions for each pixel present in the RGB frame. The colored segmentation mask showing the class predictions is displayed, as well as overlayed over the original RGB stream. We calculate the max/min x/y values of each class prediction from the segmentation mask, which we us to approximate a bounding box around the predicted class. 
  Next, for each class we create a feature vector from the portion of the depth and RGB streams corresponding to the class bounding box. The feature vector contains the following values calculated from the bounded RGB stream: ExG, Mean of ExG, Standard Deviation of ExG, ExR, Mean of ExR, Standard Deviation of ExR, and the amount of pixels corresponding to the class found in the segmentation mask. From the depth stream, the following values are calculated and included in the feature vector: the sum of the first 20 depth levels in the area of a classes bounding box. Before summing the depth values over the nearest 20 depth values, it was necessary to shift the bounding box Y values up 60 pixels, and the X values 115 pixels. This correction is done to rectify the depth stream, ensuring that the area corresponding to the neural network class prediction is accurately mapped to the depth stream. Note, that this is necessary because the depth stream is displayed at a different resolution and size than the RGB stream. 

  These feature vectors for each class are then used as inputs to the corresponding Random Forest models. Next, the 3 Random Forest biomass predicitons are plotted on 3 separate subplots in real-time along a shifting x-axis. As the BenchBot moves along the greenhouse rows overtop clover, grasses, and broadleaf species, changes to the biomass predictions can be observed as the neural network's segmentation mask generates new predictions for each camera frame in real-time. 
##

