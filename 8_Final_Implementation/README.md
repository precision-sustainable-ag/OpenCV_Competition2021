# Final Implementation

## Embedding Biomass Estimation Model

Install the following dependancies into RPi for the embedded biomass model and real-time plotting. 

> ### **dependencies**
>>
>> pip3 install joblib
>> 
>> pip3 install scikit-learn==0.24.1
>> 
>> pip3 install matplotlib
>> 
>> sudo apt-get install python3-gi-cairo

> ### **install the requirements**
>>
>> python3 install_requirements.py


In the folder /models/BiomassModel, there are three CSV files(clover, grass, broadleaf) containing DataFrames with several recorded biomass samples, along with calculated correlating "features" such as Excess of Green(ExG), Excess of Red(ExR), Depth (CAnopy Height Measurement), and Segmentation Pixel count.

Use PCA.py for creating the feature vector with RGB, Depth, and semantic segmentation image (inference image) captured/processed by the OAK-D Camera. PCA.py is also used to create 3 individual Random Forest Regression models for Clover, Broadleaf, and Grass plants named: random_forest_clover.joblib, random_forest_broadleaf.joblib, random_forest_grass.joblib.

## Complete pipeline

Use color_segmentation/py for weed species identification, and biomass estimation per specie. The general pipeline of this script is:

- RGB camera stream is created at 1080P resolution, center-cropped image with a size of 512x512 pixels
- Depth stream is created. Disparity map is visualized and NumpyArray is processed. 720P resolution at a size of 720x1280 pixels.
- Run the inference model over the RGB info. 
- Outputs class predictions for each pixel present in the RGB frame were colored in a new Window.
- The color segmentation mask showing the class predictions as well as overlayed over the original RGB stream. 
- Max/Min X/Y values were calculated over each class prediction from the segmentation mask, which we use to approximate a bounding box around the predicted class.
- That bounding box was used to create the feature vector to estimate the biomass.
- Biomass estimation results per specie were displayed into a graph.

![image](https://user-images.githubusercontent.com/70924969/125471205-2f4777a9-cd56-499f-b7a7-1d1df0f764f0.png)

Figure 1.  Screenshot of the Final implementation. Semantic segmentation labels: Red (Clover), Orange (Grass), Purple (Broadleaf) 

---

For more details please visit our [wiki](https://github.com/precision-sustainable-ag/OpenCV_Competition2021/wiki/8.-Final-Implementation) page

