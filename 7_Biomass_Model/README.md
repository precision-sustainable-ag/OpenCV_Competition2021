# Biomass Model

To run the notebooks please use Jupyter Notebooks and prepare your system with:

- python3.
- numpy.
- pandas.
- glob.
- pathlib.
- opencv (cv2).
- matplotlib
- sklearn.
- statsmodels.

Be aware you have the folders: bbox_package, and weeks. And also the .xls file with biomass info per pot. The matching in between pots, row, and stops should be checken ont he wiki page. Yu could also use the dataframe already created.

## Correlate bounding boxes with pot number and biomass data

Use these notebooks for weeks [4](https://github.com/precision-sustainable-ag/OpenCV_Competition2021/blob/master/7_Biomass_Model/VI_calculation_week4.ipynb), [5](https://github.com/precision-sustainable-ag/OpenCV_Competition2021/blob/master/7_Biomass_Model/VI_calculation_week5.ipynb), and [6](https://github.com/precision-sustainable-ag/OpenCV_Competition2021/blob/master/7_Biomass_Model/VI_calculation_week6.ipynb).

## Concatenate dataframes

Use the notebook [concatenate](https://github.com/precision-sustainable-ag/OpenCV_Competition2021/blob/master/7_Biomass_Model/concatenate_h5.ipynb) to put all data into one simple dataframe.

## Understand the behavior of variables, training and testing

Use [this notebook](https://github.com/precision-sustainable-ag/OpenCV_Competition2021/blob/master/7_Biomass_Model/PCA_v2.ipynb) for understanding the nature of variables, training and testing the models. See the description of the models into the wki page.

----

For more details please visit our [wiki](https://github.com/precision-sustainable-ag/OpenCV_Competition2021/wiki/7.-Biomass-estimation-model) page.



